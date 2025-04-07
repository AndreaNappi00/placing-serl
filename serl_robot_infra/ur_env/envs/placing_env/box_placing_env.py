import numpy as np
from typing import Tuple
import gymnasium as gym
import copy
from scipy.spatial.transform import Rotation as R
import time
import warnings

from ur_env.envs.ur5_env import UR5Env
from ur_env.envs.placing_env.config import UR5PlacingCornerConfig

from franka_env.utils.transformations import construct_homogeneous_matrix, construct_homogeneous_vector

# used for float value comparisons (pressure of vacuum-gripper)
def is_close(value, target):
    return abs(value - target) < 1e-4

def rot_vec_diff(r1, r2):
    """
    Compute the difference between two rotation vectors.
    """
    r1 = R.from_rotvec(r1)
    r2 = R.from_rotvec(r2)
    return (r1.inv() * r2).magnitude()

class BoxPlacingCornerEnv(UR5Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=UR5PlacingCornerConfig)
        
        # Get existing spaces from parent
        # obs_space_definition = dict(self.observation_space.spaces)
        
        # # Add new spaces
        # obs_space_definition["boxes"] = gym.spaces.Sequence(
        #     gym.spaces.Box(-np.inf, np.inf, shape=(6,))
        # )
        # obs_space_definition["trajectory"] = gym.spaces.Box(
        #     -np.inf, np.inf, shape=(7,)
        # )
        
        # # Update observation space with merged spaces
        # self.observation_space = gym.spaces.Dict(obs_space_definition)
        
        self.force_desired = -12
        self.force_tolerance = 3
        self.angle_tolerance = np.pi/3*2
        self.trajectory_dir = np.zeros(6)
    
    def reset(self, **kwargs):
        
        self.last_action[:] = 0.
        self.force_goal_history = np.zeros(10)
        self.low_pass_filter = np.zeros((7, 5))
        
        super_return = super().reset(**kwargs)
        
        time.sleep(1)
        
        if self.pose_est:
            self._update_box_pos_estimate()
            self._update_box_orientation_estimate()
            self._update_box_size_estimate()
            self._update_trajectory_dir()
        
        self.cost_infos['forces'] = False
        self.cost_infos['box_position'] = False
        self.cost_infos['ee_box_distance'] = False
        self.last_box_pose = np.concatenate([self.box_position, self.box_orientation])
        
        return super_return
    
    def goal_reached(self):
        return 'forces' in self.cost_infos and self.cost_infos['forces'] and \
            'box_pose' in self.cost_infos and self.cost_infos['box_pose'] and \
            not self.gripper_state[1]
    
    def _update_trajectory_dir(self):
        scaling = 2
        T_o_r = construct_homogeneous_matrix(self.curr_reset_pose)
        T_r_goal = construct_homogeneous_matrix(self.goal_pose)
        T_o_goal =  T_o_r @ T_r_goal
        goal_pose_o = np.concatenate([T_o_goal[:3, 3], R.from_matrix(T_o_goal[:3, :3]).as_rotvec()])

        if self.goal_reached(): #move up slowly
            self.trajectory_dir[:3] = np.array([0., 0., 1.]) * (1/scaling)
        elif self.gripper_state[0]:   # go to goal
            self.trajectory_dir[:3] = goal_pose_o[:3] - (self.box_position + self.box_error)
            self.trajectory_dir[2] = 0.
            if np.linalg.norm(self.trajectory_dir[:3]) > 0.025:
                self.trajectory_dir[:3] /= np.linalg.norm(self.trajectory_dir[:3])
            self.trajectory_dir[:3] *= (1/scaling)
        else:   # box dropped # go to box
            # It is assumed that self.curr_reset_pose is defined (for instance, saved during reset)
            target_picking = self.box_position + self.box_error
            target_picking[2] += 0.14 + self.box_size / 2

            # Compute the total horizontal displacement from the starting position to the target
            horizontal_total = 0.1
            # Compute the horizontal progress made so far
            horizontal_progress = np.linalg.norm(self.curr_pos[:2] - target_picking[:2])
            s = np.clip(horizontal_progress / horizontal_total, 0, 1) if horizontal_total > 0 else 0

            # Define a parabolic offset that is zero at s=0 and s=1 and peaks at s=0.5
            H = 0.05  # Maximum additional height (adjustable)
            parabolic_offset = 4 * H * s * (1 - s)

            # Compute the first three components of the trajectory toward the target
            traj = target_picking - self.curr_pos[:3]
            # Overwrite the z component to include the parabolic behavior
            traj[2] = (target_picking[2] - self.curr_pos[2]) + parabolic_offset

            self.trajectory_dir[:3] = traj / (np.linalg.norm(traj) * scaling)

        self.trajectory_dir[3:] = (
            R.from_rotvec(goal_pose_o[3:]) * R.from_rotvec(-self.box_orientation)
        ).as_rotvec()
        
    def update_last_box_pose(self):
        self.last_box_pose = self.box_pose.copy()

    def has_box_moved(self):
        # Check if the box has moved significantly
        position_diff = np.linalg.norm(self.box_pose[:3] - self.last_box_pose[:3])
        rotation_diff = rot_vec_diff(self.box_pose[3:], self.last_box_pose[3:])
        
        # print(f"position_diff: {position_diff}, rotation_diff: {rotation_diff}")
        
        box_moved = position_diff > 0.01 \
            or rotation_diff > 0.1
        return box_moved
                
    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        next_pos = self.curr_pos.copy()
                
        # position
        if self.low_pass_filter_k:
            self.low_pass_filter[:, :-1] = self.low_pass_filter[:, 1:]
            self.low_pass_filter[:, -1] = action
            action_filtered = np.mean(self.low_pass_filter, axis=1)
        else:
            action_filtered = action
            
        next_pos[:3] = next_pos[:3] + (action_filtered[:3] + self.trajectory_dir[:3]) * self.action_scale[0]
        # next_pos[:3] = next_pos[:3] + (self.trajectory_dir[:3]) * self.action_scale[0]
        
        self.cost_infos["intervene_action"] = action

        # orientation
        next_pos[3:] = (
            R.from_mrp(action_filtered[3:6] * self.action_scale[1] / 4.) \
            * R.from_rotvec(self.trajectory_dir[3:] / 50.) *  R.from_quat(next_pos[3:])
        ).as_quat()             # c * r  --> applies c after r

        gripper_action = action[6] * self.action_scale[2]

        safe_pos = self.clip_safety_box(next_pos)
        self._send_pos_command(safe_pos)
        self._send_gripper_command(gripper_action)

        self.curr_path_length += 1

        obs = self._get_obs(action)
        
        self.box_pose = np.concatenate([self.box_position, self.box_orientation])
        
        # print(f"box_pose: {self.box_pose}")
        
        truncated = self._is_truncated()

        dt = time.time() - start_time
        to_sleep = max(0, (1.0 / self.hz) - dt)
        if to_sleep == 0:
            warnings.warn(f"environment could not be within {self.hz} Hz, took {dt:.4f}s!")
        time.sleep(to_sleep)

        return obs, 0, 0, truncated, {}

    def _get_obs(self, action) -> dict:
        # get image before state observation, so they match better in time

        images = None
        if self.camera_mode is not None:
            images = self.get_image()
            
        self._update_currpos()
        
        if self.pose_est:
            self._update_box_pos_estimate()
            self._update_box_orientation_estimate()
            self._update_box_size_estimate()
            self._update_trajectory_dir()
        else:
            self.box_position = np.array([0.5, 0.5, 0.5])
            self.box_orientation = np.array([0., 0., 0.])
            self.trajectory_dir = np.zeros(6)
                
        state_observation = {
            "tcp_pose": self.curr_pos,
            "tcp_vel": self.curr_vel,
            "gripper_state": self.gripper_state,
            "tcp_force": self.curr_force,
            "tcp_torque": self.curr_torque,
            "action": action,
            "boxes": np.concatenate([self.box_position, self.box_orientation]), # in robot_base frame
            "trajectory": self.trajectory_dir,
            "goal_pose": np.zeros(6)
        }

        if images is not None:
            return copy.deepcopy(dict(images=images, state=state_observation))
        else:
            return copy.deepcopy(dict(state=state_observation))
        
    def get_force_cost(self, obs):  #overloaded
        if self.announced_goals['forces']:
            return 0.
        alpha_reward = 10
        alpha_cost = 0.1
        delta_err_x = self.force_desired - obs["state"]["tcp_force"][0]
        delta_err_y = self.force_desired - obs["state"]["tcp_force"][1]
        
        reward =  alpha_reward * np.exp(- (delta_err_x ** 2 + delta_err_y ** 2) / (self.force_tolerance ** 2))
        magnitude_cost = alpha_cost * np.power(np.max([0, np.linalg.norm(obs["state"]["tcp_force"]) - (np.abs(self.force_desired) + self.force_tolerance)]), 2)
        direction = obs["state"]["tcp_force"] / np.linalg.norm(obs["state"]["tcp_force"])
        direction_cost = alpha_cost * np.power(np.max([0, np.cos(self.angle_tolerance) - np.dot(obs["state"]["tcp_force"], direction) / (np.linalg.norm(obs["state"]["tcp_force"] * 1e-6))]), 2)
        z_cost = alpha_cost * np.power(np.max([0, np.abs(obs["state"]["tcp_force"][2]) - 20]), 2)
        cost = magnitude_cost + z_cost
        
        return cost - reward
    
    def update_force_goal(self, obs):   #overloaded
        forces = obs["state"]["tcp_force"][:2]
        gripper_active = obs["state"]["gripper_state"][1] > 0.5
        release_action = obs["state"]["action"][-1] < -0.5

        force_goal = all(
            np.abs(self.force_desired) - self.force_tolerance < np.abs(force)
            for force in forces
        )
        
        if self.announced_goals['forces'] and release_action:
            pass
        elif force_goal or sum(self.force_goal_history) > 3:
            self.announced_goals['forces'] = True
        elif gripper_active:
            self.announced_goals['forces'] = False
        self.force_goal_history = np.roll(self.force_goal_history, 1)
        self.force_goal_history[0] = force_goal
        
    def update_box_pose_goal(self, obs):    #overloaded
        angle_diff = (R.from_rotvec(obs["state"]["boxes"][3:]).inv() * R.from_rotvec(self.target_orientation)).magnitude()
        pos_diff = np.linalg.norm(obs["state"]["boxes"][:2] - self.goal_pose[:2])
        z_diff = np.abs(obs["state"]["boxes"][2] - self.goal_pose[2])
        gripper_state = obs["state"]["gripper_state"][1]

        pose_in_goal = pos_diff < 0.05 and z_diff < 0.02 and angle_diff < 0.15

        if pose_in_goal:
            self.announced_goals['box_pose'] = True
        elif self.announced_goals['box_pose']:
            if not pose_in_goal:
                self.announced_goals['box_pose'] = False
    
    def clip_costs(self):
        if "orientation_cost" in self.cost_infos:
            self.cost_infos["orientation_cost"] = min(50., self.cost_infos["orientation_cost"])
        if "orientation_cost_box" in self.cost_infos:
            self.cost_infos["orientation_cost_box"] = min(50., self.cost_infos["orientation_cost_box"])
        # if "suction_cost" in self.cost_infos:
        #     self.cost_infos["suction_cost"] = min(50., self.cost_infos["suction_cost"])
        if "force_cost" in self.cost_infos:
            self.cost_infos["force_cost"] = min(50., self.cost_infos["force_cost"])
    
    def compute_reward(self, obs, action) -> float: #overloaded
        
        # huge action gives negative reward (like in mountain car)
        norm_action = np.linalg.norm(action[:3])
        if norm_action > 0.:
            action_cost = 0.2 * (1 - np.dot(action[:3]/norm_action, self.trajectory_dir[:3]/np.linalg.norm(self.trajectory_dir[:3])))
        else:
            action_cost = 0
        action_diff_cost = 0.2 * np.sum(np.power(action - self.last_action, 2))    #0.2

        self.last_action[:] = action
        step_cost = 0.05
        
        gripper_release_cost = 0
        if obs["state"]["gripper_state"][1] and action[-1] < -0.5 and not self.announced_goals['forces']:
            gripper_release_cost = 50
        
        suction_cost = 0
        suction_reward = 0
        if self.announced_goals['forces'] and self.announced_goals['box_pose']:
            suction_reward = 2 * float(action[-1] < -0.5)
        elif obs["state"]["gripper_state"][1] > 0.5:
            suction_reward = 2
        else:
            suction_cost = 1 * float(action[-1] > 0.5)
        
        relative_rotation = R.from_quat(obs["state"]["tcp_pose"][3:]) * R.from_quat(self.curr_reset_pose[3:]).inv()
        angle = relative_rotation.magnitude()
        orientation_cost = max(angle - 0.005, 0.) * 1.
        
        max_pose_diff = 0.05  # set to 5cm
        pos_diff = obs["state"]["boxes"][:3] - self.goal_pose[:3]
        position_cost = 5. * np.sum(
            np.where(np.abs(pos_diff) > max_pose_diff, np.abs(pos_diff - np.sign(pos_diff) * max_pose_diff), 0.0)
        )
        
        orientation_cost_box = 1. - sum(R.from_rotvec(obs["state"]["boxes"][3:]).as_quat() * R.from_rotvec(self.target_orientation).as_quat()) ** 2
        orientation_cost_box = max(orientation_cost_box - 0.005, 0.) * 1.
        
        max_height_diff = 0.05  # set to 10cm
        height_diff = obs["state"]["tcp_pose"][2] - self.goal_pose[2] - 0.18
        # position_cost += 10. * height_diff if height_diff > max_height_diff else 0.
        
        force_cost = self.get_force_cost(obs)
        self.update_force_goal(obs)
        self.update_box_pose_goal(obs)
        
        cost_info = dict(
            action_cost=action_cost,
            step_cost=step_cost,
            suction_reward=suction_reward,
            suction_cost=suction_cost,
            orientation_cost=orientation_cost,
            orientation_cost_box=orientation_cost_box,
            position_cost=position_cost,
            action_diff_cost=action_diff_cost,
            force_cost=force_cost,
            gripper_release_cost=gripper_release_cost,
            total_reward=-action_cost - step_cost + suction_reward - suction_cost \
                - orientation_cost - action_diff_cost - force_cost - gripper_release_cost + orientation_cost_box
        )
        for key, info in cost_info.items():
            self.cost_infos[key] = info + (0. if key not in self.cost_infos else self.cost_infos[key])
        for key, info in self.announced_goals.items():
            self.cost_infos[key] = info
        
        self.clip_costs()
        
        if self.reached_goal_state(obs):
            self.config.SUCCESS_COUNT += 1
            return 300. - action_cost - orientation_cost - action_diff_cost - force_cost - position_cost\
                - suction_cost + suction_reward - orientation_cost_box - gripper_release_cost
        else:
            return 0. - action_cost - orientation_cost - suction_cost - position_cost\
                - step_cost - action_diff_cost - force_cost + suction_reward\
                - orientation_cost_box - gripper_release_cost

    def reached_goal_state(self, obs) -> bool:  #overloaded           
        ee_box_distance_goal = np.linalg.norm(obs["state"]["tcp_pose"][2] - obs["state"]["boxes"][2]) > 0.25        
        
        if ee_box_distance_goal and not self.announced_goals['ee_box_distance']:
            self.announced_goals['ee_box_distance'] = True
                 
        return self.announced_goals['forces'] \
                and ee_box_distance_goal \
                and self.announced_goals['box_position']