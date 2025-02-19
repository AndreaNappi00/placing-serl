import numpy as np
from typing import Tuple
import gymnasium as gym
import copy
from scipy.spatial.transform import Rotation as R
import time
import warnings

from ur_env.envs.ur5_env import UR5Env
from ur_env.envs.placing_env.config import UR5PlacingCornerConfig


# used for float value comparisons (pressure of vacuum-gripper)
def is_close(value, target):
    return abs(value - target) < 1e-4

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
        
        self.announced_goals = {
            'box_position': False,
            'ee_box_distance': False,
            'forces': False,
        }
        self.upper_bound = -8
        self.lower_bound = -18
        self.force_desired = -13
        self.force_tolerance = 4
        self.angle_tolerance = np.pi/3*2
    
    def reset(self, **kwargs):
        
        self.last_action[:] = 0.
        self.announced_goals['forces'] = False
        self.announced_goals['box_position'] = False
        self.announced_goals['ee_box_distance'] = False
        self.low_pass_filter = np.zeros((7, 5))
        
        if self.pose_est:
            self.trajectory_dir = np.zeros(6)
            self._update_box_pos_estimate()
            self.goal_direction = self.goal_position - (self.box_position + self.box_error)
            self.goal_direction[2] = 0.
            self.goal_direction /= np.linalg.norm(self.goal_direction) * 2
        
        return super().reset(**kwargs)
    
    def _update_trajectory_dir(self):
        if self.announced_goals['forces']: #move up slowly
            if self.gripper_state[1]:
                self.trajectory_dir = np.zeros(6)
            else: # move up
                self.trajectory_dir[:3] = np.array([0., 0., 1.]) * (1/3)
        elif self.gripper_state[0]:   # go to goal
            self.trajectory_dir[:3] = self.goal_position + 0.002 * np.random.randint(-10*np.ones(3), 10*np.ones(3)) - (self.box_position + self.box_error)
            self.trajectory_dir[2] = 0.
            self.trajectory_dir[:3] /= np.linalg.norm(self.trajectory_dir[:3]) * 3
        else:   # box dropped # go to box
            target_picking = self.box_position + self.box_error
            target_picking[2] = self.goal_position[2] + 0.18
            self.trajectory_dir[:3] = target_picking - self.curr_pos[:3]
            self.trajectory_dir[:3] /= np.linalg.norm(self.trajectory_dir[:3]) * 3
        
        self.trajectory_dir[3:] = (
            R.from_rotvec(self.target_orientation) * R.from_rotvec(-self.box_orientation)
        ).as_rotvec()
                
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
        # next_pos[:3] = next_pos[:3] + (self.trajectory_dir) * self.action_scale[0]
        
        self.cost_infos["intervene_action"] = action

        # orientation
        next_pos[3:] = (
            R.from_mrp(action_filtered[3:6] * self.action_scale[1] / 4.) \
            * R.from_rotvec(self.trajectory_dir[3:] / 10.) *  R.from_quat(next_pos[3:])
        ).as_quat()             # c * r  --> applies c after r

        gripper_action = action[6] * self.action_scale[2]

        safe_pos = self.clip_safety_box(next_pos)
        self._send_pos_command(safe_pos)
        self._send_gripper_command(gripper_action)

        self.curr_path_length += 1

        obs = self._get_obs(action)

        reward = self.compute_reward(obs, action)
        # print("reward: ", reward)
        truncated = self._is_truncated()
        reward = reward if not truncated else reward - 10.  # truncation penalty
        # print("truncated reward: ", reward)
        done = self.curr_path_length >= self.max_episode_length or self.reached_goal_state(obs) or truncated

        dt = time.time() - start_time
        to_sleep = max(0, (1.0 / self.hz) - dt)
        if to_sleep == 0:
            warnings.warn(f"environment could not be within {self.hz} Hz, took {dt:.4f}s!")
        time.sleep(to_sleep)

        return obs, reward, done, truncated, self.get_cost_infos(done)

    def _get_obs(self, action) -> dict:
        # get image before state observation, so they match better in time

        images = None
        if self.camera_mode is not None:
            images = self.get_image()
            
        self._update_currpos()
        
        if self.pose_est:
            self._update_box_pos_estimate()
            self._update_box_orientation_estimate()
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
            "trajectory": self.trajectory_dir
        }

        if images is not None:
            return copy.deepcopy(dict(images=images, state=state_observation))
        else:
            return copy.deepcopy(dict(state=state_observation))
        
    # def get_force_cost(self, obs):
    #     cost = 0.
    #     if self.announced_goals['forces']:
    #         return self.force_cost - 1
    #     if obs["state"]["gripper_state"][0] < 0.1:
    #         return 0.
    #     for i in range(2):
    #         if obs["state"]["tcp_force"][i] > 0:
    #             cost += 10 * np.power(obs["state"]["tcp_force"][i], 2)
    #         elif obs["state"]["tcp_force"][i] > self.upper_bound and obs["state"]["tcp_force"][i] < -1:
    #             cost -= 10 * np.power(obs["state"]["tcp_force"][i] + 4, 2)              # this is pretty useless
    #         elif obs["state"]["tcp_force"][i] < self.lower_bound:
    #             cost += 2 * np.power(obs["state"]["tcp_force"][i] + ((self.upper_bound - self.lower_bound)/2 + self.lower_bound), 2)
    #     # print("force 1:", obs["state"]["tcp_force"][0], "force 2:", obs["state"]["tcp_force"][1], "cost: ", cost)
    #     return cost
    def get_force_cost(self, obs):
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
        z_cost = alpha_cost * np.power(np.max([0, np.abs(obs["state"]["tcp_force"][2]) - 5]), 2)
        cost = magnitude_cost + direction_cost + z_cost
        
        return cost - reward
    
    def update_force_goal(self, obs):
        force_goal = obs["state"]["tcp_force"][0] < self.upper_bound and obs["state"]["tcp_force"][1] < self.upper_bound \
            and obs["state"]["tcp_force"][0] > self.lower_bound and obs["state"]["tcp_force"][1] > self.lower_bound and obs["state"]["gripper_state"][0] > 0.1
        if any(obs["state"]["tcp_force"][i] < self.lower_bound for i in range(2)):
            force_goal = False
            self.announced_goals['forces'] = False
        elif (force_goal and not self.announced_goals['forces']):
            self.announced_goals['forces'] = True
    
    def clip_costs(self):
        if "orientation_cost" in self.cost_infos:
            self.cost_infos["orientation_cost"] = min(50., self.cost_infos["orientation_cost"])
        if "orientation_cost_box" in self.cost_infos:
            self.cost_infos["orientation_cost_box"] = min(50., self.cost_infos["orientation_cost_box"])
        if "suction_cost" in self.cost_infos:
            self.cost_infos["suction_cost"] = min(50., self.cost_infos["suction_cost"])
        if "force_cost" in self.cost_infos:
            self.cost_infos["force_cost"] = min(50., self.cost_infos["force_cost"])
    
    def compute_reward(self, obs, action) -> float:
        
        # huge action gives negative reward (like in mountain car)
        delta_pos = action[:3]/2 - self.trajectory_dir[:3]
        action_cost = 0.2 * np.sum(np.power(delta_pos, 2))  #0.2
        action_diff_cost = 0.2 * np.sum(np.power(action - self.last_action, 2))    #0.2
                  
        self.last_action[:] = action
        step_cost = 0.05
        
        gripper_release_cost = 0
        if obs["state"]["gripper_state"][1] and action[-1] < -0.5 and not self.announced_goals['forces']:
            gripper_release_cost = 10
        
        suction_cost = 0
        suction_reward = 0
        if self.announced_goals['forces']:
            suction_reward = 1 * float(action[6] < -0.5)
        elif obs["state"]["gripper_state"][1] > 0.5:
            suction_reward = 1
        else:
            suction_cost = 2 * float(action[6] > 0.5)

        pose = obs["state"]["tcp_pose"]
        
        orientation_cost = 1. - sum(obs["state"]["tcp_pose"][3:] * self.curr_reset_pose[3:]) ** 2
        orientation_cost = max(orientation_cost - 0.005, 0.) * 1.
        
        max_pose_diff = 0.05  # set to 5cm
        pos_diff = obs["state"]["tcp_pose"][:2] - self.goal_position[:2]
        position_cost = 5. * np.sum(
            np.where(np.abs(pos_diff) > max_pose_diff, np.abs(pos_diff - np.sign(pos_diff) * max_pose_diff), 0.0)
        )
        
        orientation_cost_box = 1. - sum(R.from_rotvec(obs["state"]["boxes"][3:]).as_quat() * R.from_rotvec(self.target_orientation).as_quat()) ** 2
        orientation_cost_box = max(orientation_cost_box - 0.005, 0.) * 1.
        
        max_height_diff = 0.05  # set to 10cm
        height_diff = obs["state"]["tcp_pose"][2] - self.goal_position[2] - 0.18
        # position_cost += 10. * height_diff if height_diff > max_height_diff else 0.
        
        force_cost = self.get_force_cost(obs)
        # print("forces: ", obs["state"]["tcp_force"])
        
        
        cost_info = dict(
            action_cost=action_cost,
            step_cost=step_cost,
            suction_reward=suction_reward,
            suction_cost=suction_cost,
            orientation_cost=orientation_cost,
            orientation_cost_box=orientation_cost_box,
            # position_cost=position_cost,
            action_diff_cost=action_diff_cost,
            force_cost=force_cost,
            gripper_release_cost=gripper_release_cost,
            total_cost=-(-action_cost - step_cost + suction_reward - suction_cost\
                - orientation_cost - action_diff_cost - force_cost - gripper_release_cost + orientation_cost_box)
        )
        for key, info in cost_info.items():
            self.cost_infos[key] = info + (0. if key not in self.cost_infos else self.cost_infos[key])
        for key, info in self.announced_goals.items():
            self.cost_infos[key] = info
        
        self.clip_costs()
        
        if self.reached_goal_state(obs):
            return 200. - action_cost - orientation_cost - action_diff_cost - force_cost \
                - suction_cost + suction_reward - orientation_cost_box - gripper_release_cost
        else:
            return 0. - action_cost - orientation_cost - suction_cost \
                - step_cost - action_diff_cost - force_cost + suction_reward\
                - orientation_cost_box - gripper_release_cost

    def reached_goal_state(self, obs) -> bool:
        # obs[0] == gripper pressure, obs[4] == force in Z-axis
        state = obs["state"]
        goal = self.goal_position
        # return 0.1 < state['gripper_state'][0] < 0.85 and np.linalg.norm(state['tcp_pose'][:2] - goal[:2]) < 0.01 and state['tcp_pose'][2] < 0.14
        
        # print("0:", obs["state"]["gripper_state"][0], "1:", obs["state"]["gripper_state"][1])
        # 0 is the pressure of the vacuum gripper, 1 is 1 if active and grasping, -1 if active and not grasping
        
        height_goal = state['tcp_pose'][2] < goal[2] + 0.18
        gripper_goal = 0.1 < state['gripper_state'][0] < 0.85
        orientation_goal = sum(obs["state"]["tcp_pose"][3:] * self.curr_reset_pose[3:]) ** 2 > 0.85
        self.announced_goals['box_position'] = np.linalg.norm(obs["state"]["boxes"][:3] - goal[:3]) < 0.1
        # print("box pos: ", obs["state"]["boxes"][:3], "reached?: ", box_positon_goal, "error?: ", np.linalg.norm(obs["state"]["boxes"][:3] - goal[:3]))
        # print("tcp pos: ", obs["state"]["tcp_pose"][:3])
        # print("force: ", obs["state"]["tcp_force"], "reached?: ", self.announced_goals['forces'])
        # box_orientation_goal = sum(obs["state"]["boxes"][3:] * np.array([0, 0, 1])) ** 2 > 0.9            #this is wrong, it comes as mrp
        ee_box_distance_goal = np.linalg.norm(obs["state"]["tcp_pose"][2] - obs["state"]["boxes"][2]) > 0.2

        # print(f"Force goal: {force_goal}, Height goal: {height_goal}, Gripper goal: {gripper_goal}")
        # print(f"force: {obs['state']['tcp_force']}")
        
        
        if ee_box_distance_goal and not self.announced_goals['ee_box_distance']:
            # print("End-effector distance to box reached!")
            self.announced_goals['ee_box_distance'] = True
            
        self.update_force_goal(obs)
                 
        # print("gripper_goal: ", gripper_goal, "force_goal: ", self.force_reached, "box_positon_goal: ", box_positon_goal, "ee_box_distance_goal: ", ee_box_distance_goal)
        return self.announced_goals['forces'] \
                and ee_box_distance_goal \
                and self.announced_goals['box_position'] \
                # and gripper_goal \
                # and height_goal \
                # and orientation_goal \
                # and box_orientation_goal