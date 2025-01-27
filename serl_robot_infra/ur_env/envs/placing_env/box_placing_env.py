import numpy as np
from typing import Tuple
import gymnasium as gym
import copy
from scipy.spatial.transform import Rotation as R

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
        
        self.observation_space["state"]["boxes"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        
        self.observation_space["state"]["trajectory"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(7,)
        )
        
        self.announced_goals = {
            'box_position': False,
            'ee_box_distance': False,
            'forces': False,
        }
        self.force_cost = 0.
    
    def reset(self, **kwargs):
        self.last_action[:] = 0.
        self.announced_goals['forces'] = False
        self.announced_goals['box_position'] = False
        self.announced_goals['ee_box_distance'] = False
        self.force_cost = 0.
        
        return super().reset(**kwargs)
        
    def update_trajectory(self):
        target_pos =  self.curr_pos[:3] + self.trajectory_dir
        target_rot = self.curr_pos[3:]
        self.trajectory = np.concatenate([target_pos, target_rot])

    def _get_obs(self, action) -> dict:
        # get image before state observation, so they match better in time

        images = None
        if self.camera_mode is not None:
            images = self.get_image()
            
        if self.pose_est:
            self._update_box_pos_estimate()
            self._update_box_orientation_estimate()
            self.update_trajectory()
        else:
            self.box_position = np.array([0.5, 0.5, 0.5])
            self.box_orientation = np.array([0., 0., 0.])
            self.trajectory = np.array([0., 0., 0., 0., 0., 0., 0.])
            
        self._update_currpos()
        state_observation = {
            "tcp_pose": self.curr_pos,
            "tcp_vel": self.curr_vel,
            "gripper_state": self.gripper_state,
            "tcp_force": self.curr_force,
            "tcp_torque": self.curr_torque,
            "action": action,
            "boxes": np.concatenate([self.box_position, R.from_rotvec(self.box_orientation).as_mrp()]), # in robot_base frame
            "trajectory": self.trajectory
        }

        if images is not None:
            return copy.deepcopy(dict(images=images, state=state_observation))
        else:
            return copy.deepcopy(dict(state=state_observation))
        
    def get_force_cost(self, obs):
        return 2 * np.sum(np.power(obs["state"]["tcp_force"] + 4, 2))
    
    def update_force_goal(self, obs):
        force_goal = obs["state"]["tcp_force"][0] < -2 and obs["state"]["tcp_force"][1] < -2 \
            and ((obs["state"]["tcp_force"][0] > -5 and obs["state"]["tcp_force"][1] > -5) and obs["state"]["gripper_state"][1] > 0.5)
        if (force_goal and not self.announced_goals['forces']):
            self.announced_goals['forces'] = True
            self.force_cost = self.get_force_cost(obs)
            print("Force reached!")
    
    def compute_reward(self, obs, action) -> float:
        # huge action gives negative reward (like in mountain car)
        action_cost = 1 * np.sum(np.power(action, 2))
        action_diff_cost = 2 * np.sum(np.power(obs["state"]["action"] - self.last_action, 2))
        self.last_action[:] = action
        step_cost = 0.1
        
        suction_reward = 3 * float(obs["state"]["gripper_state"][1] > 0.5)
        suction_cost = 3. * float(obs["state"]["gripper_state"][1] < -0.5)

        pose = obs["state"]["tcp_pose"]
        
        orientation_cost = 1. - sum(obs["state"]["tcp_pose"][3:] * self.curr_reset_pose[3:]) ** 2
        orientation_cost = max(orientation_cost - 0.005, 0.) * 5.
        
        max_pose_diff = 0.05  # set to 5cm
        pos_diff = obs["state"]["tcp_pose"][:2] - self.goal_position[:2]
        position_cost = 5. * np.sum(
            np.where(np.abs(pos_diff) > max_pose_diff, np.abs(pos_diff - np.sign(pos_diff) * max_pose_diff), 0.0)
        )
        
        max_height_diff = 0.05  # set to 10cm
        height_diff = obs["state"]["tcp_pose"][2] - self.goal_position[2] - 0.18
        # position_cost += 10. * height_diff if height_diff > max_height_diff else 0.
        
        force_cost = self.get_force_cost(obs)
        if self.announced_goals['forces']:
            force_cost = self.force_cost
        # print("forces: ", obs["state"]["tcp_force"])
        
        
        cost_info = dict(
            action_cost=action_cost,
            step_cost=step_cost,
            suction_reward=suction_reward,
            suction_cost=suction_cost,
            orientation_cost=orientation_cost,
            position_cost=position_cost,
            action_diff_cost=action_diff_cost,
            force_cost=force_cost,
            total_cost=-(-action_cost - step_cost + suction_reward - suction_cost \
                - orientation_cost - position_cost - action_diff_cost - force_cost)
        )
        for key, info in cost_info.items():
            self.cost_infos[key] = info + (0. if key not in self.cost_infos else self.cost_infos[key])
        cost_info["forces_reached"] = self.announced_goals['forces']
        
        if self.reached_goal_state(obs):
            return 100. - action_cost - orientation_cost - position_cost - action_diff_cost - force_cost
        else:
            return 0. + suction_reward - action_cost - orientation_cost - position_cost - \
                suction_cost - step_cost - action_diff_cost - force_cost

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
        box_positon_goal = np.linalg.norm(obs["state"]["boxes"][:3] - goal[:3]) < 0.05
        # print("box pos: ", obs["state"]["boxes"][:3], "reached?: ", box_positon_goal, "error?: ", np.linalg.norm(obs["state"]["boxes"][:3] - goal[:3]))
        # print("force: ", obs["state"]["tcp_force"], "reached?: ", force_goal)
        box_orientation_goal = sum(obs["state"]["boxes"][3:] * np.array([0, 0, 1])) ** 2 > 0.9
        ee_box_distance_goal = np.linalg.norm(obs["state"]["tcp_pose"][2] - obs["state"]["boxes"][2]) > 0.2

        # print(f"Force goal: {force_goal}, Height goal: {height_goal}, Gripper goal: {gripper_goal}")
        # print(f"force: {obs['state']['tcp_force']}")
        
        
        if ee_box_distance_goal and not self.announced_goals['ee_box_distance']:
            self.announced_goals['ee_box_distance'] = True
            print("End-effector distance to box reached!")
            
        if box_positon_goal and not self.announced_goals['box_position']:
            print("Box position reached!")
            self.announced_goals['box_position'] = True
                 
        # print("gripper_goal: ", gripper_goal, "force_goal: ", self.force_reached, "box_positon_goal: ", box_positon_goal, "ee_box_distance_goal: ", ee_box_distance_goal)
        return self.announced_goals['forces'] \
                and ee_box_distance_goal \
                # and box_positon_goal \
                # and gripper_goal \
                # and height_goal \
                # and orientation_goal \
                # and box_orientation_goal