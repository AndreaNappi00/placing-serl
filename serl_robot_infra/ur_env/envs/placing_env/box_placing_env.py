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
        self.force_cost = 0.
        self.upper_bound = -2
        self.lower_bound = -10
    
    def reset(self, **kwargs):
        self.last_action[:] = 0.
        self.announced_goals['forces'] = False
        self.announced_goals['box_position'] = False
        self.announced_goals['ee_box_distance'] = False
        self.force_cost = 0.
        
        return super().reset(**kwargs)
    
    def _update_trajectory_dir(self):
        if self.announced_goals['forces']:
            self.trajectory_dir = np.array([0., 0., 1.]) * 0.01 #move up
    
    def _update_trajectory(self):
        self._update_trajectory_dir()
        target_pos =  self.curr_pos[:3] + self.trajectory_dir
        target_rot = self.curr_pos[3:]
        self.trajectory = np.concatenate([target_pos, target_rot])
        
    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # position
        next_pos = self.curr_pos.copy()
        next_pos[:3] = next_pos[:3] + action[:3] * self.action_scale[0]  + self.trajectory_dir * self.action_scale[0]  ### Comment this when inference
        
        # print("trajectory_dir: ", self.trajectory_dir)
        # print("action: ", action[:3])
        # action[:3] = action[:3] - self.trajectory_dir                   ### Comment this when inference

        next_pos[3:] = (
                R.from_mrp(action[3:6] * self.action_scale[1] / 4.) * R.from_quat(next_pos[3:])
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
            
        if self.pose_est:
            self._update_box_pos_estimate()
            self._update_box_orientation_estimate()
            self._update_trajectory()
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
        cost = 0.
        if self.announced_goals['forces']:
            return self.force_cost - 1
        if obs["state"]["gripper_state"][0] < 0.1:
            return 0.
        for i in range(2):
            if obs["state"]["tcp_force"][i] > self.upper_bound and obs["state"]["tcp_force"][i] < -1:
                cost -= 10 * np.power(obs["state"]["tcp_force"][i] + 4, 2)              # this is pretty useless
            elif obs["state"]["tcp_force"][i] < self.lower_bound:
                cost += 2 * np.power(obs["state"]["tcp_force"][i] + ((self.upper_bound - self.lower_bound)/2 + self.lower_bound), 2)
        # print("force 1:", obs["state"]["tcp_force"][0], "force 2:", obs["state"]["tcp_force"][1], "cost: ", cost)
        return cost
    
    def update_force_goal(self, obs):
        force_goal = obs["state"]["tcp_force"][0] < self.upper_bound and obs["state"]["tcp_force"][1] < self.upper_bound \
            and obs["state"]["tcp_force"][0] > self.lower_bound and obs["state"]["tcp_force"][1] > self.lower_bound and obs["state"]["gripper_state"][0] > 0.1
        if any(obs["state"]["tcp_force"][i] < self.lower_bound for i in range(2)):
            force_goal = False
            self.announced_goals['forces'] = False
        elif (force_goal and not self.announced_goals['forces']):
            self.announced_goals['forces'] = True
            self.force_cost = self.get_force_cost(obs)
    
    def compute_reward(self, obs, action) -> float:
        # huge action gives negative reward (like in mountain car)
        
        # print("action norm:", np.linalg.norm(action[:3]))
        action_cost = 2 * np.sum(np.power(action, 2))
        # print("action_cost: ", action_cost)
        action_diff_cost = 1 * np.sum(np.power(action - self.last_action, 2))    
        # print("action_diff_cost: ", action_diff_cost)
                  
        self.last_action[:] = action
        step_cost = 0.5
        
        if self.announced_goals['forces']:
            suction_reward = 0.
        else:
            suction_reward = 3 * float(obs["state"]["gripper_state"][1] > 0.5)
        suction_cost = 3. * float(obs["state"]["gripper_state"][1] < -0.5)

        pose = obs["state"]["tcp_pose"]
        
        orientation_cost = 1. - sum(obs["state"]["tcp_pose"][3:] * self.curr_reset_pose[3:]) ** 2
        orientation_cost = max(orientation_cost - 0.005, 0.) * 1.
        
        max_pose_diff = 0.05  # set to 5cm
        pos_diff = obs["state"]["tcp_pose"][:2] - self.goal_position[:2]
        position_cost = 5. * np.sum(
            np.where(np.abs(pos_diff) > max_pose_diff, np.abs(pos_diff - np.sign(pos_diff) * max_pose_diff), 0.0)
        )
        
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
            # position_cost=position_cost,
            action_diff_cost=action_diff_cost,
            force_cost=force_cost,
            total_cost=-(-action_cost - step_cost + suction_reward - suction_cost\
                - orientation_cost - action_diff_cost - force_cost)
        )
        for key, info in cost_info.items():
            self.cost_infos[key] = info + (0. if key not in self.cost_infos else self.cost_infos[key])
        self.cost_infos["forces_reached"] = self.announced_goals['forces']
        
        if self.reached_goal_state(obs):
            return 100. - action_cost - orientation_cost - action_diff_cost - force_cost - suction_cost + suction_reward
        else:
            return 0. - action_cost - orientation_cost - suction_cost \
                 - step_cost - action_diff_cost - force_cost + suction_reward

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
            # print("End-effector distance to box reached!")
            self.announced_goals['ee_box_distance'] = True
            
        if box_positon_goal and not self.announced_goals['box_position']:
            # print("Box position reached!")
            self.announced_goals['box_position'] = True
            
        self.update_force_goal(obs)
                 
        # print("gripper_goal: ", gripper_goal, "force_goal: ", self.force_reached, "box_positon_goal: ", box_positon_goal, "ee_box_distance_goal: ", ee_box_distance_goal)
        return self.announced_goals['forces'] \
                and ee_box_distance_goal \
                # and box_positon_goal \
                # and gripper_goal \
                # and height_goal \
                # and orientation_goal \
                # and box_orientation_goal