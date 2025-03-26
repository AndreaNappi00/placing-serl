from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
from gym import Env
from franka_env.utils.transformations import (
    construct_adjoint_matrix,
    construct_adjoint_matrix_inverse,
    construct_homogeneous_matrix,
    construct_rotation_matrix,
    construct_homogeneous_vector,
    invert_homogeneous_matrix,
    rotate_rotvec
)

class RelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base UR5 environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,)), # xyz + rotvec
                "tcp_force": spaces.Box(-np.inf, np.inf, shape=(3,)), # xyz
                "tcp_torque": spaces.Box(-np.inf, np.inf, shape=(3,)), # xyz
                "gripper_state": spaces.Box(-np.inf, np.inf, shape=(2,)),
                "boxes": spaces.Box(-np.inf, np.inf, shape=(6,)), # xyz + rotvec
                "trajectory": spaces.Box(-np.inf, np.inf, shape=(6,)), # xyz + rotvec
            }
        ),
        ......
    }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...)
    """

    def __init__(self, env: Env):
        super().__init__(env)

        # Homogeneous transformation matrix from reset pose's relative frame to base frame
        self.T_r_o = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        # print("action before transf", action)
        transformed_action = self.transform_action(action) #action in network will be in base frame!!!!????
        # print("transformed_action", transformed_action)
        obs, reward, done, truncated, info = self.env.step(transformed_action) #go deeper, to spacemouse env or ur5 env

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            # print("intervene_action", info["intervene_action"]) # in base frame
            info["intervene_action"] = self.transform_action_inv(info["intervene_action"])
            # print("intervene_action transformed", info["intervene_action"]) # in end-effector frame

        # Update rotation matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        # print("action in transformed obs", transformed_obs["state"]["action"]) # in end-effector frame
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # obs['state']['tcp_pose'][:2] -= info['reset_shift']  # set rel pose to original reset pose (no random)

        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])
        
        # Update transformation matrix from the reset pose's relative frame to base frame
        self.T_r_o = np.linalg.inv(
            construct_homogeneous_matrix(obs["state"]["tcp_pose"])
        )

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the rotation and homogeneous matrix
        """
        
        A_b_ee = self.adjoint_matrix
        # A_b_ee_inv = np.linalg.inv(A_b_ee)
        obs["state"]["tcp_vel"] = A_b_ee @ obs["state"]["tcp_vel"]
        
        wrench_b = np.concatenate((obs["state"]["tcp_force"], obs["state"]["tcp_torque"]))
        wrench_ee = A_b_ee.T @ wrench_b
        obs["state"]["tcp_force"] = wrench_ee[:3]
        obs["state"]["tcp_torque"] = wrench_ee[3:]
            
        T_o_ee = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
        T_r_ee = self.T_r_o @ T_o_ee
        self.T_ee_o = invert_homogeneous_matrix(T_o_ee)

        # Reconstruct transformed tcp_pose vector
        p_r_ee = T_r_ee[:3, 3]
        theta_r_ee = R.from_matrix(T_r_ee[:3, :3]).as_quat()
        obs["state"]["tcp_pose"] = np.concatenate((p_r_ee, theta_r_ee))
        
        T_ee_goal = invert_homogeneous_matrix(T_r_ee) @ construct_homogeneous_matrix(self.unwrapped.goal_pose)
        p_ee_goal = T_ee_goal[:3, 3]
        theta_ee_goal = R.from_matrix(T_ee_goal[:3, :3]).as_rotvec()
        obs["state"]["goal_pose"] = np.concatenate((p_ee_goal, theta_ee_goal))
        
        obs["state"]["action"][:6] = self.transform_action_inv(obs["state"]["action"][:6])
        
        if self.unwrapped.config.POSE_ESTIMATION:
            obs["state"]["boxes"][:3] = (self.T_ee_o @ construct_homogeneous_vector(obs["state"]["boxes"][:3]))[:3]
            obs["state"]["boxes"][3:6] = (R.from_matrix(self.T_ee_o[:3, :3]) * R.from_rotvec(obs["state"]["boxes"][3:6])).as_rotvec()
            obs["state"]["trajectory"] = A_b_ee @ obs["state"]["trajectory"]
        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into spatial(base) frame
        using the rotation matrix
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the rotation matrix.
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = self.adjoint_matrix @ action[:6]
        return action

class RelativeReward(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)

        self.last_action = np.zeros(7)
        self.announced_goals = {
            'box_pose': False,
            'ee_box_distance': False,
            'forces': False,
        }
    
    def step(self, action: np.array):
        # print("action", action)
        obs, reward, done, truncated, info = self.env.step(action) #go deeper, to spacemouse env or ur5 env
        
        reward = self.compute_reward(obs)
        
        reward = reward if not truncated else reward - 100.  # truncation penalty
        done = self.unwrapped.curr_path_length >= self.unwrapped.max_episode_length or self.reached_goal_state(obs) or truncated
                
        new_info = self.unwrapped.get_cost_infos(done)
        new_info.update(info)
        
        return obs, reward, done, truncated, new_info
    
    def update_box_pose_goal(self, obs):        
        angle_diff = (R.from_rotvec(obs["state"]["boxes"][3:]).inv() * R.from_rotvec(obs["state"]["goal_pose"][3:])).magnitude()
        pos_diff = np.linalg.norm(obs["state"]["boxes"][:2] - obs["state"]["goal_pose"][:2])
        z_diff = np.abs(obs["state"]["boxes"][2] - obs["state"]["goal_pose"][2])

        pose_in_goal = pos_diff < 0.05 and z_diff < 0.04 and angle_diff < 0.15
        if pose_in_goal:
            self.announced_goals['box_pose'] = True
        elif self.announced_goals['box_pose']:
            if not pose_in_goal:
                self.announced_goals['box_pose'] = False

    def compute_reward(self, obs) -> float:
        action = obs["state"]["action"]
        # huge action gives negative reward (like in mountain car)
        norm_action = np.linalg.norm(action[:3])
        if norm_action > 0.:
            action_cost = 0.2 * (1 - np.dot(action[:3]/norm_action, obs["state"]["trajectory"][:3]/np.linalg.norm(obs["state"]["trajectory"][:3])))
        else:
            action_cost = 0
        action_diff_cost = 0.2 * np.sum(np.power(action - self.last_action, 2))    #0.2
        
        self.last_action[:] = action
        step_cost = 0.05

        gripper_release_cost = 0
        if obs["state"]["gripper_state"][1] and action[-1] < -0.5 and not self.announced_goals['box_pose']:
            gripper_release_cost = 50

        suction_cost = 0
        suction_reward = 0
        if self.announced_goals['box_pose'] or obs["state"]["gripper_state"][1] > 0.5:
            suction_reward = 2
        else:
            suction_cost = 1 * float(action[-1] > 0.5)
            
        relative_rotation = R.from_quat(obs["state"]["tcp_pose"][3:])
        angle = relative_rotation.magnitude()
        orientation_cost = max(angle - 0.005, 0.) * 1.

        max_pose_diff = 0.02  # set to 5mm
        pos_diff = obs["state"]["goal_pose"][:3] - obs["state"]["boxes"][:3]
        position_cost = 5. * np.sum(
            np.where(np.abs(pos_diff) > max_pose_diff, np.abs(pos_diff - np.sign(pos_diff) * max_pose_diff), 0.0)
        )
        # print("box", obs["state"]["boxes"][:])

        orientation_cost_box = 1. - sum(R.from_rotvec(obs["state"]["boxes"][3:]).as_quat() * R.from_rotvec(obs["state"]["goal_pose"][3:]).as_quat()) ** 2
        orientation_cost_box = max(orientation_cost_box - 0.005, 0.) * 1.

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
            gripper_release_cost=gripper_release_cost,
            total_reward=-action_cost - step_cost + suction_reward - suction_cost \
                - orientation_cost - action_diff_cost - gripper_release_cost + orientation_cost_box
        )
        for key, info in cost_info.items():
            self.unwrapped.cost_infos[key] = info + (0. if key not in self.unwrapped.cost_infos else self.unwrapped.cost_infos[key])
        for key, info in self.announced_goals.items():
            self.unwrapped.cost_infos[key] = info

        self.unwrapped.clip_costs()

        if self.reached_goal_state(obs):
            self.unwrapped.config.SUCCESS_COUNT += 1
            return 300. - action_cost - orientation_cost - action_diff_cost - position_cost\
                - suction_cost + suction_reward - orientation_cost_box - gripper_release_cost
        else:
            return 0. - action_cost - orientation_cost - suction_cost \
                - step_cost - action_diff_cost + suction_reward \
                - orientation_cost_box - gripper_release_cost - position_cost

    def reached_goal_state(self, obs) -> bool:
        ee_box_distance_goal = np.linalg.norm(obs["state"]["boxes"][2]) > 0.25

        if ee_box_distance_goal and not self.announced_goals['ee_box_distance']:
            self.announced_goals['ee_box_distance'] = True

        return ee_box_distance_goal \
                and self.announced_goals['box_pose'] \

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        self.last_action[:] = 0.
        self.force_goal_history = np.zeros(10)
        self.announced_goals['forces'] = False
        self.announced_goals['box_pose'] = False
        self.announced_goals['ee_box_distance'] = False
        self.low_pass_filter = np.zeros((7, 5))
    
        return obs, info
