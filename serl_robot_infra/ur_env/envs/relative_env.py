from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
from gym import Env
from franka_env.utils.transformations import (
    construct_adjoint_matrix,
    construct_adjoint_matrix_inverse,
    construct_homogeneous_matrix,
    construct_rotation_matrix,
    construct_homogenous_vector,
    invert_homogeneous_matrix,
    rotate_rotvec
)

from ur_env.envs.placing_env.config import UR5PlacingCornerConfig


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
        self.config = UR5PlacingCornerConfig

        # Homogeneous transformation matrix from reset pose's relative frame to base frame
        self.T_r_o = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        # print("action", action)
        transformed_action = self.transform_action(action) #action in network will be in base frame!!!!????
        # print("transformed_action", transformed_action)
        obs, reward, done, truncated, info = self.env.step(transformed_action) #go deeper, to spacemouse env or ur5 env

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            # print("intervene_action", self.transform_action_inv(info["intervene_action"]))
            info["intervene_action"] = self.transform_action_inv(info["intervene_action"])
            # print("intervene_action transformed", info["intervene_action"])

        # Update rotation matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        # print("action", transformed_obs["state"]["action"])
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
        T_ee_o = invert_homogeneous_matrix(T_o_ee)

        # Reconstruct transformed tcp_pose vector
        p_r_ee = T_r_ee[:3, 3]
        theta_r_ee = R.from_matrix(T_r_ee[:3, :3]).as_quat()
        obs["state"]["tcp_pose"] = np.concatenate((p_r_ee, theta_r_ee))
        
        
        obs["state"]["action"][:6] = self.transform_action_inv(obs["state"]["action"][:6])
        
        if self.config.POSE_ESTIMATION:
            obs["state"]["boxes"][:3] = (T_ee_o @ construct_homogenous_vector(obs["state"]["boxes"][:3]))[:3]
            obs["state"]["boxes"][3:6] = (R.from_matrix(T_ee_o[:3, :3]) * R.from_rotvec(obs["state"]["boxes"][3:6])).as_rotvec()
            
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
