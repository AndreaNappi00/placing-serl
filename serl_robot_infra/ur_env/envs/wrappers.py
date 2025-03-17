import gymnasium as gym
import numpy as np
from agentlace import action

from ur_env.spacemouse.spacemouse_expert import SpaceMouseExpert
import time
from scipy.spatial.transform import Rotation as R

from ur_env.utils.rotations import quat_2_euler, quat_2_mrp, quat_2_rotvec

from ur_env.utils.vacuum_gripper import VacuumGripper


ROT90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
ROT_GENERAL = np.array([np.eye(3), ROT90, ROT90 @ ROT90, ROT90.transpose()])

def quat_diff(quat1: np.ndarray, quat2: np.ndarray) -> np.ndarray:
    quat1 = R.from_quat(quat1)
    quat2 = R.from_quat(quat2)
    rel = quat2 * quat1.inv()
    return rel.as_quat()

class KinestheticTeaching(gym.ActionWrapper):
    def __init__(self, env, gripper_action_span=3):
        super().__init__(env)

        self.gripper_enabled = True

        self.last_intervene = 0
        self.left = np.array([False] * gripper_action_span, dtype=np.bool_)
        self.right = self.left.copy()

        self.invert_axes = [-1, -1, 1, -1, -1, 1]
        self.deadspace = 0.15
        
        self.past_poses = np.array([self.unwrapped.curr_pos] * 10) # (10, 7)
        self.past_grip = self.controller.robotiq_gripper.ObjectStatus.NO_OBJ_DETECTED
        
        self.controller.teaching_mode = 1
        
        self.scaling_velocities = 3.
    
    def adapt_output_mrp(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - expert_a: raw output
        Output:
        - expert_a: output adapted to force space (action)
        """

        # position = super().get_wrapper_attr("curr_pos")  # get position from ur_env
        position = self.unwrapped.curr_pos
        z_angle = np.arctan2(position[1], position[0])  # get first joint angle

        z_rot = R.from_rotvec(np.array([0, 0, z_angle]))
        action[:6] *= self.invert_axes  # if some want to be inverted
        action[:3] = z_rot.apply(action[:3])  # z rotation invariant translation

        # TODO add tcp orientation to the equation (extract z rotation from tcp pose)
        action[3:6] = z_rot.apply(action[3:6])  # z rotation invariant rotation

        return action
        
    def adapt_output_quat(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - expert_a: raw output
        Output:
        - expert_a: output adapted to force space (action)
        """

        # position = super().get_wrapper_attr("curr_pos")  # get position from ur_env
        position = self.unwrapped.curr_pos
        z_angle = np.arctan2(position[1], position[0])  # get first joint angle

        z_rot = R.from_rotvec(np.array([0, 0, z_angle]))
        action[:6] *= self.invert_axes  # if some want to be inverted
        action[:3] = z_rot.apply(action[:3])  # z rotation invariant translation

        # TODO add tcp orientation to the equation (extract z rotation from tcp pose)
        action[3:6] = z_rot.apply(action[3:6])  # z rotation invariant rotation

        return action
        
    def get_action(self) -> np.ndarray:
        # curr_pose = self.unwrapped.curr_pos
        # expert_a = curr_pose[:3] - self.past_poses[0][:3]
        expert_a = self.unwrapped.curr_vel
        
        # print(f"expert_a: {expert_a}")
                
        # q_diff = quat_diff(self.past_poses[0][:3], curr_pose[3:])
        expert_a[3:] = expert_a[3:] / self.scaling_velocities
        # print(f"expert_a after clip: {expert_a}")
                
        # q_diff = quat_2_euler(q_diff)
        # expert_a = np.concatenate((expert_a, q_diff), axis=0)
        
        # self.past_poses = np.roll(self.past_pose, -1, axis=0)
        # self.past_poses[-1] = curr_pose
        
        # return self.adapt_output(expert_a)
        return self.adapt_output_mrp(expert_a)
        
    def convert_gripper_action(self, action: VacuumGripper.ObjectStatus) -> np.ndarray:
        if action == 0.0:
            return np.zeros((1,))
        left = int(self.past_grip != VacuumGripper.ObjectStatus.NO_OBJ_DETECTED and action == VacuumGripper.ObjectStatus.NO_OBJ_DETECTED)
        right = int(self.past_grip == VacuumGripper.ObjectStatus.NO_OBJ_DETECTED and action != VacuumGripper.ObjectStatus.NO_OBJ_DETECTED)
        
        self.past_grip = action

        return np.zeros((1,)) + left - right

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        gripper_s = self.unwrapped.gripper_state[1]
        expert_a = self.get_action()
        
        # print(f"expert_a: {expert_a}")
        
        gripper_a = self.convert_gripper_action(gripper_s)
        
        if np.linalg.norm(expert_a) > 0.001 or gripper_a.any():            
            expert_a = np.concatenate((expert_a, gripper_a), axis=0)
            
            return expert_a

        return action

    def step(self, action):
        new_action = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        info["intervene_action"] = new_action
        info["left"] = self.left.any()
        info["right"] = self.right.any()
        return obs, rew, done, truncated, info

class DemonstrationWrapper(gym.ActionWrapper):
    def __init__(self, env, gripper_action_span=3):
        super().__init__(env)

        self.gripper_enabled = True

        self.last_intervene = 0
        self.left = np.array([False] * gripper_action_span, dtype=np.bool_)
        self.right = self.left.copy()

        self.invert_axes = [-1, -1, 1, -1, -1, 1]
        self.deadspace = 0.15
        
        self.past_poses = np.array([self.unwrapped.curr_pos] * 10) # (10, 7)
        self.past_grip = self.controller.robotiq_gripper.ObjectStatus.NO_OBJ_DETECTED
        
        self.controller.teaching_mode = 2
        
        self.scaling_velocities = 3.
    
    def adapt_output_mrp(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - expert_a: raw output
        Output:
        - expert_a: output adapted to force space (action)
        """

        # position = super().get_wrapper_attr("curr_pos")  # get position from ur_env
        position = self.unwrapped.curr_pos
        z_angle = np.arctan2(position[1], position[0])  # get first joint angle

        z_rot = R.from_rotvec(np.array([0, 0, z_angle]))
        action[:6] *= self.invert_axes  # if some want to be inverted
        action[:3] = z_rot.apply(action[:3])  # z rotation invariant translation

        # TODO add tcp orientation to the equation (extract z rotation from tcp pose)
        action[3:6] = z_rot.apply(action[3:6])  # z rotation invariant rotation

        return action
        
    def adapt_output_quat(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - expert_a: raw output
        Output:
        - expert_a: output adapted to force space (action)
        """

        # position = super().get_wrapper_attr("curr_pos")  # get position from ur_env
        position = self.unwrapped.curr_pos
        z_angle = np.arctan2(position[1], position[0])  # get first joint angle

        z_rot = R.from_rotvec(np.array([0, 0, z_angle]))
        action[:6] *= self.invert_axes  # if some want to be inverted
        action[:3] = z_rot.apply(action[:3])  # z rotation invariant translation

        # TODO add tcp orientation to the equation (extract z rotation from tcp pose)
        action[3:6] = z_rot.apply(action[3:6])  # z rotation invariant rotation

        return action
        
    def get_action(self) -> np.ndarray:
        # curr_pose = self.unwrapped.curr_pos
        # expert_a = curr_pose[:3] - self.past_poses[0][:3]
        expert_a = self.unwrapped.curr_vel
        expert_a = np.array([0., 0., -1., 0., 0., 0.])
        
        # print(f"expert_a: {expert_a}")
                
        # q_diff = quat_diff(self.past_poses[0][:3], curr_pose[3:])
        expert_a[3:] = expert_a[3:] / self.scaling_velocities
        # print(f"expert_a after clip: {expert_a}")
                
        # q_diff = quat_2_euler(q_diff)
        # expert_a = np.concatenate((expert_a, q_diff), axis=0)
        
        # self.past_poses = np.roll(self.past_pose, -1, axis=0)
        # self.past_poses[-1] = curr_pose
        
        # return self.adapt_output(expert_a)
        return self.adapt_output_mrp(expert_a)
        
    def convert_gripper_action(self, action: VacuumGripper.ObjectStatus) -> np.ndarray:
        if action == 0.0:
            return np.zeros((1,))
        left = int(self.past_grip != VacuumGripper.ObjectStatus.NO_OBJ_DETECTED and action == VacuumGripper.ObjectStatus.NO_OBJ_DETECTED)
        right = int(self.past_grip == VacuumGripper.ObjectStatus.NO_OBJ_DETECTED and action != VacuumGripper.ObjectStatus.NO_OBJ_DETECTED)
        
        self.past_grip = action

        return np.zeros((1,)) + left - right

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        gripper_s = self.unwrapped.gripper_state[1]
        expert_a = self.get_action()
        
        # print(f"expert_a: {expert_a}")
        
        gripper_a = self.convert_gripper_action(gripper_s)
        
        if np.linalg.norm(expert_a) > 0.001 or gripper_a.any():            
            expert_a = np.concatenate((expert_a, gripper_a), axis=0)
            
            return expert_a

        return action

    def step(self, action):
        new_action = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        info["intervene_action"] = new_action
        info["left"] = self.left.any()
        info["right"] = self.right.any()
        return obs, rew, done, truncated, info

class InterruptActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        input("press a key to start again!")
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, gripper_action_span=3):
        super().__init__(env)

        self.gripper_enabled = True

        self.expert = SpaceMouseExpert()
        self.last_intervene = 0
        self.left = np.array([False] * gripper_action_span, dtype=np.bool_)
        self.right = self.left.copy()

        self.invert_axes = [1, 1, 1, 1, 1, 1]
        self.deadspace = 0.15
        
        self.env.unwrapped.residual_learning_inference = False

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a = self.get_deadspace_action()
        intervened = False

        # If the SpaceMouse is moved or buttons are pressed, update the last intervention time.
        if np.linalg.norm(
                expert_a) > 0.001 or self.left.any() or self.right.any():  # also read buttons with no movement
            self.last_intervene = time.time()
            intervened = True

        # Handle gripper action if gripper control is enabled.
        if self.gripper_enabled:
            gripper_action = np.zeros((1,)) + int(self.left.any()) - int(self.right.any())
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if time.time() - self.last_intervene < 0.5 and intervened:
            expert_a = self.adapt_spacemouse_output(expert_a)
            return expert_a, True

        return action, False

    def get_deadspace_action(self) -> np.ndarray:
        expert_a, buttons = self.expert.get_action()

        positive = np.clip((expert_a - self.deadspace) / (1. - self.deadspace), a_min=0.0, a_max=1.0)
        negative = np.clip((expert_a + self.deadspace) / (1. - self.deadspace), a_min=-1.0, a_max=0.0)
        expert_a = positive + negative  # remove all values in deadspace
        
        self.left, self.right = np.roll(self.left, -1), np.roll(self.right, -1)  # shift them one to the left
        self.left[-1], self.right[-1] = tuple(buttons)  #memory of 3 elements

        return np.array(expert_a, dtype=np.float32)

    def adapt_spacemouse_output(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - expert_a: spacemouse raw output
        Output:
        - expert_a: spacemouse output adapted to force space (action)
        """

        position = super().get_wrapper_attr("curr_pos")  # get position from ur_env
        # position = self.unwrapped.curr_pos
        # z_angle = np.arctan2(position[1], position[0])  # get first joint angle

        z_rot = R.from_rotvec(np.array([0, 0, -np.pi]))
        action[:6] *= self.invert_axes  # if some want to be inverted
        action[:3] = z_rot.apply(action[:3])  # z rotation invariant translation

        # TODO add tcp orientation to the equation (extract z rotation from tcp pose)
        action[3:6] = z_rot.apply(action[3:6])  # z rotation invariant rotation

        return action

    def step(self, action):
        new_action, replaced = self.action(action)
        # print(f"new action: {new_action}")
        obs, rew, done, truncated, info = self.env.step(new_action)
        # info["intervene_action"] = new_action     ##change this if you want to use the spacemouse action

        # Add additional information to the info dictionary about the intervention.
        if replaced:
            info["hil_action"] = new_action # key for the human in the loop action
            
        info["intervene_action"] = new_action
        
        info["left"] = self.left.any()
        info["right"] = self.right.any()
        
        # Return the observation, reward, done flag, truncation flag, and info dictionary.
        return obs, rew, done, truncated, info


class Quat2EulerWrapper(gym.ObservationWrapper):  # not used anymore (stay away from euler angles!)
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], quat_2_euler(tcp_pose[3:]))
        )
        return observation


class Quat2MrpWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], quat_2_mrp(tcp_pose[3:]))
        )
        return observation

class Quat2rotvecWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotvec
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], quat_2_rotvec(tcp_pose[3:]))
        )
        return observation


def rotate_state(state: np.ndarray, num_rot: int):
    assert len(state.shape) == 1 and state.shape[0] % 3 == 0
    state = state.reshape((-1, 3)).transpose()
    rotated = np.dot(ROT_GENERAL[num_rot % 4], state).transpose()
    return rotated.reshape((-1))


class ObservationRotationWrapper(gym.Wrapper):
    """
    Convert every observation into the first quadrant of the Relative Frame
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        print("Observation Rotation Wrapper enabled!")
        self.num_rot_quadrant = -1

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        obs = self.rotate_observation(obs, random=True)     # rotate initial state random
        return obs, info

    def step(self, action: np.ndarray):
        action = self.rotate_action(action=action)
        obs, reward, done, truncated, info = self.env.step(action)
        # print("\nquadrant: ", self.num_rot_quadrant)
        rotated_obs = self.rotate_observation(obs)
        return rotated_obs, reward, done, truncated, info

    def rotate_observation(self, observation, random=False):
        if not random:
            x, y = (observation["state"]["tcp_pose"][:2])
            self.num_rot_quadrant = int(x < 0.) * 2 + int(x * y < 0.)  # save quadrant info
        else:
            self.num_rot_quadrant = int(time.time_ns()) % 4  # do not mess with seeded np.random

        for state in observation["state"].keys():
            if state == "gripper_state":
                continue
            elif state == "action":
                observation["state"][state][:6] = rotate_state(observation["state"][state][:6], self.num_rot_quadrant)
            else:
                observation["state"][state][:] = rotate_state(observation["state"][state],
                                                              self.num_rot_quadrant)  # rotate

        if "images" in observation:
            for image_keys in observation["images"].keys():
                observation["images"][image_keys][:] = np.rot90(
                    observation["images"][image_keys],
                    axes=(0, 1),
                    k=self.num_rot_quadrant
                )
        return observation

    def rotate_action(self, action):
        rotated_action = action.copy()
        rotated_action[:6] = rotate_state(action[:6], 4 - self.num_rot_quadrant)  # rotate
        return rotated_action
