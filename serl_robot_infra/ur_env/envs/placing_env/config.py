from ur_env.envs.ur5_env import DefaultEnvConfig
import numpy as np


class UR5PlacingCornerConfig(DefaultEnvConfig):
    # RESET_Q = np.array([[1.34231, -1.24585, 1.94961, -2.27267, -1.56428, -0.22641]])   # original one
    # RESET_Q = np.array([[1.3463, -1.3584,  1.9014, -2.1243, -1.5758, -0.2312]])
    RESET_Q = np.array([
        [-17.67, -74.775, 127.48, -143.71, -89.81, -27.89],
        # [-5.67, -74.775, 127.48, -143.71, -89.81, -27.89]
    ])
    RESET_Q = np.deg2rad(RESET_Q)
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.06,)
    RANDOM_ROT_RANGE = (0.0,)
    # ABS_POSE_LIMIT_HIGH = np.array([0.14, -0.4, 0.2, 3.2, 0.1, 3.2])            # TODO euler rotations suck :/
    # ABS_POSE_LIMIT_LOW = np.array([-0.3, -0.7, -0.006, 3.0, -0.1, -3.2])
    ABS_POSE_LIMIT_HIGH = np.array([-0.2, 0.1, 0.25, 0.05, 0.05, 0.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.6, -0.15, 0.12, -0.05, -0.05, -0.2])
    ACTION_SCALE = np.array([0.01, 0.05, 1.], dtype=np.float32)

    ROBOT_IP: str = "192.168.1.66"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])

    # GOAL_POSITION = np.array([-0.4, 0.095, -0.05])    #box_6
    GOAL_POSITION = np.array([-0.42, 0.04, -0.04])    #box_1
    BOX_ERROR = np.array([0.03, 0, 0])
    POSE_ESTIMATION = True
    POSE_ESTIMATION_IP = "ws://localhost:7777"
    WF_rot = np.array([[-1,  0,  0],
                        [ 0,  0, 1],
                        [ 0, 1,  0]], dtype=np.float32)