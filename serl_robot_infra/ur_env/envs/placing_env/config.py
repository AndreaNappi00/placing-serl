from ur_env.envs.ur5_env import DefaultEnvConfig
import numpy as np
from scipy.spatial.transform import Rotation as R

class UR5PlacingCornerConfig(DefaultEnvConfig):
    # RESET_Q = np.array([[1.34231, -1.24585, 1.94961, -2.27267, -1.56428, -0.22641]])   # original one
    # RESET_Q = np.array([[1.3463, -1.3584,  1.9014, -2.1243, -1.5758, -0.2312]])
    RESET_Q = np.array([
        [-17.66, -80.24, 124.91, -134.67, -89.81, -28.40],
        # [-5, -78.62, 122.84, -134.22, -89.81, -13.03],
        # [10, -75.62, 122.84, -134.22, -89.81, -13.03],
    ])
    RESET_Q = np.deg2rad(RESET_Q)
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.06,)
    RANDOM_ROT_RANGE = (0.0,)
    # ABS_POSE_LIMIT_HIGH = np.array([0.14, -0.4, 0.2, 3.2, 0.1, 3.2])            # TODO euler rotations suck :/
    # ABS_POSE_LIMIT_LOW = np.array([-0.3, -0.7, -0.006, 3.0, -0.1, -3.2])
    ABS_POSE_LIMIT_HIGH = np.array([-0.2, 0.1, 0.25, 0.05, 0.05, 0.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.8, -0.35, 0.12, -0.05, -0.05, -0.2])
    ACTION_SCALE = np.array([0.01, 0.05, 1.], dtype=np.float32)

    ROBOT_IP: str = "192.168.1.66"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])

    GOAL_POSITION = np.array([-0.4, 0.1, -0.04])    #box_1
    # GOAL_POSITION = np.array([-0.38, -0.01, -0.04])    #box_1 next to box_3
    # GOAL_POSITION = np.array([-0.38, -0.12, -0.04])    #box_5 next to box_1
    
    TARGET_ORIENTATION = np.array([-2.2, -2.22, 0]) #as exponential coordinates aka rotation vector
    ROTATION_GENERALIZATION = R.from_euler("xyz", np.array([0, 0, -np.pi])).as_matrix() # rotation applied to the box to bring it back to the training orientation
    BOX_ERROR = np.array([0.03, 0.01, -0.03])
    POSE_ESTIMATION = True
    POSE_ESTIMATION_IP = "ws://localhost:7777"
    WF_rot = np.array([[-1,  0,  0],
                        [ 0,  0, 1],
                        [ 0, 1,  0]], dtype=np.float32)
    LOW_PASS_FILTER = 0
    SUCCESS_COUNT = 0
    
class UR5PlacingVerticalConfig(DefaultEnvConfig):
    # RESET_Q = np.array([[1.34231, -1.24585, 1.94961, -2.27267, -1.56428, -0.22641]])   # original one
    # RESET_Q = np.array([[1.3463, -1.3584,  1.9014, -2.1243, -1.5758, -0.2312]])
    RESET_Q = np.array([
        [-31.22, -83.23, 112.54, -118.31, -90.98, -42.14],
        # [-5, -78.62, 122.84, -134.22, -89.81, -13.03],
        # [10, -75.62, 122.84, -134.22, -89.81, -13.03],
    ])
    RESET_Q = np.deg2rad(RESET_Q)
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.06,)
    RANDOM_ROT_RANGE = (0.0,)
    # ABS_POSE_LIMIT_HIGH = np.array([0.14, -0.4, 0.2, 3.2, 0.1, 3.2])            # TODO euler rotations suck :/
    # ABS_POSE_LIMIT_LOW = np.array([-0.3, -0.7, -0.006, 3.0, -0.1, -3.2])
    ABS_POSE_LIMIT_HIGH = np.array([-0.2, 0.3, 0.5, 0.05, 0.05, 0.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.8, -0.35, 0.12, -0.05, -0.05, -0.2])
    ACTION_SCALE = np.array([0.01, 0.05, 1.], dtype=np.float32)

    ROBOT_IP: str = "192.168.1.66"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])

    GOAL_POSITION = np.array([-0.476, 0.145, -0.018])    #box_1
    # GOAL_POSITION = np.array([-0.38, -0.01, -0.04])    #box_1 next to box_3
    # GOAL_POSITION = np.array([-0.38, -0.12, -0.04])    #box_5 next to box_1
    
    TARGET_ORIENTATION = np.array([-1.2, -1.25, -1.2]) #as exponential coordinates aka rotation vector
    # TARGET_ORIENTATION = np.array([1.32, -1.31, 1.14]) #as exponential coordinates aka rotation vector
    ROTATION_GENERALIZATION = R.from_euler("xyz", np.array([0, 0, 0])).as_matrix() # rotation applied to the box to bring it back to the training orientation
    BOX_ERROR = np.array([0.0, 0.0, 0.0])
    POSE_ESTIMATION = True
    POSE_ESTIMATION_IP = "ws://localhost:7777"
    WF_rot = np.array([[-1,  0,  0],
                        [ 0,  0, 1],
                        [ 0, 1,  0]], dtype=np.float32)
    LOW_PASS_FILTER = 0
    SUCCESS_COUNT = 0