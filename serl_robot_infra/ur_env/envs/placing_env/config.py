from ur_env.envs.ur5_env import DefaultEnvConfig
import numpy as np
from scipy.spatial.transform import Rotation as R

class UR5PlacingCornerConfig(DefaultEnvConfig):
    # RESET_Q = np.array([[1.34231, -1.24585, 1.94961, -2.27267, -1.56428, -0.22641]])   # original one
    # RESET_Q = np.array([[1.3463, -1.3584,  1.9014, -2.1243, -1.5758, -0.2312]])
    RESET_Q = np.array([
        [-28.86, -69.02, 117.94, -138.81, -89.76, -28.88],
        # [-5, -78.62, 122.84, -134.22, -89.81, -13.03],
        # [10, -75.62, 122.84, -134.22, -89.81, -13.03],
    ])
    RESET_Q = np.deg2rad(RESET_Q)
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.06,)
    RANDOM_ROT_RANGE = (0.0,)
    # ABS_POSE_LIMIT_HIGH = np.array([0.14, -0.4, 0.2, 3.2, 0.1, 3.2])            # TODO euler rotations suck :/
    # ABS_POSE_LIMIT_LOW = np.array([-0.3, -0.7, -0.006, 3.0, -0.1, -3.2])
    ABS_POSE_LIMIT_HIGH = np.array([-0.3, 0.2, 0.4, 0.05, 0.05, 0.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.65, -0.1, 0.08, -0.05, -0.05, -0.2])
    ACTION_SCALE = np.array([0.01, 0.05, 1.], dtype=np.float32)

    ROBOT_IP: str = "192.168.1.66"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])

    GOAL_POSE = np.array([0.024, 0.02, 0.21, 0., 0., 3.14])     #box_1
    GOAL_POSE = np.array([0.03, 0.03, 0.21, 0., 0., 3.14])      #box_340
    GOAL_POSE = np.array([0.035, 0.03, 0.21, 0., 0., 3.14])     #box_330
    # GOAL_POSITION = np.array([-0.38, -0.01, -0.04])    #box_1 next to box_3
    # GOAL_POSITION = np.array([-0.38, -0.12, -0.04])    #box_5 next to box_1
    
    ROTATION_GENERALIZATION = R.from_euler("xyz", np.array([0, 0, 0])).as_matrix() # rotation applied to the box to bring it back to the training orientation
    BOX_ERROR = np.array([0.0, 0.0, 0.0])
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
        [-28.86, -79.38, 110.09, -120.59, -89.77, -28.95] #-28.95          -> box_1 straight
        # [-26.69, -76.06, 115.90, -129.74, -89.76, -26.74] # -26.74                 -> box_1 horizontal
    ])
    RESET_Q = np.deg2rad(RESET_Q)
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.06,)
    RANDOM_ROT_RANGE = (0.0,)
    # ABS_POSE_LIMIT_HIGH = np.array([0.14, -0.4, 0.2, 3.2, 0.1, 3.2])            # TODO euler rotations suck :/
    # ABS_POSE_LIMIT_LOW = np.array([-0.3, -0.7, -0.006, 3.0, -0.1, -3.2])
    ABS_POSE_LIMIT_HIGH = np.array([-0.3, 0.2, 0.4, 0.05, 0.05, 0.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.65, -0.1, 0.12, -0.05, -0.05, -0.2])
    # ABS_POSE_LIMIT_HIGH = np.array([1.2, 1.3, 1.5, 0.05, 0.05, 0.2])
    # ABS_POSE_LIMIT_LOW = np.array([-1.8, -1.35, -1.12, -0.05, -0.05, -0.2])
    ACTION_SCALE = np.array([0.01, 0.05, 1.], dtype=np.float32)

    ROBOT_IP: str = "192.168.1.66"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])

    # GOAL_POSE = np.array([0.01, 0., 0.3, 0., -1.5, 0.])    #box_4
    GOAL_POSE = np.array([-0.0, 0.0, 0.3, 0., -1.57, 0.])   #box_1 vertical
    # GOAL_POSE = np.array([0., 0., 0.3, 0., 3.14, 0.])    #box_1 horizontal
    
    # TARGET_ORIENTATION = np.array([ 1.2335, 1.2125, -1.1869]) #as exponential coordinates aka rotation vector
    # TARGET_ORIENTATION = np.array([ -1.1989, 1.1933, 1.198]) #as exponential coordinates aka rotation vector
    ROTATION_GENERALIZATION = R.from_euler("xyz", np.array([0, 0, 0])).as_matrix() # rotation applied to the box to bring it back to the training orientation
    BOX_ERROR = np.array([0.0, 0.0, 0.0])
    POSE_ESTIMATION = True
    POSE_ESTIMATION_IP = "ws://localhost:7777"
    WF_rot = np.array([[-1,  0,  0],
                        [ 0,  0, 1],
                        [ 0, 1,  0]], dtype=np.float32)
    LOW_PASS_FILTER = 0
    SUCCESS_COUNT = 0