from scipy.spatial.transform import Rotation as R
import numpy as np

def rotate_rotvec(rotvec, rotation_matrix):
    """
    Rotate a rotation vector by a given rotation matrix
    :args: rotvec: (x, y, z)
           rotation_matrix: 3x3 rotation matrix
    """
    return (R.from_matrix(rotation_matrix) * R.from_rotvec(rotvec)).as_rotvec()


def construct_adjoint_matrix(tcp_pose):
    """
    Construct the adjoint matrix for a spatial velocity vector
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[:3, 3:] = skew_matrix @ rotation
    if np.linalg.det(adjoint_matrix) < 0.1:
        print("Determinant of adjoint matrix is too low")
    return adjoint_matrix

def construct_adjoint_matrix_inverse(tcp_pose):
    """
    Construct the adjoint matrix for a spatial velocity vector
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation.T
    adjoint_matrix[3:, 3:] = rotation.T
    adjoint_matrix[3:, :3] = -rotation.T @ skew_matrix
    return adjoint_matrix


def construct_rotation_matrix(tcp_pose):
    """
    Construct the adjoint matrix for a spatial velocity vector
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    return R.from_quat(tcp_pose[3:]).as_matrix()


def construct_homogeneous_matrix(tcp_pose):
    """
    Construct the homogeneous transformation matrix from given pose.
    args: tcp_pose: (x, y, z, qx, qy, qz, qw) or (x, y, z, rx, ry, rz)
    """
    if len(tcp_pose) == 6:
        rotation = R.from_rotvec(tcp_pose[3:]).as_matrix()
    else:
        rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T

def invert_homogeneous_matrix(matrix):
    """
    Invert homogeneous transformation matrix.
    args: matrix: 4x4 matrix
    """
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inv_rotation = rotation.T
    inv_translation = -inv_rotation @ translation
    inv_matrix = np.zeros((4, 4))
    inv_matrix[:3, :3] = inv_rotation
    inv_matrix[:3, 3] = inv_translation
    inv_matrix[3, 3] = 1
    return inv_matrix

def construct_homogeneous_vector(vector):
    """
    Construct the homogeneous vector from given vector.
    args: vector: (x, y, z)
    """
    return np.array([vector[0], vector[1], vector[2], 1])