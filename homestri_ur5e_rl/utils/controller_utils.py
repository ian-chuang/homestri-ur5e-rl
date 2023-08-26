import numpy as np
from homestri_ur5e_rl.utils.transform_utils import (
    quat_multiply,
    quat_conjugate,
    quat2mat,
)

EPS = np.finfo(float).eps * 4.0

def task_space_inertia_matrix(M, J, threshold=1e-3):
    """Generate the task-space inertia matrix

    Parameters
    ----------
    M: np.array
        the generalized coordinates inertia matrix
    J: np.array
        the task space Jacobian
    threshold: scalar, optional (Default: 1e-3)
        singular value threshold, if the detminant of Mx_inv is less than
        this value then Mx is calculated using the pseudo-inverse function
        and all singular values < threshold * .1 are set = 0
    """

    # calculate the inertia matrix in task space
    M_inv = np.linalg.inv(M)
    Mx_inv = np.dot(J, np.dot(M_inv, J.T))
    if abs(np.linalg.det(Mx_inv)) >= threshold:
        # do the linalg inverse if matrix is non-singular
        # because it's faster and more accurate
        Mx = np.linalg.inv(Mx_inv)
    else:
        # using the rcond to set singular values < thresh to 0
        # singular values < (rcond * max(singular_values)) set to 0
        Mx = np.linalg.pinv(Mx_inv, rcond=threshold * 0.1)

    return Mx, M_inv




def get_rot_angle(rotation_matrix, eps=EPS):
    data = rotation_matrix.reshape(9)
    ca = (data[0]+data[4]+data[8]-1)/2.0
    t = eps * eps / 2.0
    
    if ca > 1 - t:
        # Undefined case, choose the Z-axis and angle 0
        axis = np.array([0, 0, 1])
        return 0, axis
    
    if ca < -1 + t:
        # The case of angles consisting of multiples of pi:
        # two solutions, choose a positive Z-component of the axis
        x = np.sqrt((data[0] + 1.0) / 2)
        y = np.sqrt((data[4] + 1.0) / 2)
        z = np.sqrt((data[8] + 1.0) / 2)
        
        if data[2] < 0:
            x = -x
        if data[7] < 0:
            y = -y
        if x * y * data[1] < 0:
            x = -x
        
        # z always >= 0
        # if z equals zero
        axis = np.array([x, y, z])
        return np.pi, axis
    
    axis_x = data[7] - data[5]
    axis_y = data[2] - data[6]
    axis_z = data[3] - data[1]
    mod_axis = np.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)
    
    axis = np.array([axis_x / mod_axis, axis_y / mod_axis, axis_z / mod_axis])
    angle = np.arctan2(mod_axis / 2, ca)
    
    return angle, axis




def pose_error(target_pose, ee_pose) -> np.ndarray:
    """
    Calculate the rotational error (orientation difference) between the target and current orientation.

    Parameters:
        target_ori_mat (numpy.ndarray): The target orientation matrix.
        current_ori_mat (numpy.ndarray): The current orientation matrix.

    Returns:
        numpy.ndarray: The rotational error in axis-angle representation.
    """
    target_pos = target_pose[:3]
    target_quat = target_pose[3:]
    ee_pos = ee_pose[:3]
    ee_quat = ee_pose[3:]

    err_pos = target_pos - ee_pos

    err_quat = quat_multiply(target_quat, quat_conjugate(ee_quat))
    angle, axis = get_rot_angle(quat2mat(err_quat))

    return np.concatenate([err_pos, angle * axis])
