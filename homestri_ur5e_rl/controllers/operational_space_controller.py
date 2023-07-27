from homestri_ur5e_rl.utils.mujoco_utils import (
    get_site_jac, 
    get_fullM,
)
from homestri_ur5e_rl.utils.transform_utils import (
    quat2mat,
    mat2quat,
    quat_multiply,
    quat_conjugate,
    quat2axisangle
)
from homestri_ur5e_rl.utils.controller_utils import (
    task_space_inertia_matrix,
)
import numpy as np

class OperationalSpaceController():
    def __init__(
        self,
        model,
        data,
        model_names,
        eef_name,
        joint_names,
        actuator_names,
        kp,
        ko,
        kv,
        vmax_xyz,
        vmax_abg,
        ctrl_dof,
        null_damp_kv,
    ):
        self.model = model
        self.data = data
        self.model_names = model_names
        self.eef_name = eef_name
        self.joint_names = joint_names 
        self.actuator_names = actuator_names
        self.kp = kp
        self.ko = ko
        self.kv = kv
        self.vmax_xyz = vmax_xyz
        self.vmax_abg = vmax_abg
        self.ctrl_dof = np.copy(ctrl_dof)
        self.null_damp_kv = null_damp_kv

        self.n_joints = len(self.joint_names)
        self.eef_id = self.model_names.site_name2id[self.eef_name]
        self.joint_ids = [self.model_names.joint_name2id[name] for name in self.joint_names]
        self.jnt_qpos_ids = [self.model.jnt_qposadr[id] for id in self.joint_ids]
        self.jnt_dof_ids = [self.model.jnt_dofadr[id] for id in self.joint_ids]
        self.actuator_ids = [self.model_names.actuator_name2id[name] for name in actuator_names]

        self.task_space_gains = np.array([self.kp] * 3 + [self.ko] * 3)
        self.lamb = self.task_space_gains / self.kv
        self.sat_gain_xyz = vmax_xyz / self.kp * self.kv
        self.sat_gain_abg = vmax_abg / self.ko * self.kv
        self.scale_xyz = vmax_xyz / self.kp * self.kv
        self.scale_abg = vmax_abg / self.ko * self.kv

        self.target_pos = np.zeros(3)
        self.target_ori_mat = np.eye(3)
        self.target_twist = np.zeros(6)

    def set_target_pose(self, action):
        self.target_pos = action[:3]
        self.target_ori_mat = quat2mat(action[3:])

    def set_target_twist(self, action):
        self.target_twist = action

    def run_controller(self, ctrl, vel_ctrl=True):
        q = self.data.qpos[self.jnt_qpos_ids]
        dq = self.data.qvel[self.jnt_dof_ids]

        J = get_site_jac(self.model, self.data, self.eef_id)
        J = J[:, self.jnt_dof_ids]

        M_full = get_fullM(self.model, self.data)
        M = M_full[self.jnt_dof_ids,:][:,self.jnt_dof_ids]
        Mx, M_inv = task_space_inertia_matrix(M, J)

        ee_pos = self.data.site_xpos[self.eef_id]
        ee_ori_mat = self.data.site_xmat[self.eef_id].reshape(3, 3)
        ee_twist = J @ dq

        u_task = np.zeros(6)
        if not vel_ctrl:
            u_task[:3] = ee_pos - self.target_pos
            u_task[3:] = self._rotational_error(ee_ori_mat, self.target_ori_mat)
            u_task = self._scale_signal_vel_limited(u_task)

        u = np.zeros(self.n_joints)
        if np.all(self.target_twist == 0):
            u -= self.kv * np.dot(M, dq)
        else:
            u_task += self.kv * (ee_twist - self.target_twist)

        u_task = u_task * self.ctrl_dof

        u -= np.dot(J.T, np.dot(Mx, u_task))

        u += self.data.qfrc_bias[self.jnt_dof_ids]


        u_null = np.dot(M, -self.null_damp_kv*dq)
        Jbar = np.dot(M_inv, np.dot(J.T, Mx))
        null_filter = np.eye(self.n_joints) - np.dot(J.T, Jbar.T)
        u += np.dot(null_filter, u_null)

        ctrl[self.actuator_ids] = u

    def _scale_signal_vel_limited(self, u_task):
        """Scale the control signal such that the arm isn't driven to move
        faster in position or orientation than the specified vmax values

        Parameters
        ----------
        u_task: np.array
            the task space control signal
        """
        norm_xyz = np.linalg.norm(u_task[:3])
        norm_abg = np.linalg.norm(u_task[3:])
        scale = np.ones(6)
        if norm_xyz > self.sat_gain_xyz:
            scale[:3] *= self.scale_xyz / norm_xyz
        if norm_abg > self.sat_gain_abg:
            scale[3:] *= self.scale_abg / norm_abg

        return self.kv * scale * self.lamb * u_task

    def _rotational_error(self, target_ori_mat, current_ori_mat):   
        q_t = mat2quat(target_ori_mat)
        q_c = mat2quat(current_ori_mat) 
        q_e = quat_multiply(q_t, quat_conjugate(q_c))

        u_task_orientation = quat2axisangle(q_e)

        return u_task_orientation