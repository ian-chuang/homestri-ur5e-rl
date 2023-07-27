from homestri_ur5e_rl.utils.mujoco_utils import (
    get_site_xpos_by_id, 
    get_site_xmat_by_id,
    get_site_xvelp_by_id, 
    get_site_xvelr_by_id,
    get_site_jacp_by_id, 
    get_site_jacr_by_id, 
    get_obs_by_id,
    get_obs_by_name
)
from homestri_ur5e_rl.utils.transform_utils import (
    compute_position_error,
    set_goal_position,
    set_goal_orientation
)
from homestri_ur5e_rl.utils.pd_controller_utils import (
    SpatialPDController
)
from homestri_ur5e_rl.utils.solver_utils import (
    JacobianTransposeSolver
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
        p_values=None,
        d_values=None 
    ):
        self.model = model
        self.data = data
        self.model_names = model_names
        self.eef_name = eef_name
        self.joint_names = joint_names 
        self.actuator_names = actuator_names

        self.eef_id = self.model_names.site_name2id[self.eef_name]
        self.joint_ids = [self.model_names.joint_name2id[name] for name in self.joint_names]
        self.jnt_qpos_ids = [self.model.jnt_qposadr[id] for id in self.joint_ids]
        self.jnt_dof_ids = [self.model.jnt_dofadr[id] for id in self.joint_ids]
        self.actuator_ids = [self.model_names.actuator_name2id[name] for name in actuator_names]
        
        self.goal_pos = None
        self.goal_ori_mat = None
        self.control_period = self.model.opt.timestep
        self.internal_period = 0.02

        self.spatial_controller = SpatialPDController(p_values, d_values)
        self.solver = JacobianTransposeSolver(len(self.joint_ids))

        # set goal to current end effector position and orientation
        self.reset_controller()

    def set_goal(self, goal_pos, goal_ori_mat):
        self.goal_pos = goal_pos
        self.goal_ori_mat = goal_ori_mat

    def reset_controller(self):
        self.goal_pos = get_site_xpos_by_id(self.model, self.data, self.eef_id)
        self.goal_ori_mat = get_site_xmat_by_id(self.model, self.data, self.eef_id)

        self.goal_twist = np.zeros(6)

        self.solver.sync_jnt_pos(self.get_joint_pos())

        self._compute_joint_control_cmds(np.zeros(6), self.internal_period)

    def run_controller(self):
        

        error = self._compute_error()

        joint_pos_cmds, joint_vel_cmds = self._compute_joint_control_cmds(error, self.control_period)

        return joint_pos_cmds, joint_vel_cmds
    
    def get_eef_pos(self):
        ee_pos = get_site_xpos_by_id(self.model, self.data, self.eef_id)
        ee_ori_mat = get_site_xmat_by_id(self.model, self.data, self.eef_id)
        return ee_pos, ee_ori_mat
    
    def get_joint_pos(self):
        return self.data.qpos[self.jnt_qpos_ids]
    
    def get_actuator_ids(self):
        return self.actuator_ids
    
    def _compute_error(self):
        # store end effector position and orientation using mujoco_utils
        ee_pos = get_site_xpos_by_id(self.model, self.data, self.eef_id)
        ee_ori_mat = get_site_xmat_by_id(self.model, self.data, self.eef_id)
        ee_pos_vel = get_site_xvelp_by_id(self.model, self.data, self.eef_id)
        ee_ori_vel = get_site_xvelr_by_id(self.model, self.data, self.eef_id)

        # new_ee_pos = set_goal_position(self.goal_twist[:3], ee_pos)
        # new_ee_ori_mat = set_goal_orientation(self.goal_twist[3:], ee_ori_mat)

        # compute error
        pos_error = compute_position_error(self.goal_pos, self.goal_ori_mat, ee_pos, ee_ori_mat)



        return pos_error
    
    def _compute_joint_control_cmds(self, error, period):

        # might want to do iterations thus have to update kinematic model
        # use this as resource https://github.com/deepmind/mujoco/issues/411

        # store jacobian matrices using mujoco_utils
        J_pos = get_site_jacp_by_id(self.model, self.data, self.eef_id)
        J_ori = get_site_jacr_by_id(self.model, self.data, self.eef_id)
        J_full = np.vstack([J_pos, J_ori])
        J = J_full[:, self.jnt_dof_ids]

        # compute control output
        cartesian_input = self.spatial_controller(error, self.internal_period)

        joint_pos_cmds, joint_vel_cmds = self.solver.compute_jnt_ctrl(J, self.internal_period, cartesian_input)

        return joint_pos_cmds, joint_vel_cmds