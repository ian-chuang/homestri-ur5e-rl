from homestri_ur5e_rl.controllers.joint_effort_controller import JointEffortController
from homestri_ur5e_rl.controllers.joint_velocity_controller import JointVelocityController
from homestri_ur5e_rl.controllers.force_torque_sensor_controller import ForceTorqueSensorController
from homestri_ur5e_rl.utils.mujoco_utils import (
    get_site_jac, 
    get_fullM,
)
from homestri_ur5e_rl.utils.transform_utils import (
    mat2quat,
)
from homestri_ur5e_rl.utils.controller_utils import (
    task_space_inertia_matrix,
    pose_error,
)
from homestri_ur5e_rl.utils.pid_controller_utils import (
    SpatialPIDController,
)
from mujoco._structs import MjModel
from mujoco._structs import MjData
from homestri_ur5e_rl.utils.mujoco_utils import MujocoModelNames
from typing import List
import numpy as np
from enum import Enum

class TargetType(Enum):
    POSE = 0
    TWIST = 1

class OperationalSpaceController(JointEffortController):
    def __init__(
        self,
        model: MjModel,
        data: MjData,
        model_names: MujocoModelNames,
        eef_name: str,
        joint_names: List[str],
        actuator_names: List[str],
        min_effort: List[float],
        max_effort: List[float],
        target_type: TargetType,
        kp: float,
        ko: float,
        kv: float,
        vmax_xyz: float,
        vmax_abg: float,
        null_damp_kv: float,
    ) -> None:
        """
        Operational Space Controller class to control the robot's joints using operational space control with gravity compensation.

        Parameters:
            model (mujoco._structs.MjModel): Mujoco model representing the robot.
            data (mujoco._structs.MjData): Mujoco data associated with the model.
            model_names (homestri_ur5e_rl.utils.mujoco_utils.MujocoModelNames): Names of different components in the Mujoco model.
            eef_name (str): Name of the end-effector in the Mujoco model.
            joint_names (List[str]): List of joint names for the robot.
            actuator_names (List[str]): List of actuator names for the robot.
            min_effort (List[float]): List of minimum allowable effort (torque) for each joint.
            max_effort (List[float]): List of maximum allowable effort (torque) for each joint.
            target_type (TargetType): The type of target input for the controller (POSE or TWIST).
            kp (float): Proportional gain for the PD controller in position space.
            ko (float): Proportional gain for the PD controller in orientation space.
            kv (float): Velocity gain for the PD controller.
            vmax_xyz (float): Maximum velocity for linear position control.
            vmax_abg (float): Maximum velocity for orientation control.
            ctrl_dof (List[bool]): Control degrees of freedom for each joint (True if controlled, False if uncontrolled).
            null_damp_kv (float): Damping gain for null space control.
        """
        super().__init__(model, data, model_names, eef_name, joint_names, actuator_names, min_effort, max_effort)

        self.target_type = target_type
        self.kp = kp
        self.ko = ko
        self.kv = kv
        self.vmax_xyz = vmax_xyz
        self.vmax_abg = vmax_abg
        self.null_damp_kv = null_damp_kv

        self.task_space_gains = np.array([self.kp] * 3 + [self.ko] * 3)
        self.lamb = self.task_space_gains / self.kv
        self.sat_gain_xyz = vmax_xyz / self.kp * self.kv
        self.sat_gain_abg = vmax_abg / self.ko * self.kv
        self.scale_xyz = vmax_xyz / self.kp * self.kv
        self.scale_abg = vmax_abg / self.ko * self.kv

    def run(self, target: np.ndarray, ctrl: np.ndarray) -> None:
        """
        Run the operational space controller to control the robot's joints using operational space control with gravity compensation.

        Parameters:
            target (numpy.ndarray): The desired target input for the controller.
            ctrl (numpy.ndarray): Control signals for the robot actuators.

        Notes:
            The controller sets the control signals (efforts, i.e., controller joint torques) for the actuators based on operational space control to achieve the desired target (either pose or twist).
        """
        # Get the Jacobian matrix for the end-effector.
        J = get_site_jac(self.model, self.data, self.eef_id)
        J = J[:, self.jnt_dof_ids]

        # Get the mass matrix and its inverse for the controlled degrees of freedom (DOF) of the robot.
        M_full = get_fullM(self.model, self.data)
        M = M_full[self.jnt_dof_ids, :][:, self.jnt_dof_ids]
        Mx, M_inv = task_space_inertia_matrix(M, J)

        # Get the joint velocities for the controlled DOF.
        dq = self.data.qvel[self.jnt_dof_ids]

        # Get the end-effector position, orientation matrix, and twist (spatial velocity).
        ee_pos = self.data.site_xpos[self.eef_id]
        ee_quat = mat2quat(self.data.site_xmat[self.eef_id].reshape(3, 3))
        ee_pose = np.concatenate([ee_pos, ee_quat])
        ee_twist = J @ dq

        # Initialize the task space control signal (desired end-effector motion).
        u_task = np.zeros(6)

        if self.target_type == TargetType.POSE:
            # If the target type is pose, the target contains both position and orientation.
            target_pose = target
            target_twist = np.zeros(6)

            # Scale the task space control signal while ensuring it doesn't exceed the specified velocity limits.
            u_task = self._scale_signal_vel_limited(pose_error(ee_pose, target_pose))

        elif self.target_type == TargetType.TWIST:
            # If the target type is twist, the target contains the desired spatial velocity.
            target_twist = target

        else:
            raise ValueError("Invalid target type: {}".format(self.target_type))

        # Initialize the joint effort control signal (controller joint torques).
        u = np.zeros(self.n_joints)

        if np.all(target_twist == 0):
            # If the target twist is zero (no desired motion), apply damping to the controlled DOF.
            u -= self.kv * np.dot(M, dq)
        else:
            # If the target twist is not zero, calculate the task space control signal error.
            u_task += self.kv * (ee_twist - target_twist)

        # Compute the joint effort control signal based on the task space control signal.
        u -= np.dot(J.T, np.dot(Mx, u_task))

        # Compute the null space control signal to minimize the joint efforts in the null space of the task.
        u_null = np.dot(M, -self.null_damp_kv * dq)
        Jbar = np.dot(M_inv, np.dot(J.T, Mx))
        null_filter = np.eye(self.n_joints) - np.dot(J.T, Jbar.T)
        u += np.dot(null_filter, u_null)

        # Call the parent class's run method to apply the computed joint efforts to the robot actuators.
        super().run(u, ctrl)

    def _scale_signal_vel_limited(self, u_task: np.ndarray) -> np.ndarray:
        """
        Scale the control signal such that the arm isn't driven to move faster in position or orientation than the specified vmax values.

        Parameters:
            u_task (numpy.ndarray): The task space control signal.

        Returns:
            numpy.ndarray: The scaled task space control signal.
        """
        norm_xyz = np.linalg.norm(u_task[:3])
        norm_abg = np.linalg.norm(u_task[3:])
        scale = np.ones(6)
        if norm_xyz > self.sat_gain_xyz:
            scale[:3] *= self.scale_xyz / norm_xyz
        if norm_abg > self.sat_gain_abg:
            scale[3:] *= self.scale_abg / norm_abg

        return self.kv * scale * self.lamb * u_task

class ImpedanceController(JointEffortController):
    def __init__(
        self,
        model: MjModel,
        data: MjData,
        model_names: MujocoModelNames,
        eef_name: str,
        joint_names: List[str],
        actuator_names: List[str],
        min_effort: List[float],
        max_effort: List[float],
        null_damp_kv: float,
    ) -> None:
        super().__init__(
            model, 
            data, 
            model_names, 
            eef_name, 
            joint_names, 
            actuator_names, 
            min_effort, 
            max_effort, 
        )
        self.null_damp_kv = null_damp_kv

    def run(
        self, 
        target_pose: np.ndarray,
        target_twist: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        ctrl: np.ndarray,
    ) -> None:
        # Get the Jacobian matrix for the end-effector.
        J_full = get_site_jac(self.model, self.data, self.eef_id)
        J = J_full[:, self.jnt_dof_ids]
        # Get the joint velocities 
        dq = self.data.qvel[self.jnt_dof_ids]
        # Get the mass matrix and its inverse for the controlled degrees of freedom (DOF) of the robot.
        M_full = get_fullM(self.model, self.data)
        M = M_full[self.jnt_dof_ids, :][:, self.jnt_dof_ids]
        Mx, M_inv = task_space_inertia_matrix(M, J)

        # Get the end-effector pose and twist (spatial velocity).
        ee_pos = self.data.site_xpos[self.eef_id]
        ee_quat = mat2quat(self.data.site_xmat[self.eef_id].reshape(3, 3))
        ee_pose = np.concatenate([ee_pos, ee_quat])
        ee_twist = J @ dq

        u_task_pos = kp * pose_error(target_pose, ee_pose)
        u_task_vel = kd * (target_twist -ee_twist)
        u_task = u_task_pos + u_task_vel

        # Compute the joint effort control signal based on the task space control signal.
        u = np.dot(J.T, np.dot(Mx, u_task))

        # Compute the null space control signal to minimize the joint efforts in the null space of the task.
        u_null = np.dot(M, -self.null_damp_kv * dq)
        Jbar = np.dot(M_inv, np.dot(J.T, Mx))
        null_filter = np.eye(self.n_joints) - np.dot(J.T, Jbar.T)
        u += np.dot(null_filter, u_null)

        # Call the parent class's run method to apply the computed joint efforts to the robot actuators.
        super().run(u, ctrl)  

class ComplianceController(JointVelocityController):
    def __init__(
        self,
        model: MjModel,
        data: MjData,
        model_names: MujocoModelNames,
        eef_name: str,
        joint_names: List[str],
        actuator_names: List[str],
        min_effort: List[float],
        max_effort: List[float],
        min_velocity: List[float],
        max_velocity: List[float],
        kp_jnt_vel: List[float], 
        ki_jnt_vel: List[float], 
        kd_jnt_vel: List[float], 
        kp: np.ndarray,
        ki: np.ndarray,
        kd: np.ndarray,
        control_period: float,
        ft_sensor_site: str,
        force_sensor_name: str,
        torque_sensor_name: str,
        subtree_body_name: str,
        ft_smoothing_factor: float,
    ) -> None:
        super().__init__(
            model, 
            data, 
            model_names, 
            eef_name, 
            joint_names, 
            actuator_names, 
            min_effort, 
            max_effort, 
            min_velocity,
            max_velocity,
            kp_jnt_vel,
            ki_jnt_vel,
            kd_jnt_vel,
        )
        self.ft_sensor = ForceTorqueSensorController(
            model,
            data,
            model_names,
            ft_sensor_site,
            force_sensor_name,
            torque_sensor_name,
            subtree_body_name,
            ft_smoothing_factor,
        )
        self.spatial_controller = SpatialPIDController(kp, ki, kd)
        self.control_period = control_period

    def run(
        self, 
        target_pose: np.ndarray,
        target_twist: np.ndarray,
        target_wrench: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
        error_scale: np.ndarray,
        ctrl: np.ndarray,
    ) -> None:
        # Get the Jacobian matrix for the end-effector.
        J_full = get_site_jac(self.model, self.data, self.eef_id)
        J = J_full[:, self.jnt_dof_ids]
        # Get the joint velocities 
        dq = self.data.qvel[self.jnt_dof_ids]

        # Get the end-effector pose and twist (spatial velocity).
        ee_pos = self.data.site_xpos[self.eef_id]
        ee_quat = mat2quat(self.data.site_xmat[self.eef_id].reshape(3, 3))
        ee_pose = np.concatenate([ee_pos, ee_quat])
        ee_twist = J @ dq

        # ee_wrench = self.ft_sensor.get_wrench()

        u_task_pos = stiffness * pose_error(target_pose, ee_pose)
        u_task_vel = damping * (target_twist - ee_twist)
        # u_task_eff = target_wrench - ee_wrench

        # print(f"u_task_pos: {u_task_pos}, u_task_vel: {u_task_vel}, u_task_eff: {u_task_eff}")
        u_task = error_scale * (u_task_pos + u_task_vel)

        
        u_task = self.spatial_controller(u_task, self.control_period)

        # print("u_task", "{0:0.7f}".format(u_task[2]))
        # print("u_task_eff", "{0:0.7f}".format(u_task_eff[2]))
        # print("ee_wrench", "{0:0.7f}".format(ee_wrench[2]))

        # Compute the joint effort control signal based on the task space control signal.
        u = np.dot(J.T, u_task)

        u = 0.5 * u * self.control_period

        # Call the parent class's run method to apply the computed joint efforts to the robot actuators.
        super().run(u, ctrl)  
