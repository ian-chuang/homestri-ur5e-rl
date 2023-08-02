# Import necessary modules and classes
from homestri_ur5e_rl.controllers.joint_effort_controller import JointEffortController
from homestri_ur5e_rl.utils.mujoco_utils import (
    get_fullM,
)
from mujoco._structs import MjModel
from mujoco._structs import MjData
from homestri_ur5e_rl.utils.mujoco_utils import MujocoModelNames
from typing import List
import numpy as np

class JointPositionController(JointEffortController):
    def __init__(
        self,
        model: MjModel,
        data: MjData,
        model_names: MujocoModelNames,
        eef_name: str,
        joint_names: List[str],
        actuator_names: List[str],
        kp: List[float],  
        kd: List[float],
        min_effort: List[float],  
        max_effort: List[float],
        min_position: List[float],  
        max_position: List[float],  
    ) -> None:
        """
        Joint Position Controller class to control the robot's joints using PD control for position control with gravity compensation.

        Parameters:
            model (mujoco._structs.MjModel): Mujoco model representing the robot.
            data (mujoco._structs.MjData): Mujoco data associated with the model.
            model_names (homestri_ur5e_rl.utils.mujoco_utils.MujocoModelNames): Names of different components in the Mujoco model.
            eef_name (str): Name of the end-effector in the Mujoco model.
            joint_names (List[str]): List of joint names for the robot.
            actuator_names (List[str]): List of actuator names for the robot.
            kp (List[float]): Proportional gains for the PD controller for each joint.
            kd (List[float]): Derivative gains for the PD controller for each joint. A good starting point value for kd is sqrt(kp)
            min_effort (List[float]): List of minimum allowable effort (torque) for each joint.
            max_effort (List[float]): List of maximum allowable effort (torque) for each joint.
            min_position (List[float]): List of minimum allowable position for each joint.
            max_position (List[float]): List of maximum allowable position for each joint.
        """
        super().__init__(model, data, model_names, eef_name, joint_names, actuator_names, min_effort, max_effort)

        self.min_position = np.array(min_position)
        self.max_position = np.array(max_position)
        self.kp = np.array(kp) 
        self.kd = np.array(kd)

    def run(self, target: np.ndarray, ctrl: np.ndarray) -> None:
        """
        Run the joint position controller to control the robot's joints using PD control with gravity compensation.

        Parameters:
            target (numpy.ndarray): The desired target joint positions for the robot. It should have the same length as the number of controlled joints in the robot.
            ctrl (numpy.ndarray): Control signals for the robot actuators. It should have the same length as the number of actuators in the robot.

        Notes:
            The controller sets the control signals (efforts, i.e., controller joint torques) for the actuators specified by the 'actuator_ids' attribute based on the PD control with gravity compensation to achieve the desired joint positions specified in 'target'.
        """
        # Get the joint space inertia matrix
        M_full = get_fullM(self.model, self.data)
        M = M_full[self.jnt_dof_ids, :][:, self.jnt_dof_ids]

        # Clip the target joint positions to ensure they are within the allowable position range
        target_positions = np.clip(target, self.min_position, self.max_position)

        # Get the current joint positions and velocities from Mujoco
        current_positions = self.data.qpos[self.jnt_qpos_ids]
        current_velocities = self.data.qvel[self.jnt_dof_ids]

        # Calculate the error term for PD control
        position_error = target_positions - current_positions
        velocity_error = -current_velocities

        # Calculate the control effort (u) using PD control
        u = self.kp * position_error + self.kd * velocity_error

        # Apply the joint space inertia matrix to obtain the desired joint effort (torques)
        target_effort = np.dot(M, u)

        # Pass the calculated joint torques to the parent's run_controller
        super().run(target_effort, ctrl)


    def reset(self) -> None:
        """
        Reset the controller's internal state to its initial configuration.

        Notes:
            This method does not perform any actions for resetting the controller's state.
            It is intended to be overridden in the derived classes if any specific reset behavior is required.
        """
        pass  # The reset method does not perform any actions for this controller
