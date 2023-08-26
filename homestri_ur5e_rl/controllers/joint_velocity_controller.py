# Import necessary modules and classes
from homestri_ur5e_rl.controllers.joint_effort_controller import JointEffortController
from mujoco._structs import MjModel
from mujoco._structs import MjData
from homestri_ur5e_rl.utils.mujoco_utils import MujocoModelNames
from typing import List
import numpy as np

class JointVelocityController(JointEffortController):
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
        kp: List[float], 
        ki: List[float], 
        kd: List[float], 
        antiwindup: bool = False,
        max_integral: float = 10.0,
    ) -> None:
        """
        Joint Velocity Controller class to control the robot's joints using PID control for velocity control with gravity compensation.

        Parameters:
            model (mujoco._structs.MjModel): Mujoco model representing the robot.
            data (mujoco._structs.MjData): Mujoco data associated with the model.
            model_names (homestri_ur5e_rl.utils.mujoco_utils.MujocoModelNames): Names of different components in the Mujoco model.
            eef_name (str): Name of the end-effector in the Mujoco model.
            joint_names (List[str]): List of joint names for the robot.
            actuator_names (List[str]): List of actuator names for the robot.
            min_effort (List[float]): List of minimum allowable effort (torque) for each joint.
            max_effort (List[float]): List of maximum allowable effort (torque) for each joint.
            min_velocity (List[float]): List of minimum allowable velocity for each joint.
            max_velocity (List[float]): List of maximum allowable velocity for each joint.
            kp (List[float]): Proportional gains for the PID controller for each joint.
            kd (List[float]): Derivative gains for the PID controller for each joint.
            ki (List[float]): Integral gains for the PID controller for each joint.
            antiwindup (bool, optional): Enable anti-windup to limit the integral term. Defaults to False.
            max_integral (float, optional): Maximum allowed value for the integral term. Defaults to 10.0.
        """
        super().__init__(model, data, model_names, eef_name, joint_names, actuator_names, min_effort, max_effort)

        self.min_velocity = np.array(min_velocity)
        self.max_velocity = np.array(max_velocity)
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.ki = np.array(ki)
        self.antiwindup = antiwindup
        self.max_integral = max_integral

        # Initialize previous error and integral term arrays for each joint
        self.prev_error = np.zeros(self.n_joints)
        self.integral = np.zeros(self.n_joints)

    def run(self, target: np.ndarray, ctrl: np.ndarray) -> None:
        """
        Run the joint velocity controller to control the robot's joints using PID control with gravity compensation.

        Parameters:
            target (numpy.ndarray): The desired target joint velocities for the robot. It should have the same length as the number of controlled joints in the robot.
            ctrl (numpy.ndarray): Control signals for the robot actuators. It should have the same length as the number of actuators in the robot.

        Notes:
            The controller sets the control signals (efforts, i.e., controller joint torques) for the actuators specified by the 'actuator_ids' attribute based on the PID control and gravity compensation to achieve the desired joint velocities specified in 'target'.
        """
        # Clip the target joint velocities to ensure they are within the allowable velocity range
        target_velocities = np.clip(target, self.min_velocity, self.max_velocity)

        # Calculate the error term for PID control
        current_velocities = self.data.qvel[self.jnt_dof_ids]
        error = target_velocities - current_velocities

        # Integral term
        if self.antiwindup:
            self.integral += error
            # Anti-windup to limit the integral term
            self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        else:
            self.integral += error

        # Derivative term
        derivative = error - self.prev_error

        # Calculate the desired joint effort (torques) using PID control
        target_effort = self.kp * error + self.kd * derivative + self.ki * self.integral

        # Save the current error for the next iteration
        self.prev_error = error

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
