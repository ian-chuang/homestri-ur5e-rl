# Import necessary modules and classes
from homestri_ur5e_rl.controllers.joint_controller import JointController
from mujoco._structs import MjModel
from mujoco._structs import MjData
from homestri_ur5e_rl.utils.mujoco_utils import MujocoModelNames
from typing import List
import numpy as np

class JointEffortController(JointController):
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
    ) -> None:
        """
        Joint Effort Controller class to control the robot's joints by specifying joint efforts (controller joint torques).

        Parameters:
            model (mujoco._structs.MjModel): Mujoco model representing the robot.
            data (mujoco._structs.MjData): Mujoco data associated with the model.
            model_names (homestri_ur5e_rl.utils.mujoco_utils.MujocoModelNames): Names of different components in the Mujoco model.
            eef_name (str): Name of the end-effector in the Mujoco model.
            joint_names (List[str]): List of joint names for the robot.
            actuator_names (List[str]): List of actuator names for the robot.
            min_effort (List[float]): List of minimum allowable effort (torque) for each joint.
            max_effort (List[float]): List of maximum allowable effort (torque) for each joint.
        """
        super().__init__(model, data, model_names, eef_name, joint_names, actuator_names)

        # Convert the minimum and maximum effort lists to NumPy arrays for easier manipulation
        self.min_effort = np.array(min_effort)
        self.max_effort = np.array(max_effort)

    def run(self, target: np.ndarray, ctrl: np.ndarray) -> None:
        """
        Run the joint effort controller to control the robot's joints.

        Parameters:
            target (numpy.ndarray): The desired target efforts (controller joint torques) for the actuators.
                                   It should have the same length as the number of actuators in the robot.
            ctrl (numpy.ndarray): Control signals for the robot actuators.
                                  It should have the same length as the number of actuators in the robot.

        Notes:
            - The controller clamps the target efforts to ensure they are within the allowable effort range.
              If a target effort exceeds the specified maximum or minimum effort for a joint, it will be clipped to the corresponding bound.
        """

        # Clip the target efforts to ensure they are within the allowable effort range
        target_effort = np.clip(target, self.min_effort, self.max_effort)
        
        # Get the gravity compensation for each joint
        gravity_compensation = self.data.qfrc_bias[self.jnt_dof_ids]

        # Add gravity compensation to the target effort
        target_effort += gravity_compensation

        # Call the base controller's `run_controller` method with the clipped target efforts
        super().run(target_effort, ctrl)

    def reset(self) -> None:
        """
        Reset the controller's internal state to its initial configuration.

        Notes:
            This method does not perform any actions for resetting the controller's state.
            It is intended to be overridden in the derived classes if any specific reset behavior is required.
        """
        pass  # The reset method does not perform any actions for this controller
