# Import necessary modules and classes
from mujoco._structs import MjModel
from mujoco._structs import MjData
from homestri_ur5e_rl.utils.mujoco_utils import MujocoModelNames
from typing import List
import numpy as np

class JointController:
    def __init__(
        self,
        model: MjModel,
        data: MjData,
        model_names: MujocoModelNames,
        eef_name: str,
        joint_names: List[str],
        actuator_names: List[str]
    ) -> None:
        """
        Base controller class to define a common interface for robot controllers.

        Parameters:
            model (mujoco._structs.MjModel): Mujoco model representing the robot.
            data (mujoco._structs.MjData): Mujoco data associated with the model.
            model_names (homestri_ur5e_rl.utils.mujoco_utils.MujocoModelNames): Names of different components in the Mujoco model.
            eef_name (str): Name of the end-effector in the Mujoco model.
            joint_names (List[str]): List of joint names for the robot.
            actuator_names (List[str]): List of actuator names for the robot.
        """
        self.model = model
        self.data = data
        self.model_names = model_names
        self.eef_name = eef_name
        self.joint_names = joint_names
        self.actuator_names = actuator_names

        self.n_joints = len(self.joint_names)
        # Get the site ID of the end-effector from its name in the Mujoco model
        self.eef_id = self.model_names.site_name2id[self.eef_name]
        # Get the joint IDs corresponding to the joint names
        self.joint_ids = [self.model_names.joint_name2id[name] for name in self.joint_names]
        # Get the qpos (position) IDs corresponding to the joint IDs
        self.jnt_qpos_ids = [self.model.jnt_qposadr[id] for id in self.joint_ids]
        # Get the dof (degree of freedom) IDs corresponding to the joint IDs
        self.jnt_dof_ids = [self.model.jnt_dofadr[id] for id in self.joint_ids]
        # Get the actuator IDs corresponding to the actuator names
        self.actuator_ids = [self.model_names.actuator_name2id[name] for name in actuator_names]

    def run(self, target: np.ndarray, ctrl: np.ndarray) -> None:
        """
        Run the robot controller.

        Parameters:
            target (numpy.ndarray): The desired target joint positions or states for the robot.
                                   The size of `target` should be (n_joints,) where n_joints is the number of robot joints.
            ctrl (numpy.ndarray): Control signals for the robot actuators from `mujoco._structs.MjData.ctrl` of size (nu,).
        """
        # Set the control signals for the actuators to the desired target joint positions or states
        ctrl[self.actuator_ids] = target

    def reset(self) -> None:
        """
        Reset the controller's internal state to its initial configuration.

        Notes:
            This method is intended to reset any internal variables or states of the controller.
        """
        # Raise a NotImplementedError to indicate that the method needs to be implemented in the derived classes
        raise NotImplementedError("The 'reset' method must be implemented in the derived class.")
