# Import necessary modules and classes
from mujoco._structs import MjModel
from mujoco._structs import MjData
import mujoco
from homestri_ur5e_rl.utils.mujoco_utils import MujocoModelNames
import numpy as np

class ForceTorqueSensorController:
    def __init__(
        self,
        model: MjModel,
        data: MjData,
        model_names: MujocoModelNames,
        ft_sensor_site: str,
        force_sensor_name: str,
        torque_sensor_name: str,
        subtree_body_name: str,
        smoothing_factor: float = 0.1,
    ) -> None:
        """
        Base controller class to define a common interface for robot controllers.

        Parameters:
            model (mujoco._structs.MjModel): Mujoco model representing the robot.
            data (mujoco._structs.MjData): Mujoco data associated with the model.
            model_names (homestri_ur5e_rl.utils.mujoco_utils.MujocoModelNames): Names of different components in the Mujoco model.
            ft_sensor_site (str): Name of the force-torque sensor site in the Mujoco model.
            force_sensor_name (str): Name of the force sensor in the Mujoco model.
            torque_sensor_name (str): Name of the torque sensor in the Mujoco model.
            subtree_body_name (str): Name of the body on which the force-torque sensor is mounted.
        """
        self.model = model
        self.data = data
        self.model_names = model_names
        self.ft_sensor_site = ft_sensor_site
        self.force_sensor_name = force_sensor_name
        self.torque_sensor_name = torque_sensor_name
        self.subtree_body_name = subtree_body_name
        self.smoothing_factor = smoothing_factor

        # Get IDs for the site, force sensor, torque sensor, and subtree body
        self.ft_site_id = self.model_names.site_name2id[self.ft_sensor_site]
        self.force_id = self.model_names.sensor_name2id[self.force_sensor_name]
        self.torque_id = self.model_names.sensor_name2id[self.torque_sensor_name]
        self.subtree_body_id = self.model_names.body_name2id[self.subtree_body_name]

        # Make sure the specified sensors are of the correct type
        assert model.sensor_type[self.force_id] == mujoco.mjtSensor.mjSENS_FORCE
        assert model.sensor_type[self.torque_id] == mujoco.mjtSensor.mjSENS_TORQUE

        # Get addresses for force and torque sensor data
        self.force_adr = model.sensor_adr[self.force_id]
        self.torque_adr = model.sensor_adr[self.torque_id]

        # Number of dimensions for force and torque vectors (typically 3 for each)
        self.force_ndim = 3
        self.torque_ndim = 3

        # Initialize zero signal offset as a 6-dimensional zero array
        self.zero_signal_offset = np.zeros(6)
        self.filtered_wrench = None

    def _gravity_compensation(self) -> np.ndarray:
        """
        Calculate the gravity compensation for the subtree.

        Returns:
            np.ndarray: Wrench containing the gravity compensation forces and torques in world frame (6-dimensional array).
        """

        # Get the center of mass of the specified body (CoM in world coordinates)
        subtree_com = self.data.subtree_com[self.subtree_body_id]

        # Get the mass of the subtree of the specified body
        subtree_mass = self.model.body_subtreemass[self.subtree_body_id]

        # Get the gravity vector from the physics model
        gravity = self.model.opt.gravity

        # Get the position of the FT sensor site in world coordinates
        ft_pos = self.data.site_xpos[self.ft_site_id]

        # Calculate the relative position vector from the FT sensor to the subtree's CoM
        relative_pos = subtree_com - ft_pos

        # The relative gravity acting on the subtree's CoM (typically just the gravity vector)
        relative_gravity = gravity

        # Calculate the gravitational force
        F_gravity = subtree_mass * (relative_gravity)

        # Calculate the gravitational torque
        T_gravity = np.cross(relative_pos, F_gravity)

        return np.concatenate([F_gravity, T_gravity])

    def zero_signal(self) -> None:
        """
        Capture the current wrench as the zero signal offset.
        """
        self.zero_signal_offset = self.get_wrench()
        self.filtered_wrench = None

    def get_wrench(self) -> np.ndarray:
        """
        Get the current wrench (force and torque) applied at the force-torque sensor.

        Returns:
            np.ndarray: Wrench containing the forces and torques applied at the force-torque sensor in world frame (6-dimensional array).
        """

        # Get the orientation matrix of the force-torque (FT) sensor
        ft_ori_mat = self.data.site_xmat[self.ft_site_id].reshape(3, 3)

        # Get the raw force and torque measurements from the sensor
        force = self.data.sensordata[self.force_adr : self.force_adr + self.force_ndim]
        torque = self.data.sensordata[self.torque_adr : self.torque_adr + self.torque_ndim]

        # Transform the force and torque from the sensor frame to the world frame
        force = ft_ori_mat @ force
        torque = ft_ori_mat @ torque

        # Concatenate the force and torque vectors to obtain the resultant wrench
        wrench = np.concatenate([force, torque])

        # Add the gravity compensation term to the wrench
        # wrench += self._gravity_compensation()

        # Subtract the zero signal offset from the wrench
        # wrench -= self.zero_signal_offset

        # Apply the smoothing filter
        # if self.filtered_wrench is None:
        #     self.filtered_wrench = wrench
        # else:
        #     self.filtered_wrench = (1 - self.smoothing_factor) * self.filtered_wrench + self.smoothing_factor * wrench

        # return self.filtered_wrench

        return wrench




