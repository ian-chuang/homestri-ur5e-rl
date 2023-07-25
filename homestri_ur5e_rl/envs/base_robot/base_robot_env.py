from os import path

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

from gymnasium_robotics.envs.franka_kitchen.utils import (
    get_config_root_node,
    read_config_from_node,
)
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames, robot_get_obs

MAX_CARTESIAN_DISPLACEMENT = 0.2
MAX_ROTATION_DISPLACEMENT = 0.5

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.2,
    "azimuth": 70.0,
    "elevation": -35.0,
    "lookat": np.array([-0.2, 0.5, 2.0]),
}


class BaseRobot(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 12,
    }

    def __init__(
        self,
        model_path="../assets/base_robot/scene.xml",
        frame_skip=40,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            model_path,
        )

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float64)

        super().__init__(
            xml_file_path,
            frame_skip,
            observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.init_qpos = self.data.qpos
        self.init_qvel = self.data.qvel

        config_path = path.join(
            path.dirname(__file__),
            "../assets/base_robot/robot_config.xml",
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
        self.model_names = MujocoModelNames(self.model)        

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        self.do_simulation(action, self.frame_skip)

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()

        return obs, 0.0, False, False, {}

    def _get_obs(self):
        # Gather simulated observation
        robot_qpos, robot_qvel = robot_get_obs(
            self.model, self.data, self.model_names.joint_names
        )

        return np.concatenate((robot_qpos.copy(), robot_qvel.copy()))

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        obs = self._get_obs()

        return obs
