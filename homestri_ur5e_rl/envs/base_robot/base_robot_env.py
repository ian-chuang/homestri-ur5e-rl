from os import path
import sys

import numpy as np
from homestri_ur5e_rl.controllers.operational_space_controller import OperationalSpaceController
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames, robot_get_obs

from homestri_ur5e_rl.utils.transform_utils import (
    mat2euler,
    euler2mat,
)


MAX_CARTESIAN_DISPLACEMENT = 0.2
MAX_ROTATION_DISPLACEMENT = 0.5

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.2,
    "azimuth": 0.0,
    "elevation": -20.0,
    "lookat": np.array([0, 0, 1]),
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

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        self.init_ctrl = self.data.ctrl.copy()


        config_path = path.join(
            path.dirname(__file__),
            "../assets/base_robot/robot_config.xml",
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
        self.model_names = MujocoModelNames(self.model) 

        self.controller = OperationalSpaceController(
            self.model, 
            self.data, 
            self.model_names,
            'robot0:eef_site', 
            [
                'robot0:ur5e:shoulder_pan_joint',
                'robot0:ur5e:shoulder_lift_joint',
                'robot0:ur5e:elbow_joint',
                'robot0:ur5e:wrist_1_joint',
                'robot0:ur5e:wrist_2_joint',
                'robot0:ur5e:wrist_3_joint',
            ],
            [
                'robot0:ur5e:shoulder_pan',
                'robot0:ur5e:shoulder_lift',
                'robot0:ur5e:elbow',
                'robot0:ur5e:wrist_1',
                'robot0:ur5e:wrist_2',
                'robot0:ur5e:wrist_3',
            ],
            [500, 500, 500, 500, 500, 500],
            [0, 0, 0, 0, 0, 0],
        )


        self.init_qpos_config = {
            "robot0:ur5e:shoulder_pan_joint": 0,
            "robot0:ur5e:shoulder_lift_joint": -np.pi / 2.0,
            "robot0:ur5e:elbow_joint": -np.pi / 2.0,
            "robot0:ur5e:wrist_1_joint": -np.pi / 2.0,
            "robot0:ur5e:wrist_2_joint": np.pi / 2.0,
            "robot0:ur5e:wrist_3_joint": 0,
        }
        for joint_name, joint_pos in self.init_qpos_config.items():
            joint_id = self.model_names.joint_name2id[joint_name]
            qpos_id = self.model.jnt_qposadr[joint_id]
            self.init_qpos[qpos_id] = joint_pos

        self.init_ctrl_config = {
            "robot0:ur5e:shoulder_pan": 0,
            "robot0:ur5e:shoulder_lift": -np.pi / 2.0,
            "robot0:ur5e:elbow": -np.pi / 2.0,
            "robot0:ur5e:wrist_1": -np.pi / 2.0,
            "robot0:ur5e:wrist_2": np.pi / 2.0,
            "robot0:ur5e:wrist_3": 0,
        }
        for actuator_name, actuator_pos in self.init_ctrl_config.items():
            actuator_id = self.model_names.actuator_name2id[actuator_name]
            self.init_ctrl[actuator_id] = actuator_pos


    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        pos = np.array([-0.134, -.1, 1.4])
        ori =  np.array([0, np.pi, 0])
        
        self.controller.set_goal(pos, euler2mat(ori))

        for i in range(self.frame_skip):

            joint_pos, joint_vel = self.controller.run_controller()

            ctrl = np.zeros(self.model.nu)
            ctrl[self.controller.get_actuator_ids()] = joint_pos
            

            self.do_simulation(ctrl, n_frames=1)

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
        
        self.data.ctrl[:] = self.init_ctrl
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        self.controller.reset_controller()

        return obs
