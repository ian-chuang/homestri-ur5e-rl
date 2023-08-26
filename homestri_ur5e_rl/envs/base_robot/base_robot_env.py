from os import path
import sys

import numpy as np
from homestri_ur5e_rl.controllers.operational_space_controller import ImpedanceController, ComplianceController, OperationalSpaceController, TargetType
from homestri_ur5e_rl.controllers.joint_effort_controller import JointEffortController
from homestri_ur5e_rl.controllers.joint_velocity_controller import JointVelocityController
from homestri_ur5e_rl.controllers.joint_position_controller import JointPositionController
from homestri_ur5e_rl.controllers.force_torque_sensor_controller import ForceTorqueSensorController
from gymnasium import spaces
from homestri_ur5e_rl.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames, robot_get_obs

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})


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

        self.init_qvel = self.data.qvel.copy()
        self.init_ctrl = self.data.ctrl.copy()


        config_path = path.join(
            path.dirname(__file__),
            "../assets/base_robot/robot_config.xml",
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
        self.model_names = MujocoModelNames(self.model) 


        
        

        # self.controller = ImpedanceController(
        #     model=self.model, 
        #     data=self.data, 
        #     model_names=self.model_names,
        #     eef_name='robot0:eef_site', 
        #     joint_names=[
        #         'robot0:ur5e:shoulder_pan_joint',
        #         'robot0:ur5e:shoulder_lift_joint',
        #         'robot0:ur5e:elbow_joint',
        #         'robot0:ur5e:wrist_1_joint',
        #         'robot0:ur5e:wrist_2_joint',
        #         'robot0:ur5e:wrist_3_joint',
        #     ],
        #     actuator_names=[
        #         'robot0:ur5e:shoulder_pan',
        #         'robot0:ur5e:shoulder_lift',
        #         'robot0:ur5e:elbow',
        #         'robot0:ur5e:wrist_1',
        #         'robot0:ur5e:wrist_2',
        #         'robot0:ur5e:wrist_3',
        #     ],
        #     min_effort=[-150, -150, -150, -150, -150, -150],
        #     max_effort=[150, 150, 150, 150, 150, 150],
        #     null_damp_kv=10,
        # )

        self.controller = OperationalSpaceController(
            model=self.model, 
            data=self.data, 
            model_names=self.model_names,
            eef_name='robot0:eef_site', 
            joint_names=[
                'robot0:ur5e:shoulder_pan_joint',
                'robot0:ur5e:shoulder_lift_joint',
                'robot0:ur5e:elbow_joint',
                'robot0:ur5e:wrist_1_joint',
                'robot0:ur5e:wrist_2_joint',
                'robot0:ur5e:wrist_3_joint',
            ],
            actuator_names=[
                'robot0:ur5e:shoulder_pan',
                'robot0:ur5e:shoulder_lift',
                'robot0:ur5e:elbow',
                'robot0:ur5e:wrist_1',
                'robot0:ur5e:wrist_2',
                'robot0:ur5e:wrist_3',
            ],
            min_effort=[-150, -150, -150, -150, -150, -150],
            max_effort=[150, 150, 150, 150, 150, 150],
            target_type=TargetType.TWIST,
            kp=200.0,
            ko=200.0,
            kv=50.0,
            vmax_xyz=0.2,
            vmax_abg=1,
            null_damp_kv=10,
        )


        # self.controller = JointEffortController(
        #     model=self.model, 
        #     data=self.data, 
        #     model_names=self.model_names,
        #     eef_name='robot0:eef_site', 
        #     joint_names=[
        #         'robot0:ur5e:shoulder_pan_joint',
        #         'robot0:ur5e:shoulder_lift_joint',
        #         'robot0:ur5e:elbow_joint',
        #         'robot0:ur5e:wrist_1_joint',
        #         'robot0:ur5e:wrist_2_joint',
        #         'robot0:ur5e:wrist_3_joint',
        #     ],
        #     actuator_names=[
        #         'robot0:ur5e:shoulder_pan',
        #         'robot0:ur5e:shoulder_lift',
        #         'robot0:ur5e:elbow',
        #         'robot0:ur5e:wrist_1',
        #         'robot0:ur5e:wrist_2',
        #         'robot0:ur5e:wrist_3',
        #     ],
        #     min_effort=[-150, -150, -150, -150, -150, -150],
        #     max_effort=[150, 150, 150, 150, 150, 150],
        # )

        # self.controller = JointVelocityController(
        #     model=self.model, 
        #     data=self.data, 
        #     model_names=self.model_names,
        #     eef_name='robot0:eef_site', 
        #     joint_names=[
        #         'robot0:ur5e:shoulder_pan_joint',
        #         'robot0:ur5e:shoulder_lift_joint',
        #         'robot0:ur5e:elbow_joint',
        #         'robot0:ur5e:wrist_1_joint',
        #         'robot0:ur5e:wrist_2_joint',
        #         'robot0:ur5e:wrist_3_joint',
        #     ],
        #     actuator_names=[
        #         'robot0:ur5e:shoulder_pan',
        #         'robot0:ur5e:shoulder_lift',
        #         'robot0:ur5e:elbow',
        #         'robot0:ur5e:wrist_1',
        #         'robot0:ur5e:wrist_2',
        #         'robot0:ur5e:wrist_3',
        #     ],
        #     min_effort=[-150, -150, -150, -150, -150, -150],
        #     max_effort=[150, 150, 150, 150, 150, 150],
        #     min_velocity=[-1, -1, -1, -1, -1, -1],
        #     max_velocity=[1, 1, 1, 1, 1, 1],
        #     kp=[100, 100, 100, 100, 100, 100],
        #     ki=0,
        #     kd=0,
        # )

        # self.controller = JointPositionController(
        #     model=self.model, 
        #     data=self.data, 
        #     model_names=self.model_names,
        #     eef_name='robot0:eef_site', 
        #     joint_names=[
        #         'robot0:ur5e:shoulder_pan_joint',
        #         'robot0:ur5e:shoulder_lift_joint',
        #         'robot0:ur5e:elbow_joint',
        #         'robot0:ur5e:wrist_1_joint',
        #         'robot0:ur5e:wrist_2_joint',
        #         'robot0:ur5e:wrist_3_joint',
        #     ],
        #     actuator_names=[
        #         'robot0:ur5e:shoulder_pan',
        #         'robot0:ur5e:shoulder_lift',
        #         'robot0:ur5e:elbow',
        #         'robot0:ur5e:wrist_1',
        #         'robot0:ur5e:wrist_2',
        #         'robot0:ur5e:wrist_3',
        #     ],
        #     min_effort=[-150, -150, -150, -150, -150, -150],
        #     max_effort=[150, 150, 150, 150, 150, 150],
        #     min_position=[-1, -1, -1, -1, -1, -1],
        #     max_position=[1, 1, 1, 1, 1, 1],
        #     kp=[100, 100, 100, 100, 100, 100],
        #     kd=[20, 20, 20, 20, 20, 20],
        # )


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



        # self.controller = ComplianceController(
        #     model=self.model, 
        #     data=self.data, 
        #     model_names=self.model_names,
        #     eef_name='robot0:eef_site', 
        #     joint_names=[
        #         'robot0:ur5e:shoulder_pan_joint',
        #         'robot0:ur5e:shoulder_lift_joint',
        #         'robot0:ur5e:elbow_joint',
        #         'robot0:ur5e:wrist_1_joint',
        #         'robot0:ur5e:wrist_2_joint',
        #         'robot0:ur5e:wrist_3_joint',
        #     ],
        #     actuator_names=[
        #         'robot0:ur5e:shoulder_pan',
        #         'robot0:ur5e:shoulder_lift',
        #         'robot0:ur5e:elbow',
        #         'robot0:ur5e:wrist_1',
        #         'robot0:ur5e:wrist_2',
        #         'robot0:ur5e:wrist_3',
        #     ],
        #     min_effort=[-150, -150, -150, -150, -150, -150],
        #     max_effort=[150, 150, 150, 150, 150, 150],
        #     min_velocity=[-1, -1, -1, -1, -1, -1],
        #     max_velocity=[1, 1, 1, 1, 1, 1],
        #     kp_jnt_vel=[100, 100, 100, 100, 100, 100],
        #     ki_jnt_vel=0,
        #     kd_jnt_vel=0,
        #     kp=[10, 10, 10, 10, 10, 10],
        #     ki=[0, 0, 0, 0, 0, 0],
        #     kd=[3, 3, 3, 8, 8, 8],
        #     control_period=self.model.opt.timestep,
        #     ft_sensor_site='robot0:eef_site',
        #     force_sensor_name='robot0:eef_force',
        #     torque_sensor_name='robot0:eef_torque',
        #     subtree_body_name='robot0:ur5e:wrist_3_link',
        #     ft_smoothing_factor=0,
        # )

    def step(self, action):
        # target_pose = np.array([-0.1,-0.6,1.1,0,1,0,0])
        # target_twist = np.array([0,0,0,0,0,0])
        # target_wrench = np.array([0,0,0,0,0,0])

        for i in range(self.frame_skip):
            ctrl = self.data.ctrl.copy()
            # self.controller.run(
            #     target_pose, 
            #     target_twist,
            #     target_wrench,
            #     np.array([200,200,200,200,200,200]),
            #     np.array([50,50,50,50,50,50]),
            #     1,
            #     ctrl
            # )

            print(action)

            self.controller.run(
                action[:6], 
                ctrl
            )

            ctrl[6] = action[6]

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
        self.set_state(qpos, qvel)
        obs = self._get_obs()

        return obs
