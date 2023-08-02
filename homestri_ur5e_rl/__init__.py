from gymnasium.envs.registration import register

register(
    id="BaseRobot-v0",
    entry_point="homestri_ur5e_rl.envs.base_robot:BaseRobot",
)