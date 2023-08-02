import gymnasium as gym
import homestri_ur5e_rl
from homestri_ur5e_rl.input_devices.keyboard_input import (
    KeyboardInput,
)

keyboard_input = KeyboardInput()

env = gym.make("BaseRobot-v0", render_mode="human")

observation, info = env.reset(seed=42)
while True:
    action = env.action_space.sample()

    action = keyboard_input.get_action()

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

