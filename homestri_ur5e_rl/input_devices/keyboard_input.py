from pynput import keyboard
import numpy as np

class KeyboardInput:
    def __init__(self):
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.gripper_position = 0
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            key_char = key.char.lower()
            if key_char in ['w', 'a', 's', 'd', 'q', 'e']:
                linear_mapping = {
                    'w': [0.2, 0, 0],
                    's': [-0.2, 0, 0],
                    'a': [0, 0.2, 0],
                    'd': [0, -0.2, 0],
                    'q': [0, 0, 0.2],
                    'e': [0, 0, -0.2],
                }
                self.linear_velocity = np.array(linear_mapping[key_char])
            elif key_char in ['i', 'j', 'k', 'l', 'u', 'o']:
                angular_mapping = {
                    'i': [0.5, 0, 0],
                    'k': [-0.5, 0, 0],
                    'j': [0, 0.5, 0],
                    'l': [0, -0.5, 0],
                    'u': [0, 0, 0.5],
                    'o': [0, 0, -0.5],
                }
                self.angular_velocity = np.array(angular_mapping[key_char])

        except AttributeError:
            if key == keyboard.Key.space:
                # Toggle gripper position between 0 and 255
                self.gripper_position = 255 if self.gripper_position == 0 else 0

    def on_release(self, key):
        try:
            key_char = key.char.lower()
            if key_char in ['w', 'a', 's', 'd', 'q', 'e']:
                self.linear_velocity = np.zeros(3)
            elif key_char in ['i', 'j', 'k', 'l', 'u', 'o']:
                self.angular_velocity = np.zeros(3)
        except AttributeError:
            pass

    def get_action(self):
        action = np.concatenate((self.linear_velocity, self.angular_velocity, np.array([self.gripper_position])))
        return action