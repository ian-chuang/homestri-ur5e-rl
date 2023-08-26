from setuptools import setup, find_packages

setup(
    name='homestri_ur5e_rl',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'gymnasium-robotics',
        'mujoco',
        'gymnasium',
        'pynput',
    ],
    # Other information like author, author_email, description, etc.
)
