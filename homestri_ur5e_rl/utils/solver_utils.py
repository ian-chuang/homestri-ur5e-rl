import numpy as np

class JacobianTransposeSolver():
    def __init__(self, number_joints):
        self.last_positions = np.zeros(number_joints)

    def sync_jnt_pos(self, jnt_pos):
        self.last_positions = jnt_pos

    def compute_jnt_ctrl(self, jacobian, period, net_force):
        # Compute joint jacobian
        jacobian_transpose = np.transpose(jacobian)

        # Compute joint accelerations according to: \ddot{q} = H^{-1} ( J^T f)
        current_accelerations = np.dot(jacobian_transpose, net_force)

        # Integrate once, starting with zero motion
        current_velocities = 0.5 * current_accelerations * period

        # Integrate twice, starting with zero motion
        current_positions = self.last_positions + 0.5 * current_velocities * period

        # Update for the next cycle
        self.last_positions = current_positions

        return current_positions, current_velocities