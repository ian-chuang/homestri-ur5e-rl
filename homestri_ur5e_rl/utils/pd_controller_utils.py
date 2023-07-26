import numpy as np

class PDController:
    def __init__(self, kp, kd):
        self.kp = kp  # Proportional gain
        self.kd = kd  # Derivative gain
        self.prev_error = 0.0

    def __call__(self, error, dt):
        # Calculate the derivative of the error
        error_derivative = (error - self.prev_error) / dt
        
        # Calculate the control output
        control_output = self.kp * error + self.kd * error_derivative
        
        # Store the current error for the next iteration
        self.prev_error = error
        
        return control_output


class SpatialPDController:
    def __init__(self, kp_values=None, kd_values=None):
        self.m_cmd = np.zeros(6)
        if kp_values is None:
            kp_values = [0] * 6
        if kd_values is None:
            kd_values = [0] * 6
        self.m_pd_controllers = [PDController(kp_values[i], kd_values[i]) for i in range(6)]

    def set_pd_gains(self, kp_values, kd_values):
        # Set the P and D gains for each dimension
        for i in range(6):
            self.m_pd_controllers[i].kp = kp_values[i]
            self.m_pd_controllers[i].kd = kd_values[i]

    def __call__(self, error, period):
        # Call operator for one control cycle
        controlled_output = np.zeros(6)

        for i in range(6):
            # Apply each PD controller for each element in the error vector
            controlled_output[i] = self.m_pd_controllers[i](error[i], period)

        return controlled_output