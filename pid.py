class PIDController:
    # sample enam vähem good values PIDController(Kp=0.5, Ki=0.1, Kd=0.05, dt=0.1)
    # out_angle = car_angle + pid.compute(ai_angle, car_angle)
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt

        self.integral = 0.0
        self.previous_error = 0.0

    def compute(self, desired_angle, current_angle):
        error = desired_angle - current_angle
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt

        control_output = (
            self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        )

        self.previous_error = error

        return control_output
