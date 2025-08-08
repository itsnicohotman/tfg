import numpy as np


class PIDController:
    def __init__(self, kp: np.array, ki: np.array, kd: np.array, outputLimit):
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._size = kp.size
        self._integral = np.zeros(self._size)
        self._previousError = np.zeros(self._size)
        self._outputLimit = outputLimit

    def reset(self):
        self._integral = np.zeros(self._size)
        self._previousError = np.zeros(self._size)

    def compute(self, error, dt):
        self._integral += error * dt

        derivative = (error - self._previousError) / dt
        self._previousError = error

        output = self._kp * error + self._ki * self._integral + self._kd * derivative

        # Anti-windup
        for i in range(len(output)):
            if output[i] > self._outputLimit:
                output[i] = self._outputLimit
                self._integral[i] -= error[i] * dt
            elif output[i] < -self._outputLimit:
                output[i] = -self._outputLimit
                self._integral[i] -= error[i] * dt

        return np.array(output)
