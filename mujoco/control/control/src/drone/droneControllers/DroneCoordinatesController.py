import threading
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import List
import os

from src.drone.droneControllers.TrajectoryPlanner import TrajectoryPlanner

matplotlib.use("Agg")
output_dir = "/home/nicolas/tfg/Memoria/GraficasPosicion"
os.makedirs(output_dir, exist_ok=True)


class CoordinatesController(threading.Thread):
    """Controller for drone's position and orientation: [x, y, z, yaw].
    Follows Daniel Mellinger's Differential flatness in  Trajectory generation and control for quadrotors.
    """

    def __init__(
        self,
        shutdownEvent: threading.Event,
        coordinatesUpdateEvent: threading.Event,
        coordinatesLock: threading.Lock,
        coordinatesData: dict,
        coordinatesAnglesLock: threading.Lock,
        coordinatesThrustLock: threading.Lock,
    ):
        super().__init__()
        self._shutdownEvent = shutdownEvent
        self._coordinatesUpdateEvent = coordinatesUpdateEvent
        self._coordinatesAnglesLock = coordinatesAnglesLock
        self._coordinatesThrustLock = coordinatesThrustLock
        self._coordinatesLock = coordinatesLock
        self._coordinatesData = coordinatesData
        self._timestamp = 0.0
        self._timestep = 0.001

        # Position -> Velocity PID Controller
        pKp = 3.35
        pKi = 0.0
        pKd = 0.0
        pIntegralLimit = np.inf
        self._MAX_VEL = 5.0
        self._positionPID = PIDController(
            kp=pKp,
            ki=pKi,
            kd=pKd,
            outputLimit=self._MAX_VEL,
            integralLimit=pIntegralLimit,
        )

        # Velocity -> Acceleration PID Controller
        vKp = 25.0
        vKi = 20.0
        vKd = 0.0
        vIntegralLimit = np.inf
        self._MAX_ACCEL = 5.0
        self._velocityPID = PIDController(
            kp=vKp,
            ki=vKi,
            kd=vKd,
            outputLimit=self._MAX_ACCEL,
            integralLimit=vIntegralLimit,
        )

        # Inputs
        self._currentTimestamp = 0.0
        self._timestep = 0.001
        self._currentPosition = np.array([0.0, 0.0, 0.0])
        self._currentTargetPosition = np.array([0.0, 0.0, 0.0])
        self._currentVelocity = np.array([0.0, 0.0, 0.0])  # Velocimeter
        self._currentTargetVelocity = np.array([0.0, 0.0, 0.0])
        self._currentAcceleration = np.array([0.0, 0.0, 0.0])  # Accelerometer
        self._gravity = np.array([0.0, 0.0, 9.81])

        # Control output
        self._refAngles = np.array([0.0, 0.0, 0.0])  # Roll, pitch, yaw (rad)
        self._baseThrust = 0.0

        # Store for analysis
        self._historyReferencePosition = []
        self._historyCurrentPosition = []
        self._historyReferenceVelocity = []
        self._historyCurrentVelocity = []
        self._historyReferenceAcceleration = []
        self._historyCurrentAcceleration = []
        self._historyControlAction = []
        self._historyTime = []
        self._historyRefAngles = []

    def run(self):
        print("[DroneCoordinatesController] Initialized.")
        iteration = 0
        previousTime = 0

        while not self._shutdownEvent.is_set():
            iteration += 1
            self.updateMovementReference()

            currentTime = self._timestamp
            dt = currentTime - previousTime if previousTime > 0 else self._timestep
            previousTime = currentTime

            if iteration == 1 or iteration % 20 == 0:
                # Coordinates controller: P controller whose output is the target velocity to the new position.
                # Will execute with a frequency of f = 20 Hz (given that timestep = 0.001)
                print(
                    f"[COORDINATES CONTROLLER] type of _currentTargetPosition: {type(self._currentTargetPosition)}"
                )
                print(
                    f"[COORDINATES CONTROLLER] type _currentPosition: {type(self._currentPosition)}"
                )
                errorPosition = self._currentTargetPosition - self._currentPosition
                desiredVelocity = (
                    self._positionPID.compute(error=errorPosition, dt=dt)
                    + self._currentTargetVelocity
                )

                # Velocity controller: P controller whose output is the target acceleration.
                # Will execute with a frequency of f = 50 Hz (given that timestep = 0.001)
                errorVelocity = desiredVelocity - self._currentVelocity
                desiredAcceleration = (
                    self._velocityPID.compute(error=errorVelocity, dt=dt)
                    + self._gravity
                )

            if iteration == 1 or iteration % 4 == 0:
                # Acceleration controller: PID controller whose output are the reference Euler angles for the stability controller
                # Will execute with a frequency of f = 250 Hz (given that timestep = 0.001)

                # yawRef = atan2(yref-y, xref-x)
                yawRef = np.atan2(
                    self._currentTargetPosition[1] - self._currentPosition[1],
                    self._currentTargetPosition[0] - self._currentPosition[0],
                )
                (referenceEulerAngles, thrust) = self.computeAttitudeAndThrust(
                    desiredAcceleration=desiredAcceleration, yawRef=yawRef
                )
                with self._coordinatesAnglesLock:
                    self._refAngles = np.array(referenceEulerAngles)

                with self._coordinatesThrustLock:
                    self._baseThrust = thrust

            # Store for analysis
            self._historyTime.append(currentTime)
            self._historyCurrentPosition.append(self._currentPosition.tolist())
            self._historyReferencePosition.append(self._currentTargetPosition.tolist())
            self._historyReferenceVelocity.append(self._currentTargetVelocity.tolist())
            self._historyCurrentVelocity.append(self._currentVelocity.tolist())
            self._historyReferenceAcceleration.append(
                self._currentTargetAcceleration.tolist()
            )
            self._historyCurrentAcceleration.append(self._currentAcceleration.tolist())
            self._historyRefAngles.append(self._refAngles.tolist())

        self.graphErrors()

    def stop(self):
        self._shutdownEvent.set()
        print("[DroneCoordinatesController] Shutting Down.")
        # self.graphErrors()

    def computeAttitudeAndThrust(self, desiredAcceleration, yawRef):
        # Unitary thrust vector
        zBody = desiredAcceleration / np.linalg.norm(desiredAcceleration)

        yaw = yawRef
        xC = np.array([np.cos(yaw), np.sin(yaw), 0])

        yBody = np.cross(zBody, xC)
        yBody /= np.linalg.norm(yBody)

        xBody = np.cross(yBody, zBody)

        # Rotation matrix
        R = np.vstack((xBody, yBody, zBody)).T

        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(-R[2, 0])

        thrust = np.linalg.norm(desiredAcceleration)

        return ([roll, pitch, yaw], thrust)

    def updateMovementReference(self):
        self._coordinatesUpdateEvent.wait()  # doubtful about the use of the event
        with self._coordinatesLock:
            self._currentTargetPosition = np.array(
                self._coordinatesData["positionReference"], dtype=np.float64
            )
            self._currentPosition = np.array(
                self._coordinatesData["currentPosition"], dtype=np.float64
            )
            self._currentVelocity = np.array(
                self._coordinatesData["currentVelocity"], dtype=np.float64
            )
            self._currentTargetVelocity = np.array(
                self._coordinatesData["currentTargetVelocity"], dtype=np.float64
            )
            self._currentAcceleration = np.array(
                self._coordinatesData["currentAcceleration"], dtype=np.float64
            )
            self._currentTargetAcceleration = np.array(
                self._coordinatesData["currentTargetAcceleration"], dtype=np.float64
            )
        self._coordinatesUpdateEvent.clear()

    # Accessed from DroneControl
    def setCoordinatesData(self, coordinatesData):
        with self._coordinatesLock:
            self._coordinatesData = coordinatesData

    def getAttitudeReference(self):
        with self._coordinatesAnglesLock:
            referenceEulerAngles = self._refAngles
        return referenceEulerAngles

    def getBaseThrust(self):
        with self._coordinatesThrustLock:
            baseThrust = self._baseThrust
        return baseThrust

    def graphErrors(self):
        if not self._historyTime or not self._historyRotorAdjustments:
            return

        rateErrors = np.array(self._historyRateError)
        angleErrors = np.array(self._historyAngleError)
        time = np.array(self._historyTime)
        labels = ["Roll", "Pitch", "Yaw"]

        # --- Angle Error Plot ---
        for i in range(3):
            plt.plot(time, angleErrors[:, i], label=labels[i])
        plt.title("Attitude Angle Control Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [rad]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "AttitudeAngleControl_Errors.png"))
        print("[Drone Stability Controller] AttitudeAngleControl_Errors.png saved")

        # --- Rate Error Plot ---
        for i in range(3):
            plt.plot(time, rateErrors[:, i], label=labels[i])
        plt.title("Attitude Rate Control Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [rad/s]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "AttitudeRateControl_Errors.png"))
        print("[Drone Stability Controller] AttitudeRateControl_Errors.png saved")

        reference_angles = np.array(self._historyReferenceAngles)
        current_angles = np.array(self._historyCurrentAngles)

        for i in range(3):
            plt.figure()
            plt.plot(time, reference_angles[:, i], "--", label=f"Ref {labels[i]}")
            plt.plot(time, current_angles[:, i], label=f"Actual {labels[i]}")
            plt.title(f"{labels[i]} Control")
            plt.xlabel("Time [s]")
            plt.ylabel("Angle [°]")
            plt.legend()
            plt.savefig(
                os.path.join(output_dir, f"{labels[i]}Control_OnlyCables_WithWind.png")
            )
            print(
                f"[Drone Stability Controller] {labels[i]}Control_OnlyCables_WithWind.png saved"
            )

        roll_output = [v[0] for v in self._historyRotorAdjustments]
        pitch_output = [v[1] for v in self._historyRotorAdjustments]
        yaw_output = [v[2] for v in self._historyRotorAdjustments]

        plt.figure()
        plt.plot(self._historyTime, roll_output, label="Roll Output")
        plt.plot(self._historyTime, pitch_output, label="Pitch Output")
        plt.plot(self._historyTime, yaw_output, label="Yaw Output")
        plt.xlabel("Time [s]")
        plt.ylabel("Control Output")
        plt.title("Control Output vs Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ControlOutput.png"))
        print("[Drone Stability Controller] ControlOutput.png saved")


class PIDController:
    def __init__(self, kp, ki, kd, outputLimit, integralLimit=np.inf):
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._integral = np.zeros(3)
        self._previousError = np.zeros(3)
        self._integralLimit = integralLimit
        self._outputLimit = outputLimit

    def reset(self):
        self._integral = np.zeros(3)
        self._previousError = np.zeros(3)

    def compute(self, error, dt):
        self._integral += error * dt
        # Anti-windup
        self._integral = np.clip(
            self._integral, -self._integralLimit, self._integralLimit
        )

        derivative = (error - self._previousError) / dt
        self._previousError = error

        output = self._kp * error + self._ki * self._integral + self._kd * derivative
        output = np.clip(output, -self._outputLimit, self._outputLimit)

        return output
