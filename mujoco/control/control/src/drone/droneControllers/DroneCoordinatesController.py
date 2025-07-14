import threading
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import os

from src.drone.droneControllers.TrajectoryPlanner import TrajectoryPlanner
from src.drone.droneControllers.DroneStabilityController import DroneStabilityController


class CoordinatesController(threading.Thread):
    """Controller for drone's position: [x, y].
    Follows Daniel Mellinger's Differential flatness in  Trajectory generation and control for quadrotors as well as PX4's Position Control.
    Position, Velocity and Acceleration controller at 50Hz.
    """

    def __init__(
        self,
        shutdownEvent: threading.Event,
        coordinatesUpdateEvent: threading.Event,
        coordinatesLock: threading.Lock,
        coordinatesData: dict,
        coordinatesAnglesLock: threading.Lock,
        # coordinatesThrustLock: threading.Lock,
    ):
        super().__init__()
        self._shutdownEvent = shutdownEvent
        self._coordinatesUpdateEvent = coordinatesUpdateEvent
        self._coordinatesAnglesLock = coordinatesAnglesLock
        # self._coordinatesThrustLock = coordinatesThrustLock
        self._coordinatesLock = coordinatesLock
        self._coordinatesData = coordinatesData

        # Position -> Velocity PID Controller
        pKp = np.array([3, 3, 0.0], dtype=np.float64)  # Kp for x, y [3 3 0]
        pKi = np.array([2.3, 2.3, 0.0], dtype=np.float64)  # Ki for x, y [2.3 2.3 0]
        pKd = np.array([0.05, 0.05, 0.0], dtype=np.float64)  # Kd for x, y [0.05 0.05 0]
        # ZIEGLER-NICHOLS: Pitch: Kc = 4, Tc = 2 --> PID: Kp = 0.59Kc, Ki = 1.18*Kc/Tc, Kd = 0.075KcTc
        self._MAX_VEL = 1.0
        self._positionPID = PIDController(
            kp=pKp,
            ki=pKi,
            kd=pKd,
            outputLimit=self._MAX_VEL,
        )

        # Velocity -> Acceleration PID Controller
        vKp = np.array([4, 4, 0.0], dtype=np.float64)  # Kp for x, y velocities [4 4 0]
        vKi = np.array([5, 5, 0.0], dtype=np.float64)  # Ki for x, y velocities [5 5 0]
        vKd = np.array(
            [0.0085, 0.0085, 0.0], dtype=np.float64
        )  # Kd for x, y velocities [0.0085 0.0085 0]
        self._MAX_ACCEL = 2.0
        self._velocityPID = PIDController(
            kp=vKp,
            ki=vKi,
            kd=vKd,
            outputLimit=self._MAX_ACCEL,
        )

        # Inputs
        self._currentTimestamp = 0.0
        self._timestep = 0.001
        self._currentPosition = np.zeros(3)
        self._currentTargetPosition = np.zeros(3)
        self._currentVelocity = np.zeros(3)  # Velocimeter
        self._currentTargetVelocity = np.zeros(3)
        self._currentAcceleration = np.zeros(3)  # Accelerometer
        self._currentOrientation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion
        self._currentAngles = np.zeros(3)  # Roll, pitch, yaw (rad)

        # Control output
        self._refAngles = np.zeros(3)  # Roll, pitch, yaw (rad)
        # self._baseThrust = 0.0
        # self._MAX_THRUST = 30

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
        # self._historyThrust = []

    def run(self):
        print("[DroneCoordinatesController] Initialized.")
        iteration = 0
        previousTime = 0

        while not self._shutdownEvent.is_set():
            iteration += 1
            self.updateState()

            currentTime = self._currentTimestamp
            dt = currentTime - previousTime if previousTime > 0 else self._timestep
            previousTime = currentTime

            if iteration == 1 or iteration % 20 == 0:
                targetVelocity = np.zeros(3)
                # Coordinates controller: P controller whose output is the target velocity to the new position.
                # Will execute with a frequency of f = 50 Hz (given that timestep = 0.001)
                print(f"[COORD CONTR] Current Position: {self._currentPosition}")
                print(f"[COORD CONTR] Target Position: {self._currentTargetPosition}")
                errorPosition = self._currentTargetPosition - self._currentPosition
                mask = np.abs(errorPosition) > 0.1
                targetVelocity[mask] = self._currentTargetVelocity[mask] * np.sign(
                    errorPosition[mask]
                )

                desiredVelocity = (
                    self._positionPID.compute(error=errorPosition, dt=dt)
                    + targetVelocity
                )
                print(f"[COORD CONTR] Desired Velocity: {desiredVelocity}")

                # Velocity controller: P controller whose output is the target acceleration.
                # Will execute with a frequency of f = 50 Hz (given that timestep = 0.001)
                errorVelocity = desiredVelocity - self._currentVelocity
                desiredAcceleration = np.array(
                    self._velocityPID.compute(error=errorVelocity, dt=dt)
                )
                print(f"[COORD CONTR] Desired Acceleration: {desiredAcceleration}")

                # if iteration == 1 or iteration % 10 == 0:
                # Acceleration controller: PID controller whose output are the reference Euler angles for the stability controller
                # Will execute with a frequency of f = 100 Hz (given that timestep = 0.001)

                referenceEulerAngles = self.computeAttitudeAndThrust(
                    desiredAcceleration=desiredAcceleration
                )
                with self._coordinatesAnglesLock:
                    self._refAngles = np.array(referenceEulerAngles)

                if iteration % 10 == 0:
                    print(f"[COORD CONTR] Ref. Angles: {referenceEulerAngles}")

                # with self._coordinatesThrustLock:
                #    self._baseThrust = thrust

            # Store for analysis
            self._historyTime.append(currentTime)
            self._historyCurrentPosition.append(self._currentPosition.tolist())
            self._historyReferencePosition.append(self._currentTargetPosition.tolist())
            self._historyReferenceVelocity.append(self._currentTargetVelocity.tolist())
            self._historyCurrentVelocity.append(self._currentVelocity.tolist())
            self._historyReferenceAcceleration.append(desiredAcceleration.tolist())
            self._historyCurrentAcceleration.append(self._currentAcceleration.tolist())
            self._historyRefAngles.append(self._refAngles.tolist())
            # self._historyThrust.append(self._baseThrust)

        self.graphErrors()

    def stop(self):
        self._shutdownEvent.set()
        print("[DroneCoordinatesController] Shutting Down.")
        self.graphErrors()

    def computeAttitudeAndThrust(self, desiredAcceleration):
        """# yawRef = atan2(yref-y, xref-x) (rad)
        deltaX = self._currentTargetPosition[0] - self._currentPosition[0]
        deltaY = self._currentTargetPosition[1] - self._currentPosition[1]

        if deltaX <= 1e-5:  # Avoid division by zero
            yaw = self._currentAngles[2]  # Keep current yaw
        else:
            yaw = np.atan2(deltaY, deltaX)
            if yaw > np.pi:
                yaw -= 2 * np.pi
            elif yaw < -np.pi:
                yaw += 2 * np.pi"""
        # Calculate the reference angles (roll, pitch, yaw) based on the desired acceleration
        MAX_ROLL = 0.52  # 30 degrees in radians
        MAX_PITCH = 0.52  # 30 degrees in
        gravity = 9.81
        # print(f"[COORD CONTR] Desired Acceleration: {desiredAcceleration}")
        """ desiredAcceleration = np.concatenate(
            desiredAcceleration[:2], np.array([gravity])
        ) """
        desiredAcceleration[2] = gravity
        # print(f"[COORD CONTR] Adjusted Desired Acceleration: {desiredAcceleration}")

        yaw = 0  # Assume no yaw change for simplicity

        """ pitch = desiredAcceleration[0] / gravity
        roll = -desiredAcceleration[1] / gravity """

        pitch = np.arcsin(desiredAcceleration[0] / np.linalg.norm(desiredAcceleration))
        pitch = np.clip(pitch, -MAX_PITCH, MAX_PITCH)
        roll = np.arcsin(-desiredAcceleration[1] / np.linalg.norm(desiredAcceleration))
        roll = np.clip(roll, -MAX_ROLL, MAX_ROLL)

        """ # Unitary thrust vector
        zBody = self.normalizeVector(desiredAcceleration)
        xC = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
        # print(f"[COORD CONTR] xC: {xC}")

        yBody = self.normalizeVector(np.cross(zBody, xC))

        # print(f"[COORD CONTR] zBody: {zBody}, yBody: {yBody}")
        xBody = np.cross(yBody, zBody)

        # Rotation matrix
        R = np.column_stack((xBody, yBody, zBody)).T

        print(f"[COORD CONTR] Rotation Matrix: {R}")
        # Roll is calculated from the second column of the rotation matrix
        # R[2, 1] = sin(roll)
        # R[2, 2] = cos(roll)
        roll = np.arctan2(R[2, 1], R[2, 2])
        roll = np.clip(roll, -MAX_ROLL, MAX_ROLL)

        # Pitch is calculated from the first column of the rotation matrix
        # R[2, 0] = -sin(pitch)
        pitch = np.arcsin(-R[2, 0])
        print(f"[COORD CONTR] Pitch before clipping: {pitch}")
        pitch = np.clip(pitch, -MAX_PITCH, MAX_PITCH)
        print(f"[COORD CONTR] Pitch after clipping: {pitch}") """

        """ thrust = np.clip(
            np.linalg.norm(desiredAcceleration), -self._MAX_THRUST, self._MAX_THRUST
        ) """
        referenceAngles = np.array([roll, pitch, yaw], dtype=np.float64)
        mask = np.abs(referenceAngles) <= 1e-5
        referenceAngles[mask] = 0.0

        return referenceAngles
        # return (referenceAngles, thrust)

    def updateState(self):
        self._coordinatesUpdateEvent.wait()
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
            self._currentTimestamp = self._coordinatesData["timestamp"]
            self._timestep = self._coordinatesData["timestep"]
            self._currentOrientation = np.array(
                self._coordinatesData["currentOrientation"], dtype=np.float64
            )
        self._currentAngles = DroneStabilityController.quaternionToEuler(
            self._currentOrientation
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

    # def getBaseThrust(self):
    #    with self._coordinatesThrustLock:
    #        baseThrust = self._baseThrust
    #    return baseThrust

    def graphErrors(self):
        if not self._historyTime:
            return

        output_dir = "/home/nicolas/tfg/Memoria/GraficasPosicion"
        os.makedirs(output_dir, exist_ok=True)

        # Coordinates graph
        time = np.array(self._historyTime)
        lengthTime = len(time)
        historyPosition = np.array(self._historyCurrentPosition[:lengthTime])
        historyPositionReference = np.array(self._historyReferencePosition[:lengthTime])
        historyVelocity = np.array(self._historyCurrentVelocity[:lengthTime])
        historyVelocityReference = np.array(self._historyReferenceVelocity[:lengthTime])
        historyAcceleration = np.array(self._historyCurrentAcceleration[:lengthTime])
        historyAccelerationReference = np.array(
            self._historyReferenceAcceleration[:lengthTime]
        )
        historyPositionError = historyPositionReference - historyPosition
        historyVelocityError = historyVelocityReference - historyVelocity
        historyAccelerationError = historyAccelerationReference - historyAcceleration
        # historyRotorForce = np.array(self._historyThrust[:lengthTime])
        referenceAngles = np.array(self._historyRefAngles[:lengthTime])
        labelsCart = ["X", "Y"]
        labelsAngles = ["Roll", "Pitch", "Yaw"]

        # Position error graph - dummy
        plt.figure()
        plt.clf()
        for i in range(2):
            plt.plot(time, historyPositionError[:, i], label=labelsCart[i])
        plt.title("Position Control Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [m]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "Position_Errors_dummy.png"))
        print("[Drone Coordinates Controller] Position_Errors_dummy.png saved")

        # Position error graph
        plt.figure()
        plt.clf()
        for i in range(2):
            plt.plot(time, historyPositionError[:, i], label=labelsCart[i])
        plt.title("Position Control Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [m]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "Position_Errors.png"))
        print("[Drone Coordinates Controller] Position_Errors.png saved")

        # Velocity error graph
        plt.figure()
        plt.clf()
        for i in range(2):
            plt.plot(time, historyVelocityError[:, i], label=labelsCart[i])
        plt.title("Velocity Control Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [m]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "Velocity_Errors.png"))
        print("[Drone Coordinates Controller] Velocity_Errors.png saved")

        # Acceleration error graph
        plt.figure()
        plt.clf()
        for i in range(2):
            plt.plot(time, historyAccelerationError[:, i], label=labelsCart[i])
        plt.title("Acceleration Control Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [m]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "Acceleration_Errors.png"))
        print("[Drone Coordinates Controller] Acceleration_Errors.png saved")

        # Thrust graph
        # plt.figure()
        # plt.clf()
        # plt.plot(time, historyRotorForce, label="Rotor Force")
        # plt.title("Position Control - base thrust per rotor")
        # plt.xlabel("Time [s]")
        # plt.ylabel("Rotor Force [N]")
        # plt.legend()
        # plt.savefig(os.path.join(output_dir, "PositionControl_RotorForce.png"))
        # plt.close()
        # print("[Drone Coordinates Controller] PositionControl_RotorForce.png saved")

        # Coordinate Position control
        for i in range(2):
            plt.figure()
            plt.clf()
            plt.plot(
                time, historyPositionReference[:, i], "--", label=f"Ref {labelsCart[i]}"
            )
            plt.plot(time, historyPosition[:, i], label=f"Actual {labelsCart[i]}")
            plt.title(f"{labelsCart[i]} Position Control")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.legend()
            plt.savefig(
                os.path.join(output_dir, f"{labelsCart[i]}_PositionControl.png")
            )
            print(
                f"[Drone Coordinates Controller] {labelsCart[i]}_PositionControl.png saved"
            )

        # Coordinate Velocity control
        for i in range(2):
            plt.figure()
            plt.clf()
            plt.plot(
                time, historyVelocityReference[:, i], "--", label=f"Ref {labelsCart[i]}"
            )
            plt.plot(time, historyVelocity[:, i], label=f"Actual {labelsCart[i]}")
            plt.title(f"{labelsCart[i]} Velocity Control")
            plt.xlabel("Time [s]")
            plt.ylabel("Velocity [m/s]")
            plt.legend()
            plt.savefig(
                os.path.join(output_dir, f"{labelsCart[i]}_VelocityControl.png")
            )
            print(
                f"[Drone Coordinates Controller] {labelsCart[i]}_VelocityControl.png saved"
            )

        # Coordinate Acceleration control
        for i in range(2):
            plt.figure()
            plt.clf()
            plt.plot(
                time,
                historyAccelerationReference[:, i],
                "--",
                label=f"Ref {labelsCart[i]}",
            )
            plt.plot(time, historyAcceleration[:, i], label=f"Actual {labelsCart[i]}")
            plt.title(f"{labelsCart[i]} Acceleration Control")
            plt.xlabel("Time [s]")
            plt.ylabel("Acceleration [m/s]")
            plt.legend()
            plt.savefig(
                os.path.join(output_dir, f"{labelsCart[i]}_AccelerationControl.png")
            )
            print(
                f"[Drone Coordinates Controller] {labelsCart[i]}_AccelerationControl.png saved"
            )

        # Reference angles graph
        for i in range(3):
            plt.figure()
            plt.clf()
            plt.plot(time, referenceAngles[:, i], "--", label=f"Ref {labelsAngles[i]}")
            plt.title(f"{labelsAngles[i]} Reference Angles")
            plt.xlabel("Time [s]")
            plt.ylabel("Angle [°]")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{labelsAngles[i]}_Ref.png"))
            print(f"[Drone Stability Controller] {labelsAngles[i]}_Ref.png saved")

    @staticmethod
    def normalizeVector(vector):
        """Normalizes a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm < 1e-5:
            return vector
        return vector / norm


class PIDController:
    def __init__(self, kp: np.array, ki: np.array, kd: np.array, outputLimit):
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._integral = np.zeros(3)
        self._previousError = np.zeros(3)
        self._outputLimit = outputLimit

    def reset(self):
        self._integral = np.zeros(3)
        self._previousError = np.zeros(3)

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

        return output
