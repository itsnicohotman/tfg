import threading
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import os

from src.drone.droneControllers.TrajectoryPlanner import TrajectoryPlanner


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
        self._historyThrust = []

    def run(self):
        print("[DroneCoordinatesController] Initialized.")
        iteration = 0
        previousTime = 0

        while not self._shutdownEvent.is_set():
            iteration += 1
            self.updateMovementReference()

            currentTime = self._currentTimestamp
            dt = currentTime - previousTime if previousTime > 0 else self._timestep
            previousTime = currentTime

            if iteration == 1 or iteration % 20 == 0:
                # Coordinates controller: P controller whose output is the target velocity to the new position.
                # Will execute with a frequency of f = 50 Hz (given that timestep = 0.001)
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
            self._historyThrust.append(self._baseThrust)

        self.graphErrors()

    def stop(self):
        self._shutdownEvent.set()
        print("[DroneCoordinatesController] Shutting Down.")
        self.graphErrors()

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
            self._currentTimestamp = self._coordinatesData["timestamp"]
            self._timestep = self._coordinatesData["timestep"]
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
        if not self._historyTime:
            return

        self._historyCurrentPosition.append(self._currentPosition.tolist())
        self._historyReferencePosition.append(self._currentTargetPosition.tolist())
        self._historyReferenceVelocity.append(self._currentTargetVelocity.tolist())
        self._historyReferenceAcceleration.append(
            self._currentTargetAcceleration.tolist()
        )
        self._historyCurrentAcceleration.append(self._currentAcceleration.tolist())
        self._historyRefAngles.append(self._refAngles.tolist())

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
        """ lengthPosGraph = min(len(historyPositionReference), len(historyPosition))
        lengthVelGraph = min(len(historyVelocityReference), len(historyVelocity))
        lengthAccGraph = min(
            len(historyAccelerationReference), len(historyAcceleration)
        )
        historyPositionError = (
            historyPositionReference[:lengthPosGraph] - historyPosition[:lengthPosGraph]
        )
        historyVelocityError = (
            historyVelocityReference[:lengthVelGraph] - historyVelocity[:lengthVelGraph]
        )
        historyAccelerationError = (
            historyAccelerationReference[:lengthAccGraph]
            - historyAcceleration[:lengthAccGraph]
        ) """
        historyRotorForce = np.array(self._historyThrust[:lengthTime])
        labels = ["X", "Y", "Z"]

        print(f"[COORDINATES CONTROLLER] Shapes: \n")
        print(f"[COORDINATES CONTROLLER] time:{time.shape} \n")
        print(f"time: {time}")
        print(f"[COORDINATES CONTROLLER] Position:{historyPosition.shape} \n")
        print(
            f"[COORDINATES CONTROLLER] Position Ref:{historyPositionReference.shape} \n"
        )
        print(f"[COORDINATES CONTROLLER] Velocity:{historyVelocity.shape} \n")
        print(
            f"[COORDINATES CONTROLLER] Velocity Ref:{historyVelocityReference.shape} \n"
        )
        print(f"[COORDINATES CONTROLLER] Acceleration:{historyAcceleration.shape} \n")
        print(
            f"[COORDINATES CONTROLLER] Acceleration Ref:{historyAccelerationReference.shape} \n"
        )
        print(
            f"[COORDINATES CONTROLLER] Position Error:{historyPositionError.shape} \n"
        )
        print(
            f"[COORDINATES CONTROLLER] Velocity Error:{historyVelocityError.shape} \n"
        )
        print(
            f"[COORDINATES CONTROLLER] Acceleration Error:{historyAccelerationError.shape} \n"
        )

        # Position error graph - dummy
        for i in range(3):
            plt.plot(time, historyPositionError[:, i], label=labels[i])
        plt.title("Position Control Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [m]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "Position_Errors_dummy.png"))
        print("[Drone Coordinates Controller] Position_Errors_dummy.png saved")

        # Position error graph
        for i in range(3):
            plt.plot(time, historyPositionError[:, i], label=labels[i])
        plt.title("Position Control Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [m]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "Position_Errors.png"))
        print("[Drone Coordinates Controller] Position_Errors.png saved")

        # Velocity error graph
        for i in range(3):
            plt.plot(time, historyVelocityError[:, i], label=labels[i])
        plt.title("Velocity Control Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [m]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "Velocity_Errors.png"))
        print("[Drone Coordinates Controller] Velocity_Errors.png saved")

        # Acceleration error graph
        for i in range(3):
            plt.plot(time, historyAccelerationError[:, i], label=labels[i])
        plt.title("Acceleration Control Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [m]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "Acceleration_Errors.png"))
        print("[Drone Coordinates Controller] Acceleration_Errors.png saved")

        # Thrust graph
        plt.figure()
        plt.clf()
        plt.plot(time, historyRotorForce, label="Rotor Force")
        plt.title("Position Control - base thrust per rotor")
        plt.xlabel("Time [s]")
        plt.ylabel("Rotor Force [N]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "PositionControl_RotorForce.png"))
        plt.close()
        print("[Drone Coordinates Controller] PositionControl_RotorForce.png saved")

        # Coordinate Position control
        for i in range(3):
            plt.figure()
            plt.plot(
                time, historyPositionReference[:, i], "--", label=f"Ref {labels[i]}"
            )
            plt.plot(time, historyPosition[:, i], label=f"Actual {labels[i]}")
            plt.title(f"{labels[i]} Position Control")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{labels[i]}_PositionControl.png"))
            print(
                f"[Drone Coordinates Controller] {labels[i]}_PositionControl.png saved"
            )

        # Coordinate Velocity control
        for i in range(3):
            plt.figure()
            plt.plot(
                time, historyVelocityReference[:, i], "--", label=f"Ref {labels[i]}"
            )
            plt.plot(time, historyVelocity[:, i], label=f"Actual {labels[i]}")
            plt.title(f"{labels[i]} Velocity Control")
            plt.xlabel("Time [s]")
            plt.ylabel("Velocity [m/s]")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{labels[i]}_VelocityControl.png"))
            print(
                f"[Drone Coordinates Controller] {labels[i]}_VelocityControl.png saved"
            )

        # Coordinate Acceleration control
        for i in range(3):
            plt.figure()
            plt.plot(
                time, historyAccelerationReference[:, i], "--", label=f"Ref {labels[i]}"
            )
            plt.plot(time, historyAcceleration[:, i], label=f"Actual {labels[i]}")
            plt.title(f"{labels[i]} Acceleration Control")
            plt.xlabel("Time [s]")
            plt.ylabel("Acceleration [m/s]")
            plt.legend()
            plt.savefig(
                os.path.join(output_dir, f"{labels[i]}_AccelerationControl.png")
            )
            print(
                f"[Drone Coordinates Controller] {labels[i]}_AccelerationControl.png saved"
            )


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
