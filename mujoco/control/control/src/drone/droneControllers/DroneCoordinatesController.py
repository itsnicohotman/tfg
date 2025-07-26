import threading
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import matplotlib

from src.drone.droneControllers.TrajectoryPlanner import TrajectoryPlanner
from src.drone.droneControllers.DroneStabilityController import DroneStabilityController


class CoordinatesController(threading.Thread):
    """Controller for drone's position: [x, y].
    Follows PX4's Position Control architecture.
    Position, Velocity and Acceleration controller at 50Hz.
    """

    def __init__(
        self,
        shutdownEvent: threading.Event,
        coordinatesUpdateEvent: threading.Event,
        coordinatesLock: threading.Lock,
        coordinatesData: dict,
        coordinatesAnglesLock: threading.Lock,
    ):
        super().__init__()
        self._shutdownEvent = shutdownEvent
        self._coordinatesUpdateEvent = coordinatesUpdateEvent
        self._coordinatesResultsLock = coordinatesAnglesLock
        self._coordinatesLock = coordinatesLock
        self._coordinatesData = coordinatesData

        # Position -> Velocity PID Controller
        pKp = np.array([3, 3, 0.0], dtype=np.float64)  # Kp for x, y [3 3 0]
        pKi = np.array([2.3, 2.3, 0.0], dtype=np.float64)  # Ki for x, y [2.3 2.3 0]
        pKd = np.array([0.08, 0.08, 0.0], dtype=np.float64)  # Kd for x, y [0.05 0.05 0]
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

        # Check if the height reference has been reached
        self._stabilityTime = (
            0.5  # Time in seconds to consider the height reference as reached
        )
        self._errorTolerance = 0.05  # Tolerance for height and velocity errors
        self._previousHeightReference = 0.0
        self._positionReferenceChanged = False
        self._reachedReference = False
        self._inToleranceSince = None

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

        self._debugging = False
        self._timeDebugging = False

    def run(self):
        print("[DroneCoordinatesController] Initialized.")
        iteration = 0
        previousTime = 0

        PERIOD = 0.02
        nextTime = time.perf_counter()

        while not self._shutdownEvent.is_set():
            debug = self._debugging and iteration % 300 == 0
            timeDebug = self._timeDebugging and iteration % 300 == 0

            iteration += 1
            self.updateState()

            currentTime = self._currentTimestamp
            dt = currentTime - previousTime if previousTime > 0 else self._timestep
            if dt <= 0:
                dt = self._timestep
            previousTime = currentTime

            targetVelocity = np.zeros(3)
            # Coordinates controller: P controller whose output is the target velocity to the new position.
            # Will execute with a frequency of f = 50 Hz (given that timestep = 0.001). dt = 0.02
            errorPosition = self._currentTargetPosition - self._currentPosition
            mask = np.abs(errorPosition) > 0.1
            targetVelocity[mask] = self._currentTargetVelocity[mask] * np.sign(
                errorPosition[mask]
            )

            desiredVelocity = (
                self._positionPID.compute(error=errorPosition, dt=dt) + targetVelocity
            )

            # Velocity controller: PID controller whose output is the target acceleration.
            # Will execute with a frequency of f = 50 Hz (given that timestep = 0.001). dt = 0.02
            errorVelocity = desiredVelocity - self._currentVelocity
            desiredAcceleration = np.array(
                self._velocityPID.compute(error=errorVelocity, dt=dt)
            )

            # Acceleration controller: PID controller whose output are the reference Euler angles for the stability controller
            # Will execute with a frequency of f = 50 Hz (given that timestep = 0.001). dt = 0.02

            referenceEulerAngles = self.computeAttitude(
                desiredAcceleration=desiredAcceleration
            )

            reachedReference = self.checkReachedReferences(
                positionError=errorPosition[:2],
                velocityError=errorVelocity[:2],
            )

            with self._coordinatesResultsLock:
                self._refAngles = np.array(referenceEulerAngles)
                self._reachedReference = reachedReference

            if debug:
                print(
                    f"[DroneCoordinatesController] Current Position: {self._currentPosition}, Target Position: {self._currentTargetPosition}"
                )
                print(
                    f"[DroneCoordinatesController] Desired Velocity: {desiredVelocity}, Desired Acceleration: {desiredAcceleration}"
                )
                print(
                    f"[DroneCoordinatesController] Ref. Angles: {referenceEulerAngles}"
                )

                if reachedReference:
                    print(
                        "[DroneCoordinatesController] Position reference reached and stable."
                    )
            if timeDebug:
                print(f"[DroneCoordinatesController] dt: {dt}")

            # Store for analysis
            self._historyTime.append(self._currentTimestamp)
            self._historyCurrentPosition.append(self._currentPosition.tolist())
            self._historyReferencePosition.append(self._currentTargetPosition.tolist())
            self._historyReferenceVelocity.append(desiredVelocity.tolist())
            self._historyCurrentVelocity.append(self._currentVelocity.tolist())
            self._historyReferenceAcceleration.append(desiredAcceleration.tolist())
            self._historyCurrentAcceleration.append(self._currentAcceleration.tolist())
            self._historyRefAngles.append(self._refAngles.tolist())

            nextTime += PERIOD
            sleepDuration = nextTime - time.perf_counter()
            if sleepDuration > 0.0001:
                time.sleep(sleepDuration - 0.00005)  # Sleeps most of the time

            # Busy-wait the last ~50 µs
            while time.perf_counter() < nextTime:
                pass

        self.graphErrors()

    def stop(self):
        self._shutdownEvent.set()
        self._coordinatesUpdateEvent.set()
        print("[DroneCoordinatesController] Shutting Down.")

    def computeAttitude(self, desiredAcceleration):
        """Calculate the reference angles (roll, pitch, yaw) based on the desired acceleration"""
        MAX_ROLL = 0.52  # 30 degrees in radians
        MAX_PITCH = 0.52  # 30 degrees in radians
        gravity = 9.81
        desiredAcceleration[2] = gravity

        yaw = 0  # Assume no yaw change for simplicity

        pitch = np.arcsin(desiredAcceleration[0] / np.linalg.norm(desiredAcceleration))
        pitch = np.clip(pitch, -MAX_PITCH, MAX_PITCH)
        roll = np.arcsin(-desiredAcceleration[1] / np.linalg.norm(desiredAcceleration))
        roll = np.clip(roll, -MAX_ROLL, MAX_ROLL)

        referenceAngles = np.array([roll, pitch, yaw], dtype=np.float64)
        mask = np.abs(referenceAngles) <= 1e-5
        referenceAngles[mask] = 0.0

        if self._reachedReference:
            referenceAngles = np.zeros(3)

        return referenceAngles

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
    def checkReachedReferences(self, positionError, velocityError):
        """Check if the target position has been reached."""
        reachedReference = False
        xPositionError = positionError[0]
        xVelocityError = velocityError[0]
        yPositionError = positionError[1]
        yVelocityError = velocityError[1]

        if self._positionReferenceChanged:
            self._positionReferenceChanged = False
            reachedReference = False
            self._inToleranceSince = None
            print(
                "[DroneCoordinatesController] Position reference changed, resetting check."
            )
        else:
            if (
                abs(xPositionError) < self._errorTolerance
                and abs(xVelocityError) < self._errorTolerance
                and abs(yPositionError) < self._errorTolerance
                and abs(yVelocityError) < self._errorTolerance
            ):
                if self._inToleranceSince is None:
                    self._inToleranceSince = time.time()
                elif (time.time() - self._inToleranceSince) >= self._stabilityTime:
                    reachedReference = True
                    self._positionPID.reset()
                    self._velocityPID.reset()
            else:
                self._inToleranceSince = None
        return reachedReference

    def setCoordinatesData(self, coordinatesData):
        with self._coordinatesLock:
            self._coordinatesData = coordinatesData

    def getAttitudeReference(self):
        with self._coordinatesResultsLock:
            referenceEulerAngles = self._refAngles
        return referenceEulerAngles

    def getReachedReference(self):
        with self._coordinatesResultsLock:
            return self._reachedReference

    def graphErrors(self):
        if not self._historyTime:
            return

        output_dir = "/home/nicolas/tfg/Memoria/GraficasPosicion"
        os.makedirs(output_dir, exist_ok=True)
        matplotlib.use("Agg")

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
        referenceAngles = np.array(self._historyRefAngles[:lengthTime])
        labelsCart = ["X", "Y"]
        labelsAngles = ["Roll", "Pitch", "Yaw"]

        # --- Position error - dummy ---
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

        # --- Position error ---
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

        # --- Velocity error ---
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

        # --- Acceleration error ---
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

        # --- Coordinate Position control ---
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

        # --- Coordinate Velocity control ---
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

        # --- Coordinate Acceleration control ---
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

        # --- Reference angles graph ---
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

        return np.array(output)
