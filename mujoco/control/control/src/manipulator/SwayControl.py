import time
import queue
import threading
from src.manipulator.robotControllers import RobotControl
from include.Messages import RobotSensorData, RobotControlAction
from include.PIDController import PIDController
import numpy as np


class SwayControl(threading.Thread):
    def __init__(
        self,
        shutdownEvent: threading.Event,
    ):
        super().__init__()
        self._shutdownEvent = shutdownEvent
        self._sensorData: RobotSensorData
        self._sensorLock = threading.Lock()
        self._controlLock = threading.Lock()
        self._robotControlAction: RobotControlAction = RobotControlAction(
            jointMotors={}  # Dictionary with manipulator's control values: key=motor name, value=action value
        )

        self._updateEvent = threading.Event()

        # Joint Position -> Joint Velocity PID Controller
        pKp = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64
        )  # Kp for Joints [q11 q12 q13 q14 q21 q22 q23 q24]
        pKi = np.zeros(8, dtype=np.float64)  # Ki for Joints [0 0 0 0 0 0 0 0]
        pKd = np.zeros(8, dtype=np.float64)  # Kd for Joints [0 0 0 0 0 0 0 0]
        self._MAX_VEL = 3  # Maximum joint velocity
        self._positionPID = PIDController(
            kp=pKp,
            ki=pKi,
            kd=pKd,
            outputLimit=self._MAX_VEL,
        )

        # Joint Velocity -> Joint Torque PID Controller - 5 times faster than position PID
        vKp = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        vKi = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        vKd = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        self._MAX_TORQUE = 1.0  # Maximum torque for the motors
        self._velocityPID = PIDController(
            kp=vKp,
            ki=vKi,
            kd=vKd,
            outputLimit=self._MAX_TORQUE,
        )

        # To check if the references have been reached.
        self._lastCheckTime = None
        self._reachedReferences = False
        self._toleranceThreshold = 0.01
        self._stabilityTime = 2

        self._timestep = 0.001
        self._currentTimestamp = 0.0
        self._previousTimestamp = 0.0
        self._currentPosition = np.zeros(3)
        self._currentAcceleration = np.zeros(3)
        self._currentAngularVelocity = np.zeros(3)
        self._currentVelocity = np.zeros(3)
        self._currentTargetPosition = np.zeros(3)

        self._targetQPosition = np.zeros(8)  # Target joint positions
        self._qPosition = np.zeros(8)  # Current joint positions
        self._qVelocity = np.zeros(8)  # Current joint velocities
        self._robotControlAction.jointMotors = {}

        self._debugging = False
        self._debug = False
        self._timeDebugging = False

    def run(self):
        print("[SwayControl] Initialized.")
        iteration = 0

        previousPosTimestamp = 0.0
        previousVelTimestamp = 0.0

        nextTime = time.perf_counter()
        tStart = time.time()
        while not self._shutdownEvent.is_set():
            t0 = time.time()
            self.updateCurrentState()

            # For first testing, change references every 10 seconds
            tnow = time.time()
            if tnow - tStart > 10:
                self._targetQPosition = np.array(
                    [1.57, 0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0]
                )

            if tnow - tStart > 20:
                self._targetQPosition = np.array(
                    [1.57, 0.75, 0.0, 0.0, -1.57, -0.75, 0.0, 0.0]
                )

            if tnow - tStart > 30:
                self._targetQPosition = np.array(
                    [1.57, 0.75, 0.0, 0.4, -1.57, -0.75, 0.0, 0.4]
                )

            iteration += 1
            # self._debug = self._debugging and iteration % 300 == 0

            if iteration == 1 or iteration % 5 == 0:
                posDt = self._currentTimestamp - previousPosTimestamp
                qdReference = self._positionPID.compute(
                    error=self._targetQPosition - self._qPosition, dt=posDt
                )
                previousPosTimestamp = self._currentTimestamp

            velDt = self._currentTimestamp - previousVelTimestamp
            torque = self._velocityPID.compute(
                error=qdReference - self._qVelocity, dt=velDt
            )
            previousVelTimestamp = self._currentTimestamp

            # Update control action
            jointMotors = {}
            for i in range(8):
                jointNames = [
                    "left_shoulder_roll",
                    "left_shoulder_pitch",
                    "left_elbow_roll",
                    "left_elbow_pitch",
                    "right_shoulder_roll",
                    "right_shoulder_pitch",
                    "right_elbow_roll",
                    "right_elbow_pitch",
                ]
                jointMotors[jointNames[i]] = torque[i]

            with self._controlLock:
                self._robotControlAction.jointMotors = jointMotors
            # print(f"[SwayControl] Sensor Data: {self._sensorData}")

            # Sincronización precisa
            nextTime += self._timestep
            sleepDuration = nextTime - time.perf_counter()
            if sleepDuration > 0:
                time.sleep(sleepDuration)
            else:
                # Si el procesamiento tarda más de 1 ms, ajusta el siguiente ciclo
                nextTime = time.perf_counter()

            if self._timeDebugging:
                dtReal = time.time() - t0
                dtSim = self._currentTimestamp - self._previousTimestamp
                print(f"[SwayControl] dtSim: {dtSim}, dtReal: {dtReal}")

        print("[SwayControl] Shutting down.")

    def stop(self):
        print("[SwayControl] Shutting down.")
        self._updateEvent.set()
        self._shutdownEvent.set()

    def updateCurrentState(self):
        self._updateEvent.wait()
        self._updateEvent.clear()
        if self._sensorData is not None:
            with self._sensorLock:
                sensorData = self._sensorData
            self._previousTimestamp = self._currentTimestamp
            self._currentTimestamp = sensorData.timestamp
            self._timestep = sensorData.timestep
            self._currentPosition = np.array(sensorData.manipulatorPosition)
            self._currentAcceleration = np.array(
                sensorData.sensorValues["robot_linacc"]
            )
            self._currentAngularVelocity = np.array(
                sensorData.sensorValues["robot_gyro"]
            )
            self._currentVelocity = np.array(sensorData.sensorValues["robot_vel"])
            self._currentTargetPosition = np.array(
                sensorData.dronePosition
                - [0.0, 0.0, 0.6245]  # Adjusted for cables and robot base offset
            )
            self._qPosition = sensorData.jointsPosition
            self._qVelocity = sensorData.jointsVelocity
            if self._currentTargetPosition[0] is None:
                self._currentTargetPosition[0] = self._currentPosition[0]
            if self._currentTargetPosition[1] is None:
                self._currentTargetPosition[1] = self._currentPosition[1]
            if self._currentTargetPosition[2] is None:
                self._currentTargetPosition[2] = self._currentPosition[2]
        else:
            pass

    # Accessed from MujocoSimulation thread.
    def setSensorData(self, sensorData: RobotSensorData):
        with self._sensorLock:
            self._sensorData = sensorData

    def getControlInputs(self):
        with self._controlLock:
            return self._robotControlAction

    def setUpdateEvent(self):
        self._updateEvent.set()
