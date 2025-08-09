import time
import queue
import threading
from src.manipulator.robotControllers import RobotControl
from include.Messages import RobotSensorData, RobotControlAction
from include.PIDController import PIDController
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os


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
            [1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64
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
        vKp = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
        vKi = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
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

        # Store for analysis
        self._historyReferencePosition = []
        self._historyCurrentPosition = []
        self._historyReferenceVelocity = []
        self._historyCurrentVelocity = []
        self._historyReferenceAcceleration = []
        self._historyCurrentAcceleration = []
        self._historyTime = []
        self._historyControlAction = []
        self._historyQPosition = []
        self._historyQPositionReference = []
        self._historyQVelocity = []
        self._historyQVelocityReference = []

        self._debugging = False
        self._debug = False
        self._timeDebugging = True

    def run(self):
        print("[SwayControl] Initialized.")
        real_start = time.time()
        sim_start = self._currentTimestamp
        iteration = 0

        previousPosTimestamp = 0.0
        previousVelTimestamp = 0.0
        PERIOD_POS_CONTR = 0.005
        PERIOD_VEL_CONTR = 0.001
        posDt = 0
        velDt = 0

        nextTime = time.perf_counter()
        tStart = time.time()
        while not self._shutdownEvent.is_set():
            t0 = time.time()
            timeDebug = self._timeDebugging and iteration % 200 == 0
            self.updateCurrentState()

            # if self._timeDebugging:
            #     print(
            #         f"[SwayControl] Time since start: {time.time() - tStart:.2f} seconds"
            #     )
            #     print(
            #         f"[SwayControl] current timestamp: {self._currentTimestamp}, previous timestamp: {self._previousTimestamp}"
            # )
            # print(f"[SwayControl] posDt: {posDt}, velDt: {velDt}")

            # For first testing, change references every 5 seconds
            tnow = time.time()
            if tnow > 5:
                self._targetQPosition = np.array(
                    [1.57, 0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0]
                )

            if tnow > 10:
                self._targetQPosition = np.array(
                    [1.57, 0.75, 0.0, 0.0, -1.57, -0.75, 0.0, 0.0]
                )

            if tnow > 15:
                self._targetQPosition = np.array(
                    [1.57, 0.75, 0.0, 0.4, -1.57, -0.75, 0.0, 0.4]
                )

            iteration += 1
            # self._debug = self._debugging and iteration % 300 == 0

            t1 = time.time()

            if iteration == 1 or iteration % 5 == 0:
                posDt = self._currentTimestamp - previousPosTimestamp
                if posDt <= 0:
                    posDt = PERIOD_POS_CONTR
                qdReference = self._positionPID.compute(
                    error=self._targetQPosition - self._qPosition, dt=posDt
                )
                previousPosTimestamp = self._currentTimestamp

            velDt = self._currentTimestamp - previousVelTimestamp
            if velDt <= 0:
                velDt = PERIOD_VEL_CONTR
            torque = self._velocityPID.compute(
                error=qdReference - self._qVelocity, dt=velDt
            )
            previousVelTimestamp = self._currentTimestamp

            t2 = time.time()

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
            t3 = time.time()
            # print(f"[SwayControl] Sensor Data: {self._sensorData}")

            # Store for analysis
            self._historyTime.append(self._currentTimestamp)
            self._historyQPosition.append(self._qPosition.copy())
            # print(f"[SwayControl] Current QPosition: {self._qPosition.copy()}")
            self._historyQPositionReference.append(self._targetQPosition.copy())
            self._historyQVelocity.append(self._qVelocity.copy())
            self._historyQVelocityReference.append(qdReference.copy())
            self._historyControlAction.append(torque.copy())

            elapsedCycle = time.time() - t0

            # Sincronización precisa
            nextTime += self._timestep
            sleepDuration = nextTime - time.perf_counter()

            if timeDebug:
                real_elapsed = time.time() - real_start
                sim_elapsed = self._currentTimestamp - sim_start

                print(
                    f"[SwayControl] Elapsed Cycle: {elapsedCycle}, Sleep Duration; {sleepDuration}."
                )
                print(
                    f"[SwayControl] Sim Time: {sim_elapsed:.2f} s, Real Time: {real_elapsed:.2f} s"
                )
                elapsedCalcControl = t2 - t1
                elapsedSetReference = t1 - tnow
                elapsedUpdateState = t1 - t0
                elapsedSetMotorsDict = t3 - t2
                elapsedTotal = t3 - t0
                print(
                    f"[SwayControl]  Update: {elapsedUpdateState}, Set ref.: {elapsedSetReference:.6f}, \n Calculate Control: {elapsedCalcControl:.6f}, send Motors: {elapsedSetMotorsDict:.6f}, \n TOTAL: {elapsedTotal}"
                )

            if sleepDuration > 0:
                time.sleep(sleepDuration)
            else:
                # Si el procesamiento tarda más de 1 ms, ajusta el siguiente ciclo
                nextTime = time.perf_counter()

            # if self._timeDebugging:
            #     dtReal = time.time() - t0
            #     dtSim = self._currentTimestamp - self._previousTimestamp
            #     print(f"[SwayControl] dtSim: {dtSim}, dtReal: {dtReal}")

        self.graphResults()
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

    def graphResults(self):
        # Create output directory if it doesn't exist
        output_dir = "/home/nicolas/tfg/Memoria/GraficasRobot"
        os.makedirs(output_dir, exist_ok=True)
        matplotlib.use("Agg")

        time = np.array(self._historyTime)
        qPos = np.array(self._historyQPosition)  # shape: (steps, joints)
        qPosRef = np.array(self._historyQPositionReference)  # shape: (steps, joints)
        qVel = np.array(self._historyQVelocity)
        qVelRef = np.array(self._historyQVelocityReference)
        control = np.array(self._historyControlAction)

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

        leftIndices = range(0, 4)  # Left Arm joints
        rightIndices = range(4, 8)  # Right Arm joints

        # --- 1. Gráficas individuales por articulación ---
        for i, name in enumerate(jointNames):
            # Posición
            plt.figure()
            plt.plot(time, qPos[:, i], label="Actual")
            plt.plot(time, qPosRef[:, i], label="Referencia", linestyle="--")
            plt.xlabel("Tiempo [s]")
            plt.ylabel("Posición [rad]")
            plt.title(f"Posición articular - {name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"pos_{name}.png"))
            plt.close()

            # Velocidad
            plt.figure()
            plt.plot(time, qVel[:, i], label="Actual")
            plt.plot(time, qVelRef[:, i], label="Referencia", linestyle="--")
            plt.xlabel("Tiempo [s]")
            plt.ylabel("Velocidad [rad/s]")
            plt.title(f"Velocidad articular - {name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"vel_{name}.png"))
            plt.close()

        # --- 2. Gráfica con todas las articulaciones por brazo (posiciones y referencias) ---
        # Brazo izquierdo
        plt.figure()
        for i in leftIndices:
            plt.plot(time, qPos[:, i], label=f"{jointNames[i]} actual")
            plt.plot(time, qPosRef[:, i], linestyle="--", label=f"{jointNames[i]} ref")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Posición [rad]")
        plt.title("Posiciones brazo izquierdo")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "pos_brazo_izquierdo.png"))
        plt.close()

        # Brazo derecho
        plt.figure()
        for i in rightIndices:
            plt.plot(time, qPos[:, i], label=f"{jointNames[i]} actual")
            plt.plot(time, qPosRef[:, i], linestyle="--", label=f"{jointNames[i]} ref")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Posición [rad]")
        plt.title("Posiciones brazo derecho")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "pos_brazo_derecho.png"))
        plt.close()

        # --- 3. Acciones de control por brazo ---
        # Brazo izquierdo
        plt.figure()
        for i in leftIndices:
            plt.plot(time, control[:, i], label=jointNames[i])
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Torque [Nm]")
        plt.title("Acciones de control - brazo izquierdo")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "control_brazo_izquierdo.png"))
        plt.close()

        # Brazo derecho
        plt.figure()
        for i in rightIndices:
            plt.plot(time, control[:, i], label=jointNames[i])
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Torque [Nm]")
        plt.title("Acciones de control - brazo derecho")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "control_brazo_derecho.png"))
        plt.close()

        print(f"✅ Gráficas guardadas en: {os.path.abspath(output_dir)}")
