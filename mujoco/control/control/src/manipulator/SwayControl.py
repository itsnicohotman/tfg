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
            [1, 1, 12, 12, 1, 1, 12, 12], dtype=np.float64
        )  # Kp for Joints [q11 q12 q13 q14 q21 q22 q23 q24]
        pKi = np.array([0, 0, 5, 5, 0, 0, 5, 5])  # Ki for Joints [0 0 0 0 0 0 0 0]
        pKd = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # Kd for Joints [0 0 0 0 0 0 0 0]
        self._MAX_VEL = 1  # Maximum joint velocity
        self._positionPID = PIDController(
            kp=pKp,
            ki=pKi,
            kd=pKd,
            outputLimit=self._MAX_VEL,
        )

        # Joint Velocity -> Joint Torque PID Controller - 5 times faster than position PID
        vKp = np.array([1, 1, 10, 10, 1, 1, 10, 10], dtype=np.float64)
        vKi = np.array([1, 1, 5, 5, 1, 1, 5, 5], dtype=np.float64)
        vKd = np.array([0, 0, 0, 0.01, 0, 0, 0, 0.01], dtype=np.float64)
        self._MAX_TORQUE = 1.0  # Maximum torque for the motors
        self._velocityPID = PIDController(
            kp=vKp,
            ki=vKi,
            kd=vKd,
            outputLimit=self._MAX_TORQUE,
        )

        # Inputs
        self._sensorData: RobotSensorData = None
        self._currentArmsHookPositions = None
        self._currentDroneHookPositions = None

        # Outputs
        self._robotControlAction: RobotControlAction = RobotControlAction(
            jointMotors={}  # Dictionary with manipulator's control values: key=motor name, value=action value
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
        self._currentDroneAcceleration = np.zeros(3)
        self._currentCableVectorAvg = np.zeros(3)
        self._prevThetas = (0.0, 0.0)
        self._prevDt = 0.0

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
        self._timeDebugging = False

    def run(self):
        print("[SwayControl] Initialized.")
        real_start = time.time()
        sim_start = self._currentTimestamp
        iteration = 0

        previousPosTimestamp = 0.0
        previousVelTimestamp = 0.0
        previousQRefTimestamp = 0.0
        PERIOD_POS_CONTR = 0.005
        PERIOD_VEL_CONTR = 0.001
        PERIOD_QREF_CONTR = 0.02
        posDt = 0
        velDt = 0

        nextTime = time.perf_counter()
        tStart = time.time()
        while not self._shutdownEvent.is_set():
            t0 = time.time()
            timeDebug = self._timeDebugging and iteration % 200 == 0
            self.updateCurrentState()
            iteration += 1
            # For first testing, change references every 5 seconds
            tSinceStart = time.time() - tStart

            ###############################
            # Calculate Shoulder pitch and roll references
            ###############################
            if iteration == 1 or iteration % 50 == 0:
                self._prevDt = self._currentTimestamp - previousQRefTimestamp
                if self._prevDt <= 0:
                    self._prevDt = PERIOD_QREF_CONTR
                qsp_ref, qsr_ref = self.computeShoulderReferences()
                print(
                    f"[SwayControl] Shoulder references: qsp_ref={qsp_ref:.3f}, qsr_ref={qsr_ref:.3f}"
                )
                self._targetQPosition[1] = qsp_ref  # left_shoulder_pitch
                self._targetQPosition[0] = qsr_ref  # left_shoulder_roll
                self._targetQPosition[5] = qsp_ref  # right_shoulder_pitch
                self._targetQPosition[4] = qsr_ref  # right_shoulder_roll

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
                print(
                    f"[SwayControl] Elapsed Cycle: {elapsedCycle}, Sleep Duration; {sleepDuration}."
                )
                print(f"[SwayControl] posDt: {posDt}, velDt: {velDt}")
                # real_elapsed = time.time() - real_start
                # sim_elapsed = self._currentTimestamp - sim_start
                # print(
                #     f"[SwayControl] Sim Time: {sim_elapsed:.2f} s, Real Time: {real_elapsed:.2f} s"
                # )
                # elapsedCalcControl = t2 - t1
                # elapsedSetReference = t1 - tnow
                # elapsedUpdateState = t1 - t0
                # elapsedSetMotorsDict = t3 - t2
                # elapsedTotal = t3 - t0
                # print(
                #     f"[SwayControl]  Update: {elapsedUpdateState}, Set ref.: {elapsedSetReference:.6f}, \n Calculate Control: {elapsedCalcControl:.6f}, send Motors: {elapsedSetMotorsDict:.6f}, \n TOTAL: {elapsedTotal}"
                # )
                # print(
                #     f"[SwayControl] Time since start: {time.time() - tStart:.2f} seconds"
                # )
                # print(
                #     f"[SwayControl] current timestamp: {self._currentTimestamp}, previous timestamp: {self._previousTimestamp}"
                # )

            if sleepDuration > 0:
                time.sleep(sleepDuration)
            else:
                # Si el procesamiento tarda más de 1 ms, ajusta el siguiente ciclo
                nextTime = time.perf_counter()

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
            self._currentDroneAcceleration = sensorData.droneAcceleration
            self._currentArmsHookPositions = sensorData.armsHooks
            self._currentDroneHookPositions = sensorData.droneHooks
            if self._currentTargetPosition[0] is None:
                self._currentTargetPosition[0] = self._currentPosition[0]
            if self._currentTargetPosition[1] is None:
                self._currentTargetPosition[1] = self._currentPosition[1]
            if self._currentTargetPosition[2] is None:
                self._currentTargetPosition[2] = self._currentPosition[2]
        else:
            pass

    def computeShoulderReferences(self):
        """
        Calculates shoulder pitch (qsp) and roll (qsr) references from the drone's acceleration and cable vector.
        """

        Kp = 0.6
        Kd = 0.3

        g = 9.81
        ax, ay = self._currentDroneAcceleration[0], self._currentDroneAcceleration[1]

        # feedforward: compensar aceleración del dron
        pitchFF = -np.arctan2(ax, g)
        rollFF = -np.arctan2(ay, g)

        # ángulos de la cuerda respecto a z
        unitVectors, self._currentCableVectorAvg = self.computeCableVectors(
            armsHooks=self._currentArmsHookPositions,
            droneHooks=self._currentDroneHookPositions,
        )
        u = self._currentCableVectorAvg / (np.linalg.norm(self._currentCableVectorAvg))
        thetaX = np.arctan2(u[0], u[2])  # inclinación en x
        thetaY = np.arctan2(u[1], u[2])  # inclinación en y

        # derivadas (aprox)
        dthetaX = (thetaX - self._prevThetas[0]) / (self._prevDt)
        dthetaY = (thetaY - self._prevThetas[1]) / (self._prevDt)

        self._prevThetas = (thetaX, thetaY)

        # control feedback: PD sobre ángulo de la cuerda
        qsp_ref = pitchFF - Kp * thetaX - Kd * dthetaX
        qsr_ref = rollFF - Kp * thetaY - Kd * dthetaY

        return qsp_ref, qsr_ref

    def computeCableVectors(self, armsHooks, droneHooks):
        """
        Calcula vectores unitarios de las cuerdas en coordenadas globales.
        armsHooks: np.array shape (N,3)
        droneHooks: np.array shape (N,3)
        return:
        unit_vectors: np.array shape (N,3) con los vectores normalizados
        u_avg: np.array shape (3,) vector unitario promedio
        """
        diffs = armsHooks - droneHooks
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        unitVectors = diffs / norms

        # promedio y normalización
        uMean = np.mean(unitVectors, axis=0)
        uAvg = uMean / np.linalg.norm(uMean)

        # print("[SwayControl] Unit Vectors:", unitVectors)
        # print("[SwayControl] Average Unit Vector:", uAvg)

        return unitVectors, uAvg

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
        # Directorios de guardado
        dir_left = "/home/nicolas/tfg/Memoria/GraficasRobot/BrazoIzquierdo"
        dir_right = "/home/nicolas/tfg/Memoria/GraficasRobot/BrazoDerecho"
        os.makedirs(dir_left, exist_ok=True)
        os.makedirs(dir_right, exist_ok=True)
        matplotlib.use("Agg")

        # Convertir a arrays
        time = np.array(self._historyTime)
        q_pos = np.array(self._historyQPosition)
        q_pos_ref = np.array(self._historyQPositionReference)
        q_vel = np.array(self._historyQVelocity)
        q_vel_ref = np.array(self._historyQVelocityReference)
        control = np.array(self._historyControlAction)

        # Nombres y grupos
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
        left_indices = range(0, 4)
        right_indices = range(4, 8)

        # Colores consistentes por articulación
        color_map = plt.cm.get_cmap("tab10", len(jointNames))
        joint_colors = {name: color_map(i) for i, name in enumerate(jointNames)}

        # --- 1. Gráficas individuales por articulación ---
        for i, name in enumerate(jointNames):
            # Determinar carpeta de guardado
            save_dir = dir_left if i in left_indices else dir_right

            # Evitar gráficas vacías
            if time.size == 0 or q_pos.shape[0] == 0:
                print(f"⚠️ Sin datos para {name}, se omite.")
                continue

            # Posición
            plt.figure()
            plt.plot(time, q_pos[:, i], label="Actual", color=joint_colors[name])
            plt.plot(
                time,
                q_pos_ref[:, i],
                label="Referencia",
                linestyle="--",
                color=joint_colors[name],
            )
            plt.xlabel("Tiempo [s]")
            plt.ylabel("Posición [rad]")
            plt.title(f"Posición articular - {name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f"pos_{name}.png"))
            plt.close()

            # Velocidad
            plt.figure()
            plt.plot(time, q_vel[:, i], label="Actual", color=joint_colors[name])
            plt.plot(
                time,
                q_vel_ref[:, i],
                label="Referencia",
                linestyle="--",
                color=joint_colors[name],
            )
            plt.xlabel("Tiempo [s]")
            plt.ylabel("Velocidad [rad/s]")
            plt.title(f"Velocidad articular - {name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f"vel_{name}.png"))
            plt.close()

        # --- 2. Gráfica con todas las articulaciones por brazo ---
        # Brazo izquierdo
        plt.figure()
        for i in left_indices:
            plt.plot(
                time,
                q_pos[:, i],
                label=f"{jointNames[i]} actual",
                color=joint_colors[jointNames[i]],
            )
            plt.plot(
                time,
                q_pos_ref[:, i],
                linestyle="--",
                label=f"{jointNames[i]} ref",
                color=joint_colors[jointNames[i]],
            )
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Posición [rad]")
        plt.title("Posiciones brazo izquierdo")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(dir_left, "pos_brazo_izquierdo.png"))
        plt.close()

        # Brazo derecho
        plt.figure()
        for i in right_indices:
            plt.plot(
                time,
                q_pos[:, i],
                label=f"{jointNames[i]} actual",
                color=joint_colors[jointNames[i]],
            )
            plt.plot(
                time,
                q_pos_ref[:, i],
                linestyle="--",
                label=f"{jointNames[i]} ref",
                color=joint_colors[jointNames[i]],
            )
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Posición [rad]")
        plt.title("Posiciones brazo derecho")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(dir_right, "pos_brazo_derecho.png"))
        plt.close()

        # --- 3. Acciones de control ---
        # Brazo izquierdo
        plt.figure()
        for i in left_indices:
            plt.plot(
                time,
                control[:, i],
                label=jointNames[i],
                color=joint_colors[jointNames[i]],
            )
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Torque [Nm]")
        plt.title("Acciones de control - brazo izquierdo")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(dir_left, "control_brazo_izquierdo.png"))
        plt.close()

        # Brazo derecho
        plt.figure()
        for i in right_indices:
            plt.plot(
                time,
                control[:, i],
                label=jointNames[i],
                color=joint_colors[jointNames[i]],
            )
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Torque [Nm]")
        plt.title("Acciones de control - brazo derecho")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(dir_right, "control_brazo_derecho.png"))
        plt.close()

        print(f"✅ Gráficas guardadas en:\n  {dir_left}\n  {dir_right}")
