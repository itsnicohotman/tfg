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
from qpsolvers import solve_qp


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

        self._refFilter = FirstOrderRef(alpha=0.1)  # smoothes targetQ
        self._velLPF = LowPass(alpha=0.15)  # smoothes qVelocity

        # Inputs
        self._sensorData: RobotSensorData = None
        self._posRobotCOM = np.zeros(3)
        # self._currentArmsHookPositions = None
        # self._currentDroneHookPositions = None
        # self._velRobotCOM = np.zeros(3)
        # self._accelRobotCOM = np.zeros(3)
        # self._JRobotCOM = np.zeros((3, 8))  # Jacobian of the CoM
        # self._JRobotCOMDot = np.zeros((3, 8))  # Jacobian derivative of the CoM

        # Outputs
        self._robotControlAction: RobotControlAction = RobotControlAction(
            jointMotors={}  # Dictionary with manipulator's control values: key=motor name, value=action value
        )

        self._timestep = 0.001
        self._currentTimestamp = 0.0
        self._previousTimestamp = 0.0
        self._currentPosition = np.zeros(3)
        self._currentAcceleration = np.zeros(3)
        self._currentAngularVelocity = np.zeros(3)
        self._currentVelocity = np.zeros(3)
        self._currentTargetRobotPosition = np.zeros(3)
        self._prevDt = 0.001

        self._targetQPosition = np.zeros(8)  # Target joint positions
        self._qPosition = np.zeros(8)  # Current joint positions
        self._qVelocity = np.zeros(8)  # Current joint
        self._qVelocityRef = np.zeros(8)  # Reference joint velocities
        self._qPositionRef = np.zeros(8)  # Reference joint positions
        self._robotControlAction.jointMotors = {}
        self._robotComError = np.zeros(3)
        self._robotComPos = np.zeros(3)
        self._manipPos = np.zeros(3)

        # Store for analysis
        self._historyTime = []
        self._historyControlAction = []
        self._historyQPosition = []
        self._historyQPositionReference = []
        self._historyQVelocity = []
        self._historyQVelocityReference = []
        self._historyRobotComPosError = []
        self._historyPosManipError = []

        self._debugging = False
        self._timeDebugging = False

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
        nullMotors = True

        nextTime = time.perf_counter()
        tStart = time.time()
        while not self._shutdownEvent.is_set():
            t0 = time.time()
            timeDebug = self._timeDebugging and iteration % 200 == 0

            self.updateCurrentState()
            iteration += 1

            tSinceStart = time.time() - tStart

            ###############################
            # Set Robot's joints references
            ###############################

            if self._currentTimestamp < 5:
                # 1) q4 from 0 to 90 degrees in both arms.
                self._targetQPosition = np.array(
                    [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0, 1.57]
                )
                # print(f"[SwayControl] Target joint position: {self._targetQPosition}")
            elif self._currentTimestamp < 6:
                nullMotors = True
                # print(f"[SwayControl] Stabilising. No reference set.")
            elif self._currentTimestamp < 16:
                # 2) q1 from 0 to 90 degrees in both arms.
                self._targetQPosition = np.array(
                    [1.57, 0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0]
                )
                nullMotors = False
                # print(f"[SwayControl] Target joint position: {self._targetQPosition}")
            elif self._currentTimestamp < 17:
                nullMotors = True
                # print(f"[SwayControl] Stabilising. No reference set.")
            elif self._currentTimestamp < 27:
                # 3) q4 from 0 to 90 degrees in left arm only.
                self._targetQPosition = np.array(
                    [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0, 0.0]
                )
                nullMotors = False
                # print(f"[SwayControl] Target joint position: {self._targetQPosition}")
            elif self._currentTimestamp < 28:
                nullMotors = True
                # print(f"[SwayControl] Stabilising. No reference set.")
            elif self._currentTimestamp < 38:
                # 4) q1 from 0 to 90 degrees in left arm only.
                self._targetQPosition = np.array(
                    [1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                )
                nullMotors = False
                # print(f"[SwayControl] Target joint position: {self._targetQPosition}")
            elif self._currentTimestamp < 39:
                nullMotors = True
                # print(f"[SwayControl] Stabilising. No reference set.")
            elif self._currentTimestamp < 49:
                # 5) q2 from 0 to 90 degrees in left arm. q2 from 0 to -90 degrees in right arm.
                self._targetQPosition = np.array(
                    [0.0, 1.57, 0.0, 0.0, 0.0, -1.57, 0.0, 0.0]
                )
                # print(f"[SwayControl] Target joint position: {self._targetQPosition}")
            elif self._currentTimestamp < 50:
                nullMotors = True
                # print(f"[SwayControl] Stabilising. No reference set.")
            elif self._currentTimestamp < 60:
                # 6) q2 from 0 to 90 degrees in left arm only.
                self._targetQPosition = np.array(
                    [0.0, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                )
                # print(f"[SwayControl] Target joint position: {self._targetQPosition}")
            elif self._currentTimestamp < 61:
                nullMotors = True
                # print(f"[SwayControl] Stabilising. No reference set.")
            elif self._currentTimestamp < 71:
                self._targetQPosition = np.array(
                    [1.57, 0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0]
                )
                # print(f"[SwayControl] Target joint position: {self._targetQPosition}")
            elif self._currentTimestamp < 72:
                nullMotors = True
                # print(f"[SwayControl] Stabilising. No reference set.")
            elif self._currentTimestamp < 82:
                time.sleep(10)
                self._shutdownEvent.set()

            t1 = time.time()

            self._robotComError = self._currentTargetRobotPosition - self._robotComPos
            self._robotBaseError = self._currentTargetRobotPosition - self._manipPos

            if iteration == 1 or iteration % 5 == 0:
                targetQPosition = self._refFilter.step(self._targetQPosition)
                posDt = self._currentTimestamp - previousPosTimestamp
                if posDt <= 0:
                    posDt = PERIOD_POS_CONTR
                qdReference = self._positionPID.compute(
                    error=targetQPosition - self._qPosition, dt=posDt
                )
                previousPosTimestamp = self._currentTimestamp

            qVelocityFiltered = self._velLPF.filt(self._qVelocity)
            velDt = self._currentTimestamp - previousVelTimestamp
            if velDt <= 0:
                velDt = PERIOD_VEL_CONTR
            torque = self._velocityPID.compute(
                error=qdReference - qVelocityFiltered, dt=velDt
            )
            previousVelTimestamp = self._currentTimestamp

            t2 = time.time()

            # Update control action
            jointMotors = {}
            if nullMotors:
                torque = np.zeros(8)
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
            self._historyRobotComPosError.append(self._robotComError.copy())
            self._historyPosManipError.append(self._robotBaseError.copy())

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
                # If the process takes longer than 1 ms, adjust next cycle.
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
                self._currentTimestamp = np.copy(sensorData.timestamp)
                self._timestep = np.copy(sensorData.timestep)
                self._currentPosition = np.array(
                    sensorData.manipulatorPosition, copy=True
                )
                self._currentAcceleration = np.array(
                    sensorData.sensorValues["robot_linacc"], copy=True
                )
                self._currentAngularVelocity = np.array(
                    sensorData.sensorValues["robot_gyro"], copy=True
                )
                self._currentVelocity = np.array(
                    sensorData.sensorValues["robot_vel"], copy=True
                )
                self._currentTargetRobotPosition = np.array(
                    sensorData.dronePosition - [0.0, 0.0, 0.6245],
                    copy=True,  # Adjusted for cables and robot base offset
                )
                self._qPosition = np.copy(sensorData.jointsPosition)
                self._qVelocityPrev = self._qVelocity.copy()
                self._qVelocity = np.copy(sensorData.jointsVelocity)
            self._posRobotCOMError = (
                self._currentTargetRobotPosition - self._posRobotCOM
            )

            if self._currentTargetRobotPosition[0] is None:
                self._currentTargetRobotPosition[0] = self._currentPosition[0]
            if self._currentTargetRobotPosition[1] is None:
                self._currentTargetRobotPosition[1] = self._currentPosition[1]
            if self._currentTargetRobotPosition[2] is None:
                self._currentTargetRobotPosition[2] = self._currentPosition[2]
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
        # Save graphs in separate folders for left and right arms
        dir_left = "/home/nicolas/tfg/Memoria/GraficasRobot/BrazoIzquierdo"
        dir_right = "/home/nicolas/tfg/Memoria/GraficasRobot/BrazoDerecho"
        dir_sway = "/home/nicolas/tfg/Memoria/GraficasRobot/SwayControl"
        os.makedirs(dir_left, exist_ok=True)
        os.makedirs(dir_right, exist_ok=True)
        os.makedirs(dir_sway, exist_ok=True)
        matplotlib.use("Agg")

        # Convert to arrays for easier indexing
        time = np.array(self._historyTime)
        q_pos = np.array(self._historyQPosition)
        q_pos_ref = np.array(self._historyQPositionReference)
        q_vel = np.array(self._historyQVelocity)
        q_vel_ref = np.array(self._historyQVelocityReference)
        control = np.array(self._historyControlAction)
        robotCOMPosError = np.array(self._historyRobotComPosError)
        robotPosError = np.array(self._historyPosManipError)

        # Names and groups of joints
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

        # Colors for joints
        color_map = plt.cm.get_cmap("tab10", len(jointNames))
        joint_colors = {name: color_map(i) for i, name in enumerate(jointNames)}

        # --- Individual graphs per joint ---
        for pass_num in (0, 1):
            for i, name in enumerate(jointNames):
                save_dir = dir_left if i in left_indices else dir_right

                if pass_num == 0:
                    # Dummies
                    if i == 0:
                        plt.figure(facecolor="w")
                        plt.plot(
                            time, q_pos[:, i], label="Actual", color=joint_colors[name]
                        )
                        plt.plot(
                            time,
                            q_pos_ref[:, i],
                            label="Reference",
                            linestyle="--",
                            color=joint_colors[name],
                        )
                        plt.xlabel("Time [s]")
                        plt.ylabel("Position [rad]")
                        plt.title(f"Joint position - {pretty_name(name)} (dummy)")
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(
                            os.path.join(save_dir, f"pos_{name}_dummy.png"),
                            facecolor="w",
                            edgecolor="w",
                        )
                        plt.close()

                        plt.figure(facecolor="w")
                        plt.plot(
                            time, q_vel[:, i], label="Actual", color=joint_colors[name]
                        )
                        plt.plot(
                            time,
                            q_vel_ref[:, i],
                            label="Reference",
                            linestyle="--",
                            color=joint_colors[name],
                        )
                        plt.xlabel("Time [s]")
                        plt.ylabel("Rate [rad/s]")
                        plt.title(f"Joint rate - {pretty_name(name)} (dummy)")
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(
                            os.path.join(save_dir, f"vel_{name}_dummy.png"),
                            facecolor="w",
                            edgecolor="w",
                        )
                        plt.close()
                        break
                    else:
                        continue

                else:
                    # Actual graphs
                    # Joint position
                    plt.figure(facecolor="w")
                    plt.plot(
                        time, q_pos[:, i], label="Actual", color=joint_colors[name]
                    )
                    plt.plot(
                        time,
                        q_pos_ref[:, i],
                        label="Reference",
                        linestyle="--",
                        color=joint_colors[name],
                    )
                    plt.xlabel("Time [s]")
                    plt.ylabel("Position [rad]")
                    plt.title(f"Joint position - {pretty_name(name)}")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(
                        os.path.join(save_dir, f"pos_{name}.png"),
                        facecolor="w",
                        edgecolor="w",
                    )
                    plt.close()

                    # Joint rate
                    plt.figure(facecolor="w")
                    plt.plot(
                        time, q_vel[:, i], label="Actual", color=joint_colors[name]
                    )
                    plt.plot(
                        time,
                        q_vel_ref[:, i],
                        label="Reference",
                        linestyle="--",
                        color=joint_colors[name],
                    )
                    plt.xlabel("Time [s]")
                    plt.ylabel("Rate [rad/s]")
                    plt.title(f"Joint rate - {pretty_name(name)}")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(
                        os.path.join(save_dir, f"vel_{name}.png"),
                        facecolor="w",
                        edgecolor="w",
                    )
                    plt.close()

        # --- Graphs per arm ---
        # Left arm
        plt.figure()
        for i in left_indices:
            plt.plot(
                time,
                q_pos[:, i],
                label=f"{pretty_name(jointNames[i])} actual",
                color=joint_colors[jointNames[i]],
            )
            plt.plot(
                time,
                q_pos_ref[:, i],
                linestyle="--",
                label=f"{pretty_name(jointNames[i])} ref",
                color=joint_colors[jointNames[i]],
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Position [rad]")
        plt.title("Left arm positions")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(dir_left, "pos_left_arm.png"), facecolor="w", edgecolor="w"
        )
        plt.close()

        # Right arm
        plt.figure()
        for i in right_indices:
            plt.plot(
                time,
                q_pos[:, i],
                label=f"{pretty_name(jointNames[i])} actual",
                color=joint_colors[jointNames[i]],
            )
            plt.plot(
                time,
                q_pos_ref[:, i],
                linestyle="--",
                label=f"{pretty_name(jointNames[i])} ref",
                color=joint_colors[jointNames[i]],
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Position [rad]")
        plt.title("Right arm positions")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(dir_right, "pos_right_arm.png"), facecolor="w", edgecolor="w"
        )
        plt.close()

        # --- Control inputs ---
        # Left arm
        plt.figure()
        for i in left_indices:
            plt.plot(
                time,
                control[:, i],
                label=pretty_name(jointNames[i]),
                color=joint_colors[jointNames[i]],
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")
        plt.title("Control outputs - Left arm")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(dir_left, "control_left_arm.png"), facecolor="w", edgecolor="w"
        )
        plt.close()

        # Right arm
        plt.figure()
        for i in right_indices:
            plt.plot(
                time,
                control[:, i],
                label=pretty_name(jointNames[i]),
                color=joint_colors[jointNames[i]],
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")
        plt.title("Control outputs - Right arm")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(dir_right, "control_right_arm.png"),
            facecolor="w",
            edgecolor="w",
        )
        plt.close()

        # --- Robot COM oscillations ---
        oscillationsNames = [
            "Oscillations in X axis",
            "Oscillations in Y axis",
            "Oscillations in Z axis",
        ]
        oscillationsLabels = ["x_oscillations", "y_oscillations", "z_oscillations"]
        for i in range(2):
            plt.figure()
            plt.plot(
                time,
                robotCOMPosError[:, i],
                label=pretty_name(oscillationsLabels[i]),
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.title(oscillationsNames[i])
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(dir_sway, f"COM_{oscillationsLabels[i]}.png"),
                facecolor="w",
                edgecolor="w",
            )
            plt.close()

        # --- Robot base oscillations ---
        for i in range(3):
            plt.figure()
            plt.plot(
                time,
                robotPosError[:, i],
                label=pretty_name(oscillationsLabels[i]),
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.title(oscillationsNames[i])
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(dir_sway, f"Base_{oscillationsLabels[i]}.png"),
                facecolor="w",
                edgecolor="w",
            )
            plt.close()

        print(f"✅ Graphs saved in:\n  {dir_left}\n  {dir_right}")


def pretty_name(joint):
    """Convierte snake_case a Title Case con espacios"""
    return joint.replace("_", " ").title()


class LowPass:
    def __init__(self, alpha=0.15):
        self.y = None
        self.alpha = alpha

    def filt(self, x):
        if self.y is None:
            self.y = x.copy()
        self.y = self.alpha * x + (1 - self.alpha) * self.y
        return self.y


class FirstOrderRef:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.y = None

    def step(self, r):
        if self.y is None:
            self.y = r.copy()
        self.y = self.y + self.alpha * (r - self.y)
        return self.y
