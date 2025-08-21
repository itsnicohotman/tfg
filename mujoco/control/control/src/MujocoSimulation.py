"""This class creates a separate thread where a passive non-blocking simulaton viewer is executed.
This thread collects the model's sensors' values and sends them to the control thread.
It waits fot the control thread's answer with the correct control inputs that are later implemented in the simulation.
"""

import mujoco
import mujoco.viewer
import threading
import numpy as np
from include.Messages import (
    SensorData,
    ControlAction,
    RobotControlAction,
    RobotSensorData,
)
import time
from src.drone.DroneControl import DroneControl
from src.manipulator.SwayControl import SwayControl

from cython_modules.cy_sim_utils import extract_sensor_values, apply_control_inputs
from cython_modules.fast_mujoco_utils import extract_data


class MujocoSimulation(threading.Thread):
    def __init__(
        self,
        shutdownEvent: threading.Event,
        model,
        droneController: DroneControl,
        swayController: SwayControl,
        swayControllerEnabled: bool,
    ):
        super().__init__()  # Start thread
        self._droneController = droneController
        self._swayController = swayController
        self._swayControllerEnabled = swayControllerEnabled
        match model:
            case "drone":
                MODEL_PATH = "/home/nicolas/tfg/mujoco/model/drone/Model_GRVC_Drone.xml"
                print("Simulating drone.")
                self._droneController.setHeightReference(0.45)
            case "0 cable":
                MODEL_PATH = "/home/nicolas/tfg/mujoco/model/drone/Model_GRVC_Aerial_Manipulator_NoCable.xml"
                print("Simulating aerial manipulator hanging by 0 cables.")
                self._droneController.setHeightReference(2)
            case "1 cable":
                MODEL_PATH = "/home/nicolas/tfg/mujoco/model/drone/Model_GRVC_Aerial_Manipulator_1Cable.xml"
                print("Simulating aerial manipulator hanging by 1 cable.")
                self._droneController.setHeightReference(2)
            case "2 cable":
                MODEL_PATH = "/home/nicolas/tfg/mujoco/model/drone/Model_GRVC_Aerial_Manipulator_2Cable.xml"
                print("Simulating aerial manipulator hanging by 2 cables.")
                self._droneController.setHeightReference(2)
            case "3 cable":
                MODEL_PATH = "/home/nicolas/tfg/mujoco/model/drone/Model_GRVC_Aerial_Manipulator_3Cable.xml"
                print("Simulating aerial manipulator hanging by 3 cables.")
                self._droneController.setHeightReference(2)
            case "4 cable":
                MODEL_PATH = "/home/nicolas/tfg/mujoco/model/drone/Model_GRVC_Aerial_Manipulator_4Cable.xml"
                print("Simulating aerial manipulator hanging by 4 cables.")
                self._droneController.setHeightReference(2)
            case _:
                MODEL_PATH = "/home/nicolas/tfg/mujoco/model/drone/Model_GRVC_Drone.xml"
                print("Simulating default model: drone.")
                self._droneController.setHeightReference(0.45)

        self._model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self._data = mujoco.MjData(self._model)
        self._shutdownEvent = shutdownEvent
        self._viewer = None
        self._viewerThread = None
        self._viewerRunning = threading.Event()
        self._sensors = [self._model.sensor(i).name for i in range(self._model.nsensor)]
        self._actuatorsIDs = {
            self._model.actuator(i).name: i for i in range(self._model.nu)
        }

        hookNamesByModel = {
            "1 cable": [
                "drone_side_center",
                "arms_side_center",
            ],
            "2 cable": [
                "drone_side_right",
                "drone_side_left",
                "arms_side_right",
                "arms_side_left",
            ],
            "4 cable": [
                "drone_side_right",
                "drone_side_left",
                "drone_side_back",
                "drone_side_front",
                "arms_side_right",
                "arms_side_left",
                "arms_side_back",
                "arms_side_front",
            ],
        }
        self._armsHooks = []
        self._droneHooks = []
        self._hookSiteNamesIDs = {}
        if model in hookNamesByModel:
            for name in hookNamesByModel[model]:
                sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)
                if sid == -1:
                    raise RuntimeError(f"Site {name} not found in model.")
                self._hookSiteNamesIDs[name] = sid
        self._hookSiteNames = list(self._hookSiteNamesIDs.keys())
        self._hookSiteIds = np.array(
            list(self._hookSiteNamesIDs.values()), dtype=np.int32
        )
        hook_pairs = [
            ("drone_side_center", "arms_side_center"),
            ("drone_side_right", "arms_side_right"),
            ("drone_side_left", "arms_side_left"),
            ("drone_side_back", "arms_side_back"),
            ("drone_side_front", "arms_side_front"),
        ]

        self._droneHookIds = []
        self._armsHookIds = []

        for drone_name, arm_name in hook_pairs:
            if (
                drone_name in self._hookSiteNamesIDs
                and arm_name in self._hookSiteNamesIDs
            ):
                self._droneHookIds.append(self._hookSiteNamesIDs[drone_name])
                self._armsHookIds.append(self._hookSiteNamesIDs[arm_name])

        self._droneHookIds = np.array(self._droneHookIds, dtype=np.int32)
        self._armsHookIds = np.array(self._armsHookIds, dtype=np.int32)

        self._licasJoints = ["q11", "q12", "q13", "q14", "q21", "q22", "q23", "q24"]
        self._jointIndices = [
            self._model.joint(name).qposadr for name in self._licasJoints
        ]
        self._velocityIndices = [
            self._model.joint(name).dofadr for name in self._licasJoints
        ]
        self._fastReader = FastSensorReader(
            self._jointIndices,
            self._velocityIndices,
            self._droneHookIds,
            self._armsHookIds,
        )

        self._sensorInfo = []
        for i in range(self._model.nsensor):
            name = self._model.sensor(i).name
            dim = int(self._model.sensor(i).dim)
            self._sensorInfo.append((name, dim))

        self._dataLock = threading.Lock()

        self._timeDebugging = False
        self._debugging = False

    def run(self):
        print("[MujocoSimulation] Initialized.")
        time.sleep(1)

        real_start = time.time()
        with self._dataLock:
            sim_start = np.copy(self._data.time)

        # Start viewer thread
        self._viewerRunning.set()
        self._viewerThread = threading.Thread(target=self.launchViewer)
        self._viewerThread.start()

        # Initialize key variables
        droneID = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        if droneID == -1:
            raise RuntimeError("Body 'drone' not found in model.")

        manipulatorID = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "LiCAS_A1"
        )
        if manipulatorID == -1:
            raise RuntimeError("Body 'LiCAS_A1' not found in model.")

        actuadoresConIDMenosUno = [k for k, v in self._actuatorsIDs.items() if v == -1]
        if len(actuadoresConIDMenosUno) > 0:
            print(f"[ERROR] Actuadores con ID -1: {actuadoresConIDMenosUno}")

        iteration = 0
        timestep = self._model.opt.timestep
        nextTime = time.perf_counter()

        while not self._shutdownEvent.is_set():
            tStart = time.perf_counter()
            timeDebug = self._timeDebugging and iteration % 200 == 0
            debug = self._debugging and iteration % 200 == 0
            if timeDebug:
                t0 = time.time()
            iteration += 1

            # iteration = self.testDroneControl(droneID, iteration)

            ###########################################################
            # --- Get sensor data and send to DroneControl thread ---
            ###########################################################
            with self._dataLock:
                (
                    droneGlobalPosition,
                    manipulatorGlobalPosition,
                    qpos,
                    qvel,
                    currentTimestamp,
                    self._droneHooks,
                    self._armsHooks,
                ) = self._fastReader.read(self._data, droneID, manipulatorID)
                sensorData = np.copy(self._data.sensordata)

            if timeDebug:
                t0_1 = time.time()

            if iteration == 1:
                print(
                    f"[MujocoSimulation] Initial Manipulator position: {manipulatorGlobalPosition}, Initial Drone position: {droneGlobalPosition}"
                )

            droneSensorValues, robotSensorValues = extract_sensor_values(
                sensorData, self._sensorInfo
            )

            if timeDebug:
                t0_2 = time.time()

            manipulatorGlobalPosition = [0, 0, 0]
            sensorDataMessage = SensorData(
                timestamp=currentTimestamp,
                timestep=self._model.opt.timestep,
                sensorValues=droneSensorValues,
                dronePosition=droneGlobalPosition,
            )

            if timeDebug:
                t0_3 = time.time()

            if self._swayControllerEnabled:
                robotSensorDataMessage = RobotSensorData(
                    timestamp=currentTimestamp,
                    timestep=self._model.opt.timestep,
                    sensorValues=robotSensorValues,
                    manipulatorPosition=manipulatorGlobalPosition,
                    dronePosition=droneGlobalPosition,
                    jointsPosition=qpos,
                    jointsVelocity=qvel,
                    droneAcceleration=droneSensorValues["body_linacc"],
                    armsHooks=self._armsHooks,
                    droneHooks=self._droneHooks,
                )

            if timeDebug:
                t0_4 = time.time()

            self._droneController.setSensorData(sensorDataMessage)
            if self._swayControllerEnabled:
                self._swayController.setSensorData(robotSensorDataMessage)
            if debug:
                print(f"[MujocoSimulation] sensorDataMessage: {sensorDataMessage}")

            if timeDebug:
                t1 = time.time()

            ###########################################################
            # --- Get control output from DroneControl thread ---
            ###########################################################
            controlAction: ControlAction = self._droneController.getControlInputs()
            # allControlActions = {**controlAction.droneThrusters}
            allControlActions = dict(controlAction.droneThrusters)
            if self._swayControllerEnabled:
                robotControlAction: RobotControlAction = (
                    self._swayController.getControlInputs()
                )
                allControlActions.update(robotControlAction.jointMotors)

            with self._dataLock:
                apply_control_inputs(
                    self._data.ctrl, self._actuatorsIDs, allControlActions
                )

            if timeDebug:
                t2 = time.time()

            # Simulation step: timestep = 0.001s
            with self._dataLock:
                mujoco.mj_step(self._model, self._data)
            if not self._droneController._updateEvent.is_set():
                self._droneController.setUpdateEvent()
            if not self._swayController._updateEvent.is_set():
                self._swayController.setUpdateEvent()

            if timeDebug:
                t3 = time.time()

            # Maintain a constant simulation period.
            tFinal = time.perf_counter()
            elapsedCycle = tFinal - tStart

            nextTime += timestep
            now = time.perf_counter()
            sleepDuration = nextTime - now

            if timeDebug:
                real_elapsed = time.time() - real_start
                sim_elapsed = currentTimestamp - sim_start

                print(
                    f"[MujocoSimulation] Elapsed Cycle: {elapsedCycle}, Sleep Duration; {sleepDuration}."
                )
                print(
                    f"[MujocoSimulation] Sim Time: {sim_elapsed:.2f} s, Real Time: {real_elapsed:.2f} s"
                )
                elapsedRecvControl = t2 - t1
                elapsedSendData = t1 - t0
                elapsedStep = t3 - t2
                elapsedTotal = t3 - t0
                print(
                    f"[MujcoSimulation] Step: {elapsedStep:.6f}, recv control: {elapsedRecvControl:.6f}, send sensors: {elapsedSendData:.6f}"
                )
                print(
                    f"[MujocoSimulation] extract data: {(t0_1-t0):.6f}. extract sensors: {(t0_2-t0_1):.6f}, DroneSensorData: {(t0_3-t0_2):.6f}, RobotSensorData: {(t0_4-t0_3):.6f}, Set sensor messages: {(t1-t0_4):.6f}"
                )
                print(f"[MujocoSimulation]  TOTAL: {elapsedTotal:.6f}")

            if sleepDuration > 0.0001:
                time.sleep(sleepDuration - 0.00005)  # Sleeps most of the time

            # Busy-wait the last ~50 µs
            while time.perf_counter() < nextTime:
                pass

            if timeDebug:
                totalCycle = time.perf_counter() - tStart
                print(f"[MujocoSimulation] Total Cycle: {totalCycle}.")

        print("[MujocoSimulation] Ending simulation.")
        self._viewerRunning.clear()
        self._viewerThread.join()

    def stop(self):
        print("[MujocoSimulation] Ending simulation.")
        self._shutdownEvent.set()
        self._viewerRunning.clear()
        if self._viewerThread:
            self._viewerThread.join()

    def getHookSitesPositions(self):
        """Returns a dictionary {siteName: globalPosition} for all hook sites."""
        return {
            name: np.copy(self._data.site_xpos[sid])
            for name, sid in self._hookSiteNamesIDs.items()
        }

    def setViewersCamera(self, droneID):
        self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._viewer.cam.trackbodyid = droneID
        self._viewer.cam.distance = 5.0
        self._viewer.cam.azimuth = 0
        self._viewer.cam.elevation = -10

    def testDroneControl(self, droneID, iteration):

        if iteration == 200:
            appliedTorque = np.random.randn() * 10
            with self._dataLock:
                self._data.xfrc_applied[droneID] = [0, 0, 0, appliedTorque, 0, 0]
            print(f"Applied torque: {appliedTorque}")
            return 0
        else:
            return iteration + 1

    def launchViewer(self):
        """Launch passive viewer on a separate thread."""
        n = 0
        periodSum = 0
        PERIOD = 0.01
        with mujoco.viewer.launch_passive(
            self._model, self._data, show_left_ui=False, show_right_ui=True
        ) as viewer:
            self._viewer = viewer
            droneID = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "drone")
            self.setViewersCamera(droneID)
            while self._viewerRunning.is_set():
                try:
                    ts = time.perf_counter()
                    with self._dataLock:
                        self._viewer.sync()
                        if self._debugging and n % 200 == 0:
                            print("[Viewer Thread] Data locked.")
                    n += 1
                    tf = time.perf_counter()
                    periodSum += tf - ts
                    sleepTime = PERIOD - (tf - ts)
                    if sleepTime > 0:
                        time.sleep(sleepTime)  # FPS control (100 Hz)
                except Exception as e:
                    print(f"[Viewer Thread] Error: {e}")
                    break

            self._viewer.close()
        avgPeriod = periodSum / n
        avgFrequency = 1 / avgPeriod
        print(
            f"[MujocViewerThread] Average Frequency: {avgFrequency}, Number of Syncs: {n}"
        )


if __name__ == "__main__":
    MujocoSimulation.run()


class FastSensorReader:
    def __init__(self, jointIndices, velocityIndices, droneHookIds, armsHookIds):
        self.jointIndices = np.array(jointIndices, dtype=np.int32)
        self.velocityIndices = np.array(velocityIndices, dtype=np.int32)
        self._droneHookIds = droneHookIds
        self._armsHookIds = armsHookIds

    def read(self, data, droneId, manipulatorId):

        return extract_data(
            data.xpos,
            droneId,
            manipulatorId,
            data.qpos,
            data.qvel,
            self.jointIndices,
            self.velocityIndices,
            data.time,
            self._droneHookIds,
            self._armsHookIds,
        )
