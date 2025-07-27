"""This class creates a separate thread where a passive non-blocking simulaton viewer is executed.
This thread collects the model's sensors' values and sends them to the control thread.
It waits fot the control thread's answer with the correct control inputs that are later implemented in the simulation.
"""

import mujoco
import mujoco.viewer
import threading
import numpy as np
import queue
from include.Messages import SensorData, ControlAction
import time
from src.drone.DroneControl import DroneControl


class MujocoSimulation(threading.Thread):
    def __init__(
        self,
        shutdownEvent: threading.Event,
        model,
        droneController: DroneControl,
    ):
        super().__init__()  # Start thread
        self._droneController = droneController
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

        self._timeDebugging = False

    def run(self):
        print("[MujocoSimulation] Initialized.")
        real_start = time.time()
        sim_start = self._data.time

        # Start viewer thread
        self._viewerRunning.set()
        self._viewerThread = threading.Thread(target=self.launchViewer)
        self._viewerThread.start()

        # Initialize key variables
        sensorValues = {}
        droneID = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        droneGlobalPosition = self._data.xpos[droneID]
        if droneID == -1:
            raise RuntimeError("Body 'drone' not found in model.")

        actuadoresConIDMenosUno = [k for k, v in self._actuatorsIDs.items() if v == -1]
        if len(actuadoresConIDMenosUno) > 0:
            print(f"[ERROR] Actuadores con ID -1: {actuadoresConIDMenosUno}")

        """ manipulatorID = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, 'manipulator')
        manipulatorGlobalPosition = self._data.xpos[manipulatorID] """
        iteration = 0
        timestep = self._model.opt.timestep
        nextTime = time.perf_counter()

        time.sleep(0.5)
        while not self._shutdownEvent.is_set():
            tStart = time.perf_counter()
            timeDebug = self._timeDebugging and iteration % 200 == 0
            if timeDebug:
                t0 = time.time()
            iteration += 1

            # iteration = self.testDroneControl(droneID, iteration)

            # Get sensor data and send to DroneControl thread
            index = 0
            for i in range(self._model.nsensor):
                name = self._model.sensor(i).name
                dim = int(self._model.sensor(i).dim)
                sensorValues[name] = self._data.sensordata[index : index + dim]
                index += dim

            manipulatorGlobalPosition = [0, 0, 0]
            sensorDataMessage = SensorData(
                timestamp=self._data.time,
                timestep=self._model.opt.timestep,
                sensorValues=sensorValues,
                dronePosition=droneGlobalPosition,
                manipulatorPosition=manipulatorGlobalPosition,
            )

            self._droneController.setSensorData(sensorDataMessage)

            # Get control output from DroneControl thread
            if timeDebug:
                t1 = time.time()
            try:
                controlAction: ControlAction = self._droneController.getControlInputs()
                allControlActions = {**controlAction.droneThrusters}
                for name, value in allControlActions.items():
                    if name in self._actuatorsIDs:
                        self._data.ctrl[self._actuatorsIDs[name]] = value
                    else:
                        print(f"[WARNING] Actuator '{name}' not found in model.")

            except queue.Empty:
                pass

            if timeDebug:
                t2 = time.time()

            # Simulation step: timestep = 0.001s
            mujoco.mj_step(self._model, self._data)
            if not self._droneController._updateEvent.is_set():
                self._droneController.setUpdateEvent()

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
                sim_elapsed = self._data.time - sim_start

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
                    f"[MujcoSimulation] Step: {elapsedStep:.6f}, recv control: {elapsedRecvControl:.6f}, send sensors: {elapsedSendData:.6f}, TOTAL: {elapsedTotal}"
                )

            if sleepDuration > 0.0001:
                time.sleep(sleepDuration - 0.00005)  # Sleeps most of the time

            # Busy-wait the last ~50 µs
            while time.perf_counter() < nextTime:
                pass

            if timeDebug:
                totalCycle = time.perf_counter() - tStart
                print(f"[MujocoSimulation] Total Cycle: {totalCycle}.")

        # print("[MujocoSimulation] Ending simulation.")
        self._viewerRunning.clear()
        self._viewerThread.join()

    def stop(self):
        print("[MujocoSimulation] Ending simulation.")
        self._shutdownEvent.set()
        self._viewerRunning.clear()
        if self._viewerThread:
            self._viewerThread.join()

    def setViewersCamera(self, droneID):
        self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._viewer.cam.trackbodyid = droneID
        self._viewer.cam.distance = 5.0
        self._viewer.cam.azimuth = 0
        self._viewer.cam.elevation = -10

    def testDroneControl(self, droneID, iteration):

        if iteration == 200:
            appliedTorque = np.random.randn() * 10
            self._data.xfrc_applied[droneID] = [0, 0, 0, appliedTorque, 0, 0]
            print(f"Applied torque: {appliedTorque}")
            return 0
        else:
            return iteration + 1

    def launchViewer(self):
        """Launch passive viewer on a separate thread."""
        with mujoco.viewer.launch_passive(
            self._model, self._data, show_left_ui=False, show_right_ui=True
        ) as viewer:
            self._viewer = viewer
            droneID = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "drone")
            self.setViewersCamera(droneID)
            while self._viewerRunning.is_set():
                try:
                    self._viewer.sync()
                    time.sleep(0.001)  # FPS control (1000 Hz)
                except Exception as e:
                    print(f"[Viewer Thread] Error: {e}")
                    break

            self._viewer.close()


if __name__ == "__main__":
    MujocoSimulation.run()
