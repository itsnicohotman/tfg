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
        sensorQueue: queue.Queue,
        controlQueue: queue.Queue,
        sendSensorsEvent: threading.Event,
        recvControlEvent: threading.Event,
        shutdownEvent: threading.Event,
        model,
        droneController: DroneControl,
    ):
        super().__init__()  # Start thread
        match model:
            case "drone":
                MODEL_PATH = "/home/nicolas/tfg/mujoco/model/drone/Model_GRVC_Drone.xml"
                print("Simulating drone.")
            case "0 cable":
                MODEL_PATH = "/home/nicolas/tfg/mujoco/model/drone/Model_GRVC_Aerial_Manipulator_NoCable.xml"
                print("Simulating aerial manipulator hanging by 0 cables.")
            case "1 cable":
                MODEL_PATH = "/home/nicolas/tfg/mujoco/model/drone/Model_GRVC_Aerial_Manipulator_1Cable.xml"
                print("Simulating aerial manipulator hanging by1 cable.")
            case "2 cable":
                MODEL_PATH = "/home/nicolas/tfg/mujoco/model/drone/Model_GRVC_Aerial_Manipulator_2Cable.xml"
                print("Simulating aerial manipulator hanging by2 cables.")
            case "3 cable":
                MODEL_PATH = "/home/nicolas/tfg/mujoco/model/drone/Model_GRVC_Aerial_Manipulator_3Cable.xml"
                print("Simulating aerial manipulator hanging by 3 cables.")
            case "4 cable":
                MODEL_PATH = "/home/nicolas/tfg/mujoco/model/drone/Model_GRVC_Aerial_Manipulator_4Cable.xml"
                print("Simulating aerial manipulator hanging by 4 cables.")
            case _:
                MODEL_PATH = "/home/nicolas/tfg/mujoco/model/drone/Model_GRVC_Drone.xml"
                print("Simulating default model: drone.")

        self._model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self._data = mujoco.MjData(self._model)
        self._sensorQueue = sensorQueue
        self._controlQueue = controlQueue
        self._sendSensorsEvent = sendSensorsEvent
        self._recvControlEvent = recvControlEvent
        self._shutdownEvent = shutdownEvent
        self._viewer = None
        self._viewerThread = None
        self._viewerRunning = threading.Event()
        self._sensors = [self._model.sensor(i).name for i in range(self._model.nsensor)]
        self._actuatorsIDs = {
            self._model.actuator(i).name: i for i in range(self._model.nu)
        }
        self._droneController = droneController

    def run(self):
        real_start = time.time()
        sim_start = self._data.time
        self._viewerRunning.set()
        self._viewerThread = threading.Thread(target=self.launchViewer)
        self._viewerThread.start()
        # with mujoco.viewer.launch_passive(
        #     self._model, self._data, show_left_ui=False, show_right_ui=False
        # ) as viewer:
        #     self._viewer = viewer
        sensorValues = {}
        droneID = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        droneGlobalPosition = self._data.xpos[droneID]
        """ manipulatorID = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, 'manipulator')
        manipulatorGlobalPosition = self._data.xpos[manipulatorID] """
        # self.setViewersCamera(droneID)
        iteration = 0
        timestep = self._model.opt.timestep
        nextTime = time.time()

        while not self._shutdownEvent.is_set():
            iteration += 1
            t0 = time.time()
            mujoco.mj_step(self._model, self._data)
            t05 = time.time()
            # PROBAR CON DIFERENTES MANERAS DE ALIVIAR LA CARGA COMPUTACIONAL DEL VIEWER. POR EJEMPLO, HILO SEPARADO O RENDERIZANDO CON FRECUENCIA DE 100HZ
            # if iteration % 10 == 0:
            # self._viewer.sync()
            # self._viewer
            # t1 = time.time()
            # iteration = self.testDroneControl(droneID, iteration)
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

            self._sensorQueue.put(sensorDataMessage)
            self._sendSensorsEvent.set()

            t2 = time.time()

            self._recvControlEvent.wait(timeout=0.0001)
            t21 = time.time()
            try:
                controlAction: ControlAction = self._controlQueue.get_nowait()
                allControlActions = {**controlAction.droneThrusters}
                for name, value in allControlActions.items():
                    if name in self._actuatorsIDs:
                        self._data.ctrl[self._actuatorsIDs[name]] = value
                    else:
                        print(f"[WARNING] Actuator '{name}' not found in model.")

                self._recvControlEvent.clear()
            except queue.Empty:
                pass

            t3 = time.time()

            if iteration % 1000 == 0:
                elapsedRecvControl = t3 - t2
                elapsedRecvControlNoWait = t3 - t21
                elapsedSendData = t2 - t05
                elapsedStep = t05 - t0
                elapsedTotal = t3 - t0
                # elapsedViewer = t1 - t05
                # print(
                #     f"[TIMING] Step: {elapsedStep:.6f}, Viewer: {elapsedViewer}, receive control: {elapsedRecvControl:.6f}, send sensors: {elapsedSendData:.6f}"
                # )
                print(
                    f"[TIMING] Step: {elapsedStep:.6f}, recv control: {elapsedRecvControl:.6f},recv control without Wait: {elapsedRecvControlNoWait:.6f}, send sensors: {elapsedSendData:.6f}, TOTAL: {elapsedTotal}"
                )
                real_elapsed = time.time() - real_start
                sim_elapsed = self._data.time - sim_start
                print(f"[Tiempos] Sim: {sim_elapsed:.2f} s, Real: {real_elapsed:.2f} s")

            nextTime += timestep
            sleepDuration = nextTime - time.time()
            if sleepDuration > 0:
                time.sleep(sleepDuration)
                if iteration % 100 == 0:
                    print(f"[TIMING] Sleep duration: {sleepDuration}")
            else:
                nextTime = time.time()

        print("[MujocoSimulation] Ending simulation.")
        self._viewerRunning.clear()
        self._viewerThread.join()

    def stop(self):
        print("[MujocoSimulation] Ending simulation.")
        self._shutdownEvent.set()
        # self._viewer.close()
        self._viewerRunning.clear()
        if self._viewerThread:
            self._viewerThread.join()

    def setViewersCamera(self, droneID):
        # self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
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
                    self._viewer.sync()  # Usa state_only=True si tu versión lo soporta
                    time.sleep(0.001)  # Control de FPS (1000 Hz)
                except Exception as e:
                    print(f"[Viewer Thread] Error: {e}")
                    break

            self._viewer.close()


if __name__ == "__main__":
    MujocoSimulation.run()
