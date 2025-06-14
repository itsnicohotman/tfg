""" This class creates a separate thread where a passive non-blocking simulaton viewer is executed.
This thread collects the model's sensors' values and sends them to the control thread. 
It waits fot the control thread's answer with the correct control inputs that are later implemented in the simulation. """

import mujoco
import mujoco.viewer
import threading
import numpy as np
import queue
from include.Messages import SensorData, ControlAction

class MujocoSimulation(threading.Thread):
    def __init__(self, sensorQueue: queue.Queue, controlQueue: queue.Queue, sendSensorsEvent: threading.Event, 
                 recvControlEvent: threading.Event, shutdownEvent:threading.Event, model):
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
        self._sensors = [self._model.sensor(i).name for i in range(self._model.nsensor)]
        self._actuatorsIDs = {self._model.actuator(i).name: i for i in range(self._model.nu)}

    def run(self):
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            self._viewer = viewer
            sensorValues = {}
            droneID = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, 'drone')
            droneGlobalPosition = self._data.xpos[droneID]
            """ manipulatorID = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, 'manipulator')
            manipulatorGlobalPosition = self._data.xpos[manipulatorID] """
            self.setViewersCamera(droneID)
            iteration = 0

            while not self._shutdownEvent.is_set():
                mujoco.mj_step(self._model, self._data)
                self._viewer.sync()
                # iteration = self.testDroneControl(droneID, iteration)
                index = 0
                for i in range(self._model.nsensor):
                    name = self._model.sensor(i).name
                    dim = int(self._model.sensor(i).dim)
                    sensorValues[name] = self._data.sensordata[index:index+dim]
                    index += dim
               
                manipulatorGlobalPosition = [0, 0, 0]
                sensorDataMessage = SensorData(timestamp = self._data.time, timestep = self._model.opt.timestep, sensorValues = sensorValues, 
                                               dronePosition = droneGlobalPosition, manipulatorPosition = manipulatorGlobalPosition)

                self._sensorQueue.put(sensorDataMessage)
                self._sendSensorsEvent.set()

                if self._recvControlEvent.wait(timeout=0.01):
                    try:
                        controlAction:ControlAction = self._controlQueue.get()
                        allControlActions = {**controlAction.droneThrusters}
                        for name, value in allControlActions.items():
                            if name in self._actuatorsIDs:
                                self._data.ctrl[self._actuatorsIDs[name]] = value
                            else:
                                print(f"[WARNING] Actuator '{name}' not found in model.")
                        
                        self._recvControlEvent.clear()
                    except queue.Empty:
                        pass      

        print("[MujocoSimulation] Ending simulation.")     

    def stop(self):
        print("[MujocoSimulation] Ending simulation.")
        self._viewer.close()
        self._shutdownEvent.set()

    def setViewersCamera(self, droneID):
        self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._viewer.cam.trackbodyid = droneID
        self._viewer.cam.distance = 5.0
        self._viewer.cam.azimuth = 180
        self._viewer.cam.elevation = -10

    def testDroneControl(self, droneID, iteration):
        
        if iteration == 200:
            appliedTorque = np.random.randn() * 10
            self._data.xfrc_applied[droneID] = [0, 0, 0, appliedTorque, 0, 0]
            print(f"Applied torque: {appliedTorque}")
            return 0
        else:
            return iteration + 1

if __name__ == "__main__":
    MujocoSimulation.run()
