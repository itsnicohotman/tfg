import threading
import time
import queue
from src.drone.droneControllers.DroneHeightController import DroneHeightController
from src.drone.droneControllers.DroneStabilityController import DroneStabilityController
from include.Messages import ControlMessage, SensorData, ControlAction
import numpy as np


class DroneControl(threading.Thread):
    def __init__(
        self,
        shutdownEvent: threading.Event,
        maneuverEvent: threading.Event,
        sendSensorsEvent: threading.Event,
        recvControlEvent: threading.Event,
        sensorQueue: queue.Queue,
        controlQueue: queue.Queue,
        droneControlQueue: queue.Queue,
    ):
        super().__init__()
        self._taskManagerManeuverEvent = maneuverEvent
        self._shutdownEvent = shutdownEvent
        self._sendSensorsEvent = sendSensorsEvent
        self._recvControlEvent = recvControlEvent

        # Height Controller
        self._heightControllerUpdateEvent = threading.Event()
        self._heightControllerResultsEvent = threading.Event()
        self._heightControllerUpdateEvent.clear()
        self._heightControllerResultsEvent.clear()
        self._heightActionLock = threading.Lock()
        self._heightLock = threading.Lock()
        self._rotorForce = 0.0
        self._heightData = {
            "timestamp": 0.0,
            "currentHeight": 0.0,
            "heightReference": 0.45,
            "currentVelocity": [0.0, 0.0, 0.0],
        }
        self._heightPKp = 3.35
        self._heightPidKp = 25.0
        self._heightPidKi = 20.0
        self._heightPidKd = 0.0
        self._heightController = DroneHeightController(
            self._shutdownEvent,
            self._heightControllerUpdateEvent,
            self._heightControllerResultsEvent,
            self._heightPidKp,
            self._heightPidKi,
            self._heightPidKd,
            self._heightPKp,
            self._heightData,
            self._heightLock,
            self._heightActionLock,
        )
        self._heightController.start()

        # Stability Controller
        self._stabilityUpdateEvent = threading.Event()
        self._stabilityResultEvent = threading.Event()
        self._stabilityUpdateEvent.clear()
        self._stabilityResultEvent.clear()
        self._stabilityActionLock = threading.Lock()
        self._rotorAdjustments = [
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # [Rotor1. Rotor2. Rotor3, Rotor4]
        self._stabilityPKp = [10, 10, 5]  # Roll, Pitch, Yaw
        self._stabilityPidKp = [15, 15, 15]
        self._stabilityPidKi = [10, 10, 15]
        self._stabilityPidKd = [0, 0.0, 0.0]
        self._stabilityLock = threading.Lock()
        self._stabilityData = {
            "timestamp": 0.0,
            "timestep": 0.0,
            "currentOrientation": [0.0, 0.0, 0.0, 0.0],
            "orientationReference": [0.0, 0.0, 0.0],  # Roll, pitch, yaw
            "currentAngularVelocity": [0.0, 0.0, 0.0],
        }

        self._stabilityController = DroneStabilityController(
            self._shutdownEvent,
            self._stabilityUpdateEvent,
            self._stabilityResultEvent,
            self._stabilityPKp,
            self._stabilityPidKp,
            self._stabilityPidKi,
            self._stabilityPidKd,
            self._stabilityLock,
            self._stabilityData,
            self._stabilityActionLock,
            self._rotorAdjustments,
        )
        self._stabilityController.start()

        self._sensorData = SensorData(
            timestamp=0.0,
            timestep=0.001,
            sensorValues={
                "body_linacc": [0.0, 0.0, 0.0],
                "body_gyro": [0.0, 0.0, 0.0],
                "body_quat": [1.0, 0.0, 0.0, 0.0],  # orientación neutral
                "drone_load_com": [0.0, 0.0, 0.0],
                "body_vel": [0.0, 0.0, 0.0],
            },
            dronePosition=[0.0, 0.0, 0.45],
            manipulatorPosition=[0.0, 0.0, 0.0],
        )
        self._controlAction = ControlAction(droneThrusters={})
        self._controlMessage: ControlMessage

        self._droneControlQueue = droneControlQueue  # receives from TaskManager Thread
        self._sensorQueue = sensorQueue  # receives from MuJoCo Simulation Thread
        self._controlQueue = controlQueue  # sends to MuJoCo Simulation Thread

        self._lastCheckTime = None
        self._reachedReference = False
        self._toleranceThreshold = 0.01
        self._stabilityTime = 2

        self._currentTimestamp = 0.0
        self._timestep = 0.001
        self._previousTimestamp = 0.0
        self._currentPosition = [0.0, 0.0, 0.0]
        self._currentCOM = [0.0, 0.0, 0.0]
        self._currentAcceleration = [0.0, 0.0, 0.0]
        self._currentAngularVelocity = [0.0, 0.0, 0.0]
        self._currentVelocity = [0.0, 0.0, 0.0]
        self._currentOrientation = [0.0, 0.0, 0.0, 0.0]

        # self._coordinatesController:DroneCoordinatesController
        # self._coordinatesController.start()

    def getControlMessage(self):
        self._taskManagerManeuverEvent.wait()
        try:
            self._controlMessage = self._droneControlQueue.get(timeout=0.001)
        except queue.Empty:
            pass

    def getSensorsData(self):
        self._sendSensorsEvent.wait()
        self._sensorData = self._sensorQueue.get(timeout=0.001)
        self._sendSensorsEvent.clear()

    def sendControlInputs(self):
        while self._recvControlEvent.is_set():
            continue
        self._controlQueue.put(self._controlAction)
        self._recvControlEvent.set()

    def run(self):
        print("[DroneControl] Initialized.")

        while not self._shutdownEvent.is_set():
            self.getControlMessage()
            self.getSensorsData()
            self.updateCurrentState()

            self.updateHeightController()
            self.updateStabilityController()
            self.updateCoordinatesController()
            self.combineControllersOutputs()
            # self.checkReachedReferences()

            self.sendControlInputs()

        print("[DroneControl] Shutting down.")

    def stop(self):
        print("[DroneControl] Shutting down.")
        self._heightController.stop()
        self._heightController.join()
        self._stabilityController.stop()
        self._stabilityController.join()
        # self._coordinatesController.stop()
        # self._coordinatesController.join()
        self._shutdownEvent.set()

    def updateCurrentState(self):
        self._previousTimestamp = self._currentTimestamp
        self._currentTimestamp = self._sensorData.timestamp
        self._timestep = self._sensorData.timestep
        self._currentPosition = self._sensorData.dronePosition
        self._currentAcceleration = self._sensorData.sensorValues["body_linacc"]
        self._currentAngularVelocity = self._sensorData.sensorValues["body_gyro"]
        self._currentOrientation = self._sensorData.sensorValues["body_quat"]
        self._currentCOM = self._sensorData.sensorValues["drone_load_com"]
        self._currentVelocity = self._sensorData.sensorValues["body_vel"]

    def updateHeightController(self):
        while self._heightControllerUpdateEvent.is_set():
            continue
        with self._heightLock:
            self._heightData["timestamp"] = self._currentTimestamp
            self._heightData["currentHeight"] = self._currentPosition[2]
            self._heightData["heightReference"] = (
                self._controlMessage.referenceZ  # o 3 si lo prefieres
            )
            self._heightData["currentVelocity"] = self._currentVelocity.copy()
        self._heightControllerUpdateEvent.set()

    def updateStabilityController(self):
        while self._stabilityUpdateEvent.is_set():
            continue
        with self._stabilityLock:
            self._stabilityData["timestamp"] = self._currentTimestamp
            self._stabilityData["timestep"] = self._timestep
            self._stabilityData["currentOrientation"] = self._currentOrientation
            self._stabilityData["orientationReference"] = [
                0.0,
                0.0,
                0.0,
            ]  # or self._orientationReference.copy() when coordinates controller is added
        self._stabilityUpdateEvent.set()

    def updateCoordinatesController(self):
        pass

    def combineControllersOutputs(self):
        baseThrust = 0.0
        baseThrust = self.getHeightControllerOutput()
        stabilityRotorAdjustments = self.getStabilityControllerOutput()
        MAX_CONTROL_INPUT = 30
        MIN_CONTROL_INPUT = -30

        thrust1 = max(
            min(baseThrust + stabilityRotorAdjustments[0], MAX_CONTROL_INPUT),
            MIN_CONTROL_INPUT,
        )
        thrust2 = max(
            min(baseThrust + stabilityRotorAdjustments[1], MAX_CONTROL_INPUT),
            MIN_CONTROL_INPUT,
        )
        thrust3 = max(
            min(baseThrust + stabilityRotorAdjustments[2], MAX_CONTROL_INPUT),
            MIN_CONTROL_INPUT,
        )
        thrust4 = max(
            min(baseThrust + stabilityRotorAdjustments[3], MAX_CONTROL_INPUT),
            MIN_CONTROL_INPUT,
        )

        self._controlAction.droneThrusters = {
            "propeller-1_motor": thrust1,  # Front-left - CW
            "propeller-2_motor": thrust2,  # Front-right - CCW
            "propeller-3_motor": thrust3,  # Back-right - CW
            "propeller-4_motor": thrust4,  # Back-left - CCW
        }

    def checkReachedReferences(self):
        if (
            self._currentPosition[2] / self._controlMessage.referenceZ
            < self._toleranceThreshold
        ):
            if self._lastCheckTime == None:
                self._lastCheckTime = time.time()
            else:
                elapsedTime = time.time() - self._lastCheckTime
                if elapsedTime >= self._stabilityTime:
                    self._reachedReferenceStable = True
        else:
            self._lastCheckTime = None
            self._reachedReferenceStable = False

        if self._reachedReferenceStable:
            self._taskManagerManeuverEvent.clear()
            print(
                "[DroneControl] Reference position has been reached. Ready to execute a new command."
            )

    def getHeightControllerOutput(self):
        self._heightControllerResultsEvent.wait()
        with self._heightActionLock:
            baseThrust = self._heightController._rotorForce
            print(f"Getting rotorForce (locked): {self._heightController._rotorForce}")
            self._heightControllerResultsEvent.clear()
        return baseThrust

    def getStabilityControllerOutput(self):
        self._stabilityResultEvent.wait()
        with self._stabilityActionLock:
            stabilityRotorAdjustments = self._rotorAdjustments
            self._stabilityResultEvent.clear()
        return stabilityRotorAdjustments
