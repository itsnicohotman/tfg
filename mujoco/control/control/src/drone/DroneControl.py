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
        self._heightActionQueue = queue.Queue()
        self._heightQueue = queue.Queue()
        self._heightKp = 0.0  # 40 parece un buen valor para empezar a configurar el PID
        self._heightKi = 0.0
        self._heightKd = 0.0
        self._heightController = DroneHeightController(
            self._shutdownEvent,
            self._heightControllerUpdateEvent,
            self._heightControllerResultsEvent,
            self._heightKp,
            self._heightKi,
            self._heightKd,
            self._heightQueue,
            self._heightActionQueue,
        )
        self._heightController.start()

        # Stability Controller
        self._stabilityUpdateEvent = threading.Event()
        self._stabilityResultEvent = threading.Event()
        self._stabilityReferencesUpdateEvent = threading.Event()
        self._stabilityUpdateEvent.clear()
        self._stabilityResultEvent.clear()
        self._stabilityReferencesUpdateEvent.clear()
        self._orientationQueue = queue.Queue()
        self._attitudeReferenceQueue = queue.Queue()
        self._attitudeActionQueue = queue.Queue()
        self._stabilityPKp = [10, 0.0, 0.0]  # Roll, Pitch, Yaw
        self._stabilityPidKp = [15, 0.0, 0.0]
        self._stabilityPidKi = [10, 0.0, 0.0]
        self._stabilityPidKd = [0, 0.0, 0.0]

        self._stabilityController = DroneStabilityController(
            self._shutdownEvent,
            self._stabilityUpdateEvent,
            self._stabilityResultEvent,
            self._stabilityReferencesUpdateEvent,
            self._stabilityPKp,
            self._stabilityPidKp,
            self._stabilityPidKi,
            self._stabilityPidKd,
            self._orientationQueue,
            self._attitudeActionQueue,
            self._attitudeReferenceQueue,
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
            },
            dronePosition=[0.0, 0.0, 1.09],
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
        self._currentVelocity = [0.0, 0.0, 0.0]
        self._currentOrientation = [0.0, 0.0, 0.0, 0.0]

        # self._coordinatesController:DroneCoordinatesController
        # self._coordinatesController.start()

    def getControlMessage(self):
        self._taskManagerManeuverEvent.wait()
        try:
            self._controlMessage = self._droneControlQueue.get(timeout=0.1)
        except queue.Empty:
            pass

    def getSensorsData(self):
        self._sendSensorsEvent.wait()
        self._sensorData = self._sensorQueue.get(timeout=0.1)
        self._sendSensorsEvent.clear()

    def sendControlInputs(self):
        while self._recvControlEvent.is_set():
            continue
        self._controlQueue.put(self._controlAction)
        self._recvControlEvent.set()

    def run(self):
        print("[DroneControl] Initialized.")

        while not self._shutdownEvent.is_set():
            # self.getControlMessage()
            # print("[DroneControl] Control Message received.")
            self.getSensorsData()
            # print("[DroneControl] Sensors Data received.")
            self.updateCurrentState()
            # print("[DroneControl] Current State updated.")

            # self.updateHeightController()
            # print("[DroneControl] Height Controller updated.")
            self.updateStabilityController()
            # print("[DroneControl] Stability Controller updated.")
            self.updateCoordinatesController()
            self.combineControllersOutputs()
            # print("[DroneControl] Control Outputs combined.")
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
        self._currentVelocity = self._sensorData.sensorValues["body_gyro"]
        self._currentOrientation = self._sensorData.sensorValues["body_quat"]
        self._currentCOM = self._sensorData.sensorValues["drone_load_com"]

    def updateHeightController(self):

        while self._heightControllerUpdateEvent.is_set():
            # print("[DroneControl] Waiting to update height Controller")
            continue
        # print("[DroneControl] Height Controller update sent.")
        (timestamp, currentHeight, heightReference) = (
            self._currentTimestamp,
            self._currentPosition[2],
            self._controlMessage.referenceZ,
        )
        self._heightQueue.put((timestamp, currentHeight, heightReference))
        self._heightControllerUpdateEvent.set()

    def updateStabilityController(self):
        while self._stabilityUpdateEvent.is_set():
            continue
        orientationData = (
            self._currentTimestamp,
            self._timestep,
            self._currentOrientation,
            self._currentVelocity,
        )
        self._orientationQueue.put(orientationData)
        self._stabilityUpdateEvent.set()

        # include when coordinates controller (which sets drone angles references) is created
        """ while self._stabilityReferencesUpdateEvent.is_set():
            continue
        # To be completed when the needed angles for a specific position calculations are made --> coordinates controller
        targetReferenceAngles = [targetRoll, targetPitch, targetYaw]
        self._attitudeReferenceQueue.put(targetReferenceAngles)
        self._stabilityReferencesUpdateEvent.set() """

    def updateCoordinatesController(self):
        pass

    def combineControllersOutputs(self):
        baseThrust = 0.0
        # baseThrust = self.getHeightControllerOutput()
        stabillityRotorAdjustments = self.getStabilityControllerOutput()
        MAX_CONTROL_INPUT = 10
        MIN_CONTROL_INPUT = -10

        thrust1 = max(
            min(baseThrust + stabillityRotorAdjustments[0], MAX_CONTROL_INPUT),
            MIN_CONTROL_INPUT,
        )
        thrust2 = max(
            min(baseThrust + stabillityRotorAdjustments[1], MAX_CONTROL_INPUT),
            MIN_CONTROL_INPUT,
        )
        thrust3 = max(
            min(baseThrust + stabillityRotorAdjustments[2], MAX_CONTROL_INPUT),
            MIN_CONTROL_INPUT,
        )
        thrust4 = max(
            min(baseThrust + stabillityRotorAdjustments[3], MAX_CONTROL_INPUT),
            MIN_CONTROL_INPUT,
        )

        self._controlAction.droneThrusters = {
            "propeller-1_motor": thrust1,  # Front-left - CW
            "propeller-2_motor": thrust2,  # Front-right - CCW
            "propeller-3_motor": thrust3,  # Back-right - CW
            "propeller-4_motor": thrust4,  # Back-left - CCW
        }

    # print(f"[DroneControl] Base thrust={baseThrust}, rotorAdjustments={stabillityRotorAdjustments}.")

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
        try:
            thruster = self._heightActionQueue.get(timeout=0.1)
        except queue.Empty:
            pass
        finally:
            self._heightControllerResultsEvent.clear()

        return thruster

    def getStabilityControllerOutput(self):
        self._stabilityResultEvent.wait()
        try:
            stabillityRotorAdjustments = self._attitudeActionQueue.get(timeout=0.1)
        except queue.Empty:
            stabillityRotorAdjustments = [0.0, 0.0, 0.0, 0.0]
        finally:
            self._stabilityResultEvent.clear()
        return stabillityRotorAdjustments
