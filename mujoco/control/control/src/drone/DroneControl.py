import threading
import time
import queue
import threading
from src.drone.droneControllers.DroneHeightController import DroneHeightController
from src.drone.droneControllers.DroneStabilityController import DroneStabilityController
from src.drone.droneControllers.DroneCoordinatesController import CoordinatesController
from include.Messages import ControlMessage, SensorData, ControlAction
import numpy as np


class DroneControl(threading.Thread):
    def __init__(
        self,
        shutdownEvent: threading.Event,
        maneuverEvent: threading.Event,
        droneControlQueue: queue.Queue,
    ):
        super().__init__()
        self._taskManagerManeuverEvent = maneuverEvent
        self._shutdownEvent = shutdownEvent

        self._updateEvent = threading.Event()

        # Height Controller
        self._heightControllerUpdateEvent = threading.Event()
        self._heightControllerUpdateEvent.clear()
        self._rotorForce = 0.0

        self._heightController = DroneHeightController(
            self._shutdownEvent,
            self._heightControllerUpdateEvent,
        )
        self._heightController.start()

        # Stability Controller
        self._stabilityUpdateEvent = threading.Event()
        self._stabilityUpdateEvent.clear()
        self._stabilityActionLock = threading.Lock()
        self._stabilityLock = threading.Lock()
        self._stabilityData = {
            "timestamp": 0.0,
            "timestep": 0.001,
            "currentOrientation": [0.0, 0.0, 0.0, 0.0],
            "orientationReference": [0.0, 0.0, 0.0],  # Roll, pitch, yaw
            "currentAngularVelocity": [0.0, 0.0, 0.0],
        }

        self._stabilityController = DroneStabilityController(
            self._shutdownEvent,
            self._stabilityUpdateEvent,
            self._stabilityLock,
            self._stabilityData,
            self._stabilityActionLock,
        )
        self._stabilityController.start()

        # Coordinates Controller
        self._coordinatesUpdateEvent = threading.Event()
        self._coordinatesAnglesLock = threading.Lock()
        self._coordinatesLock = threading.Lock()
        self._coordinatesData = {
            "timestamp": 0.0,
            "timestep": 0.001,
            "currentPosition": [0.0, 0.0, 0.0],
            "positionReference": [0.0, 0.0, 0.0],
            "currentVelocity": [0.0, 0.0, 0.0],
            "currentTargetVelocity": [0.0, 0.0, 0.0],
            "currentAcceleration": [0.0, 0.0, 0.0],
            "currentOrientation": [1.0, 0.0, 0.0, 0.0],
        }

        self._coordinatesController = CoordinatesController(
            self._shutdownEvent,
            self._coordinatesUpdateEvent,
            self._coordinatesLock,
            self._coordinatesData,
            self._coordinatesAnglesLock,
        )
        self._coordinatesController.start()

        self._previousSensorData = SensorData(
            timestamp=0.0,
            timestep=0.001,
            sensorValues={
                "body_linacc": [0.0, 0.0, 0.0],
                "body_gyro": [0.0, 0.0, 0.0],
                "body_quat": [1.0, 0.0, 0.0, 0.0],  # neutral orientation
                "drone_load_com": [0.0, 0.0, 0.0],
                "body_vel": [0.0, 0.0, 0.0],
            },
            # subtreeMass=0.0,
            dronePosition=[0.0, 0.0, 0.45],
            manipulatorPosition=[0.0, 0.0, 0.0],
        )
        self._sensorData: SensorData = None
        self._sensorLock = threading.Lock()
        self._controlAction = ControlAction(droneThrusters={})
        self._controlLock = threading.Lock()
        self._controlMessage: ControlMessage
        self._previousControlMessage = ControlMessage(
            grasp=False, referenceX=0.0, referenceY=0.0, referenceZ=0.45
        )

        self._droneControlQueue = droneControlQueue  # receives from TaskManager Thread

        self._lastCheckTime = None
        self._reachedReferences = False
        self._toleranceThreshold = 0.01
        self._stabilityTime = 2

        self._currentTimestamp = 0.0
        self._timestep = 0.001
        self._previousTimestamp = 0.0
        self._currentPosition = np.zeros(3)
        self._currentCOM = np.zeros(3)
        self._currentAcceleration = np.zeros(3)
        self._currentAngularVelocity = np.zeros(3)
        self._currentVelocity = np.zeros(3)
        self._currentOrientation = np.zeros(4)
        self._currentTargetPosition = np.zeros(3)
        self._currentTargetVelocity = np.zeros(3)

        self._debugging = False
        self._debug = False
        self._timeDebugging = False

    def run(self):
        print("[DroneControl] Initialized.")
        iteration = 0

        nextTime = time.perf_counter()
        while not self._shutdownEvent.is_set():
            tStart = time.time()
            self.getControlMessage()
            self.updateCurrentState()

            self.updateCoordinatesController()
            self.updateHeightController()
            self.updateStabilityController()

            self.combineControllersOutputs()
            self.checkReachedReferences()

            iteration += 1
            self._debug = self._debugging and iteration % 300 == 0

            # Sincronización precisa
            nextTime += self._timestep
            sleepDuration = nextTime - time.perf_counter()
            if sleepDuration > 0:
                time.sleep(sleepDuration)
            else:
                # Si el procesamiento tarda más de 1 ms, ajusta el siguiente ciclo
                nextTime = time.perf_counter()

            if self._timeDebugging:
                dtReal = time.time() - tStart
                dtSim = self._currentTimestamp - self._previousTimestamp
                print(f"[DroneControl] dtSim: {dtSim}, dtReal: {dtReal}")

        print("[DroneControl] Shutting down.")

    def stop(self):
        print("[DroneControl] Shutting down.")
        self._heightController.stop()
        self._heightController.join()
        self._stabilityController.stop()
        self._stabilityController.join()
        self._coordinatesController.stop()
        self._coordinatesController.join()
        self._shutdownEvent.set()

    def getControlMessage(self):
        # self._taskManagerManeuverEvent.wait(timeout=0.001)
        if self._taskManagerManeuverEvent.is_set():
            self._controlMessage = self._previousControlMessage
        else:
            try:
                self._controlMessage = self._droneControlQueue.get(timeout=0.001)
                self._previousControlMessage = self._controlMessage
                if self._debug:
                    print("Control Message received")
            except queue.Empty:
                self._controlMessage = self._previousControlMessage

    def updateCurrentState(self):
        self._updateEvent.wait()
        self._updateEvent.clear()
        if self._sensorData is not None:
            with self._sensorLock:
                sensorData = self._sensorData
            self._previousTimestamp = self._currentTimestamp
            self._currentTimestamp = sensorData.timestamp
            self._timestep = sensorData.timestep
            self._currentPosition = np.array(sensorData.dronePosition)
            self._currentAcceleration = np.array(sensorData.sensorValues["body_linacc"])
            self._currentAngularVelocity = np.array(
                sensorData.sensorValues["body_gyro"]
            )
            self._currentOrientation = np.array(sensorData.sensorValues["body_quat"])
            self._currentCOM = np.array(sensorData.sensorValues["drone_load_com"])
            self._currentVelocity = np.array(sensorData.sensorValues["body_vel"])
            self._currentTargetPosition = np.array(
                [
                    self._controlMessage.referenceX,
                    self._controlMessage.referenceY,
                    self._controlMessage.referenceZ,
                ]
            )
            if self._currentTargetPosition[0] is None:
                self._currentTargetPosition[0] = self._currentPosition[0]
            if self._currentTargetPosition[1] is None:
                self._currentTargetPosition[1] = self._currentPosition[1]
            if self._currentTargetPosition[2] is None:
                self._currentTargetPosition[2] = self._currentPosition[2]
        else:
            pass

    def updateHeightController(self):
        if self._heightControllerUpdateEvent.is_set():
            return
        heightData = {
            "timestamp": self._currentTimestamp,
            "currentHeight": self._currentPosition[2],
            "heightReference": self._controlMessage.referenceZ,
            "currentVelocity": self._currentVelocity.copy(),
        }
        self._heightController.setHeightData(heightData)
        self._heightControllerUpdateEvent.set()

    def updateStabilityController(self):
        if self._stabilityUpdateEvent.is_set():
            return
        with self._stabilityLock:
            self._stabilityData["timestamp"] = self._currentTimestamp
            self._stabilityData["timestep"] = self._timestep
            self._stabilityData["currentOrientation"] = self._currentOrientation
            self._stabilityData["orientationReference"] = (
                self._coordinatesController.getAttitudeReference().copy()
            )
            self._stabilityData["currentAngularVelocity"] = (
                self._currentAngularVelocity.copy()
            )
        self._stabilityUpdateEvent.set()

    def updateCoordinatesController(self):
        if self._coordinatesUpdateEvent.is_set():
            return
        coordinatesData = {}
        coordinatesData["timestamp"] = self._currentTimestamp
        coordinatesData["timestep"] = self._timestep
        coordinatesData["currentPosition"] = self._currentPosition.copy()
        coordinatesData["positionReference"] = self._currentTargetPosition.copy()
        coordinatesData["currentVelocity"] = self._currentVelocity.copy()
        coordinatesData["currentTargetVelocity"] = [1.0, 1.0, 1.0]
        coordinatesData["currentAcceleration"] = self._currentAcceleration.copy()
        coordinatesData["currentOrientation"] = self._currentOrientation.copy()

        self._coordinatesController.setCoordinatesData(coordinatesData)
        self._coordinatesUpdateEvent.set()

    def combineControllersOutputs(self):
        baseThrust = self._heightController.getRotorForce()
        stabilityRotorAdjustments = self._stabilityController.getRotorAdjustments()
        if self._debug:
            print(f"[DroneControl] Base Thrust: {baseThrust}")
            print(f"[DroneControl] Stability Adjustments: {stabilityRotorAdjustments}")

        MAX_CONTROL_INPUT = 100
        MIN_CONTROL_INPUT = 0

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

        droneThrusters = {
            "propeller-1_motor": thrust1,  # Front-left - CW
            "propeller-2_motor": thrust2,  # Front-right - CCW
            "propeller-3_motor": thrust3,  # Back-right - CW
            "propeller-4_motor": thrust4,  # Back-left - CCW
        }

        with self._controlLock:
            self._controlAction.droneThrusters = droneThrusters

    def checkReachedReferences(self):
        heightReferenceReached = self._heightController.getReachedReference()
        positionReferenceReached = self._coordinatesController.getReachedReference()

        self._reachedReferences = heightReferenceReached and positionReferenceReached
        if self._reachedReferences:
            if self._debug:
                print(
                    "[DroneControl] Height reference and position reference have been reached."
                )
            self._taskManagerManeuverEvent.clear()

        if self._debug:
            print(
                f"[DroneControl] Position reference reached: {positionReferenceReached}"
            )
            print(f"[DroneControl] Height reference reached: {heightReferenceReached}")

    # Accessed from MujocoSimulation thread.
    def setSensorData(self, sensorData: SensorData):
        with self._sensorLock:
            self._sensorData = sensorData

    def getControlInputs(self):
        with self._controlLock:
            return self._controlAction

    def setUpdateEvent(self):
        self._updateEvent.set()

    def setHeightReference(self, heightReference):
        self._previousControlMessage.referenceZ = heightReference
