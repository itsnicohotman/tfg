import time
import queue
import threading
from src.manipulator.robotControllers import RobotControl
from include.Messages import RobotSensorData, RobotControlAction
import numpy as np


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

    def run(self):
        print("[SwayControl] Initialized.")
        iteration = 0

        nextTime = time.perf_counter()
        while not self._shutdownEvent.is_set():
            time.sleep(0.001)  # Sleep to prevent busy waiting
            # tStart = time.time()
            # self.updateCurrentState()

            # iteration += 1
            # self._debug = self._debugging and iteration % 300 == 0

            # # Sincronización precisa
            # nextTime += self._timestep
            # sleepDuration = nextTime - time.perf_counter()
            # if sleepDuration > 0:
            #     time.sleep(sleepDuration)
            # else:
            #     # Si el procesamiento tarda más de 1 ms, ajusta el siguiente ciclo
            #     nextTime = time.perf_counter()

            # if self._timeDebugging:
            #     dtReal = time.time() - tStart
            #     dtSim = self._currentTimestamp - self._previousTimestamp
            #     print(f"[SwayControl] dtSim: {dtSim}, dtReal: {dtReal}")

        print("[SwayControl] Shutting down.")

    def stop(self):
        print("[SwayControl] Shutting down.")
        # self._updateEvent.set()
        # self._heightController.stop()
        # self._heightController.join()
        # self._stabilityController.stop()
        # self._stabilityController.join()
        # self._coordinatesController.stop()
        # self._coordinatesController.join()
        # self._shutdownEvent.set()

    def updateCurrentState(self):
        # self._updateEvent.wait()
        # self._updateEvent.clear()
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

    # Accessed from MujocoSimulation thread.
    def setSensorData(self, sensorData: RobotSensorData):
        with self._sensorLock:
            self._sensorData = sensorData

    def getControlInputs(self):
        with self._controlLock:
            return self._robotControlAction
