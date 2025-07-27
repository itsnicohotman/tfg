import threading
import matplotlib
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import time


class DroneHeightController(threading.Thread):
    """Controller that controls the drone's height supposing an axial and stable system."""

    def __init__(
        self,
        shutdownEvent: threading.Event,
        heightControllerUpdateEvent: threading.Event,
    ):
        super().__init__()
        self._shutdownEvent = shutdownEvent
        self._heightControllerUpdateEvent = heightControllerUpdateEvent
        self._heightUpdateLock = threading.Lock()
        self._heightResultsLock = threading.Lock()

        # Controller
        self._pidKp = 25.0
        self._pidKi = 30.0
        self._pidKd = 0.0
        self._pKp = 3.35
        self.MAX_CONTROL_INPUT = 100
        self.MIN_CONTROL_INPUT = 0
        self._rotorForce = 0.0  # Base thrust to be applied to the rotors

        # Check if the height reference has been reached
        self._stabilityTime = (
            0.5  # Time in seconds to consider the height reference as reached
        )
        self._errorTolerance = 0.05  # Tolerance for height and velocity errors
        self._previousHeightReference = 0.0
        self._heightReferenceChanged = False
        self._reachedReference = False
        self._inToleranceSince = None

        self._currentTimestamp = 0.0
        self._timestep = 0.001
        self._currentHeight = 0.0
        # Drone's z offset so that the landing gears rest on the floor = 0.45.
        self._heightReference = 0.45
        self._heightData = {
            "timestamp": 0.0,
            "currentHeight": 0.0,
            "heightReference": 0.45,
            "currentVelocity": [0.0, 0.0, 0.0],
        }

        # History vectors for plotting
        self._historyHeightReference = []
        self._historyCurrentHeight = []
        self._historyControlAction = []
        self._historyRotorForce = []
        self._historyHeightError = []
        self._historyVelocityError = []
        self._historyTime = []
        self._historyVelocity = []
        self._historyVelocityRef = []

        self._debugging = False
        self._timeDebugging = False

    def run(self):
        print("[DroneHeightController] Initialized.")
        iteration = 0
        gravity = 9.81
        mass = 4.5  # Mass of the drone in kg
        MAX_VEL = 2.0  # Maximum ascension velocity in m/s
        rotorForce = 0.0
        integralVelocityError = 0.0
        derivativeVelocityError = 0.0
        previousVelocityError = 0.0

        PID_PERIOD = 0.002
        P_PERIOD = 0.01

        self.updateHeightController()
        lastTimeP = self._currentTimestamp
        lastTimePID = self._currentTimestamp

        previousTimestampPID = 0.0
        previousTimestampP = 0.0
        dtPID = PID_PERIOD
        dtP = P_PERIOD

        while not self._shutdownEvent.is_set():
            debug = self._debugging and iteration % 200 == 0
            timeDebug = self._timeDebugging and iteration % 200 == 0

            self.updateHeightController()
            now = self._currentTimestamp

            if debug:
                print(
                    f"[DroneHeightController] Iteration: {iteration}, current timestamp: {self._currentTimestamp}"
                )
                print("[DroneHeightController] Height data updated.")

            if iteration == 0 or now - lastTimeP >= P_PERIOD:
                # Height controller: P controller whose output is the target ascension velocity.
                # Will execute with a frequency of f = 100 Hz (given that timestep = 0.001). Therefore, dtP = 0.01

                heightError = self._heightReference - self._currentHeight
                targetVelocity = np.clip(self._pKp * heightError, -MAX_VEL, MAX_VEL)
                dtP = self._currentTimestamp - previousTimestampP
                previousTimestampP = self._currentTimestamp
                lastTimeP += P_PERIOD

            if iteration == 0 or now - lastTimePID >= PID_PERIOD:
                # Velocity controller: PID controller whose output is the base thrust to be applied.
                # Will execute with a frequency of f = 500 Hz (given that timestep = 0.001). Therefore, dtPID = 0.002
                if dtPID == 0:
                    dtPID = PID_PERIOD

                velocityError = targetVelocity - self._currentVelocity[2]
                derivativeVelocityError = (
                    velocityError - previousVelocityError
                ) / dtPID
                integralVelocityError += velocityError * dtPID
                previousVelocityError = velocityError

                dtPID = self._currentTimestamp - previousTimestampPID
                previousTimestampPID = self._currentTimestamp
                lastTimePID += PID_PERIOD

                if timeDebug:
                    print(f"[DroneHeightController] dtPID: {dtPID}, dtP: {dtP}")

                rotorForce = (
                    self._pidKp * velocityError
                    + self._pidKi * integralVelocityError
                    + self._pidKd * derivativeVelocityError
                    + gravity * mass
                )

                if rotorForce > self.MAX_CONTROL_INPUT:
                    rotorForce = self.MAX_CONTROL_INPUT
                    integralVelocityError = integralVelocityError
                elif rotorForce < self.MIN_CONTROL_INPUT:
                    rotorForce = self.MIN_CONTROL_INPUT
                else:
                    integralVelocityError += velocityError * dtPID

            reachedReference = self.checkReachedReferences(heightError, velocityError)
            if debug:
                print(f"[DroneHeightController] reachedReference: {reachedReference}")
                if reachedReference:
                    print(
                        "[DroneHeightController] Height reference reached and stable."
                    )

            with self._heightResultsLock:
                self._rotorForce = rotorForce
                self._reachedReference = reachedReference

            # Store for analysis
            self._historyHeightReference.append(self._heightReference)
            self._historyCurrentHeight.append(self._currentHeight)
            self._historyRotorForce.append(rotorForce)
            self._historyHeightError.append(heightError)
            self._historyVelocityError.append(velocityError)
            self._historyTime.append(self._currentTimestamp)
            self._historyVelocity.append(self._currentVelocity[2])
            self._historyVelocityRef.append(targetVelocity)

            iteration += 1
        self.graphControlProcess()

    def stop(self):
        self._shutdownEvent.set()
        self._heightControllerUpdateEvent.set()
        print("[DroneHeightController] Shutting Down.")

    def setHeightData(self, heightData):
        with self._heightUpdateLock:
            self._heightData = heightData

    def updateHeightController(self):
        """Update the height controller with the latest height data."""
        self._heightControllerUpdateEvent.wait()  # Sets thread frequency to 1000 Hz
        with self._heightUpdateLock:
            self._currentTimestamp = self._heightData["timestamp"]
            self._currentHeight = self._heightData["currentHeight"]
            self._currentVelocity = self._heightData["currentVelocity"].copy()
            heightReference = self._heightData["heightReference"]

        if heightReference < 0.45:
            self._heightReference = 0.45
        else:
            self._heightReference = heightReference

        if self._heightReference != self._previousHeightReference:
            self._heightReferenceChanged = True
            self._previousHeightReference = self._heightReference
        self._heightControllerUpdateEvent.clear()

    def checkReachedReferences(self, heightError, velocityError):
        """Check if the height reference has been reached."""
        reachedReference = False

        if self._heightReferenceChanged:
            self._heightReferenceChanged = False
            reachedReference = False
            self._inToleranceSince = None
            print("[DroneHeightController] Height reference changed, resetting check.")
        else:
            if (
                abs(heightError) < self._errorTolerance
                and abs(velocityError) < self._errorTolerance
            ):
                if self._inToleranceSince is None:
                    self._inToleranceSince = time.time()
                elif (time.time() - self._inToleranceSince) >= self._stabilityTime:
                    reachedReference = True
            else:
                self._inToleranceSince = None
        return reachedReference

    def getRotorForce(self):
        with self._heightResultsLock:
            return self._rotorForce

    def getReachedReference(self):
        with self._heightResultsLock:
            return self._reachedReference

    def graphControlProcess(self):
        output_dir = "/home/nicolas/tfg/Memoria/GraficasAltura"
        os.makedirs(output_dir, exist_ok=True)
        matplotlib.use("Agg")

        time = np.array(self._historyTime)
        historyHeight = np.array(self._historyCurrentHeight)
        historyHeightReference = np.array(self._historyHeightReference)
        historyVelocity = np.array(self._historyVelocity)
        historyVelocityReference = np.array(self._historyVelocityRef)
        historyHeightError = np.array(self._historyHeightError)
        historyRotorForce = np.array(self._historyRotorForce)

        # --- Height graph (dummy) ---
        plt.figure()
        plt.clf()
        plt.plot(time, historyHeightError, label="Error")
        plt.title("Height Control - Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Height [m]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "HeightControl_Error_Dummy.png"))
        plt.close()
        print("[Drone Height Controller] HeightControl_Error_Dummy.png saved")

        # --- Height error ---
        plt.figure()
        plt.clf()
        plt.plot(time, historyHeightError, label="Error")
        plt.title("Height Control - Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Height [m]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "HeightControl_Error.png"))
        plt.close()
        print("[Drone Height Controller] HeightControl_Error.png saved")

        # --- Force ---
        plt.figure()
        plt.clf()
        plt.plot(time, historyRotorForce, label="Rotor Force")
        plt.title("Height Control - Rotor Force")
        plt.xlabel("Time [s]")
        plt.ylabel("Rotor Force [N]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "HeightControl_RotorForce.png"))
        plt.close()
        print("[Drone Height Controller] HeightControl_RotorForce.png saved")

        # --- Height and height reference ---
        plt.figure()
        plt.clf()
        plt.plot(time, historyHeight, label="Height")
        plt.plot(time, historyHeightReference, label="Height Reference")
        plt.title("Height Control - Reference & Actual Height")
        plt.xlabel("Time [s]")
        plt.ylabel("Height [m]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "HeightControl_Ref.png"))
        plt.close()
        print("[Drone Height Controller] HeightControl_Ref.png saved")

        # --- Velocity and velocity reference ---
        plt.figure()
        plt.clf()
        plt.plot(time, historyVelocity, label="Velocity")
        plt.plot(time, historyVelocityReference, label="Velocity Reference")
        plt.title("Velocity Control - Reference & Actual Velocity")
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "HeightControl_VelControl.png"))
        plt.close()
        print("[Drone Height Controller] HeightControl_VelControl.png saved")
