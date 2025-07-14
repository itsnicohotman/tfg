import threading
import matplotlib.pyplot as plt
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
        self.MAX_CONTROL_INPUT = 30
        self.MIN_CONTROL_INPUT = 0
        self._rotorForce = 0.0  # Base thrust to be applied to the rotors

        # AHORA HAY QUE PROBAR LA SIMULACIÓN EMPEZANDO DESDE EL SUELO Y SUBIENDO A 3 METROS CON GCS. DESPUÉS PROBAR A IR A UNA POSICIÓN Y MANTENERLA.

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

    def run(self):
        rotorForce = 0.0
        integralVelocityError = 0.0
        derivativeVelocityError = 0.0
        previousVelocityError = 0.0
        previousTimestamp = 0.0
        print("[DroneHeightController] Initialized.")
        iteration = 0
        gravity = 9.81
        mass = 4.5  # Mass of the drone in kg
        MAX_VEL = 2.0  # Maximum ascension velocity in m/s

        while not self._shutdownEvent.is_set():
            iteration += 1

            self.updateHeightController()
            """ if iteration % 15 == 0:
                print("[DroneHeightController] Height data updated.") """
            dt = self._currentTimestamp - previousTimestamp
            previousTimestamp = self._currentTimestamp

            if iteration == 1 or iteration % 10 == 0:
                # Height controller: P controller whose output is the target ascension velocity.
                # Will execute with a frequency of f = 100 Hz (given that timestep = 0.001)

                heightError = self._heightReference - self._currentHeight
                targetVelocity = np.clip(self._pKp * heightError, -MAX_VEL, MAX_VEL)

            if iteration == 1 or iteration % 2 == 0:
                # Velocity controller: PID controller whose output is the base thrust to be applied.
                # Will execute with a frequency of f = 500 Hz (given that timestep = 0.001)

                velocityError = targetVelocity - self._currentVelocity[2]
                derivativeVelocityError = (velocityError - previousVelocityError) / dt
                integralVelocityError += velocityError * dt
                previousVelocityError = velocityError

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
                integralVelocityError += velocityError * dt

            reachedReference = self.checkReachedReferences(heightError, velocityError)
            """ if iteration % 100 == 0:
                print(f"[DroneHeightController] reachedReference: {reachedReference}") """

            with self._heightResultsLock:
                self._rotorForce = rotorForce
                self._reachedReference = reachedReference

            if iteration % 100 == 0 and reachedReference:
                print("[DroneHeightController] Height reference reached and stable.")

            # Store for analysis
            self._historyHeightReference.append(self._heightReference)
            self._historyCurrentHeight.append(self._currentHeight)
            self._historyRotorForce.append(rotorForce)
            self._historyHeightError.append(heightError)
            self._historyVelocityError.append(velocityError)
            self._historyTime.append(self._currentTimestamp)
            self._historyVelocity.append(self._currentVelocity[2])
            self._historyVelocityRef.append(targetVelocity)

        self.graphControlProcess()

    def stop(self):
        self._shutdownEvent.set()
        print("[DroneHeightController] Shutting Down.")
        pass

    def setHeightData(self, heightData):
        with self._heightUpdateLock:
            self._heightData = heightData

    def updateHeightController(self):
        """Update the height controller with the latest height data."""
        self._heightControllerUpdateEvent.wait()
        self._currentTimestamp = self._heightData["timestamp"]
        self._currentHeight = self._heightData["currentHeight"]
        self._currentVelocity = self._heightData["currentVelocity"].copy()
        self._heightReference = self._heightData["heightReference"]

        if self._heightReference < 0.45:
            self._heightReference = 0.45

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
                    """ print(
                        "[DroneHeightController] Height reference reached and stable."
                    ) """
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

        # Height graph
        time = np.array(self._historyTime)
        historyHeight = np.array(self._historyCurrentHeight)
        historyHeightReference = np.array(self._historyHeightReference)
        historyVelocity = np.array(self._historyVelocity)
        historyVelocityReference = np.array(self._historyVelocityRef)
        historyHeightError = np.array(self._historyHeightError)
        historyRotorForce = np.array(self._historyRotorForce)

        # Height error graph - dummy
        plt.figure()
        plt.plot(time, historyHeightError, label="Error")
        plt.title("Height Control - Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Height [m]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "HeightControl_Error_Dummy.png"))
        print("[Drone Height Controller] HeightControl_Error_Dummy.png saved")

        # Height error graph
        plt.figure()
        plt.plot(time, historyHeightError, label="Error")
        plt.title("Height Control - Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Height [m]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "HeightControl_Error.png"))
        print("[Drone Height Controller] HeightControl_Error.png saved")

        # Force graph
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
