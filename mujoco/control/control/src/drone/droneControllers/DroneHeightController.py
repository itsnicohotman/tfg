import threading
import matplotlib.pyplot as plt
import os
import numpy as np


class DroneHeightController(threading.Thread):
    """Controller that controls the drone's height supposing an axial and stable system."""

    def __init__(
        self,
        shutdownEvent: threading.Event,
        heightControllerUpdateEvent: threading.Event,
        heightControllerResultsEvent: threading.Event,
        pidKp,
        pidKi,
        pidKd,
        pKp,
        heightData: dict,
        heightLock: threading.Lock,
        heightActionLock: threading.Lock,
    ):
        super().__init__()
        self._shutdownEvent = shutdownEvent
        self._heightControllerUpdateEvent = heightControllerUpdateEvent
        self._heightControllerResultsEvent = heightControllerResultsEvent
        self._heightData = heightData
        self._heightLock = heightLock
        self._heightActionLock = heightActionLock

        # Controller
        self._pidKp = pidKp
        self._pidKi = pidKi
        self._pidKd = pidKd
        self._pKp = pKp
        self.MAX_CONTROL_INPUT = 30
        self.MIN_CONTROL_INPUT = 0
        self._rotorForce = 0.0

        self._currentTimestamp = 0.0
        self._timestep = 0.001
        self._currentHeight = 0.0
        # Drone's z offset so that the landing gears rest on the floor = 0.45.
        self._heightReference = 0.45

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

        while not self._shutdownEvent.is_set():
            iteration += 1

            self.updateHeightController()
            dt = self._currentTimestamp - previousTimestamp
            previousTimestamp = self._currentTimestamp

            if iteration == 1 or iteration % 15 == 0:
                # Height controller: P controller whose output is the target ascension velocity.
                # Will execute with a frequency of f = 100 Hz (given that timestep = 0.001)

                heightError = self._heightReference - self._currentHeight
                targetVelocity = self._pKp * heightError

            if iteration == 1 or iteration % 2 == 0:
                # Velocity controller: PID controller whose output is the base thrust to be applied.
                # Will execute with a frequency of f = 500 Hz (given that timestep = 0.001)

                velocityError = targetVelocity - self._currentVelocity[2]
                derivativeVelocityError = (velocityError - previousVelocityError) / dt
                integralVelocityError += velocityError * dt
                previousVelocityError = velocityError

                totalForce = (
                    self._pidKp * velocityError
                    + self._pidKi * integralVelocityError
                    + self._pidKd * derivativeVelocityError
                )
            rotorForce = totalForce
            if rotorForce > self.MAX_CONTROL_INPUT:
                rotorForce = self.MAX_CONTROL_INPUT
                integralVelocityError = integralVelocityError
            elif rotorForce < self.MIN_CONTROL_INPUT:
                rotorForce = self.MIN_CONTROL_INPUT
            else:
                integralVelocityError += velocityError * dt

            # Send base thrust per rotor
            self.sendHeightControllerResults(rotorForce)

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

    def updateHeightController(self):
        self._heightControllerUpdateEvent.wait()
        with self._heightLock:
            self._currentTimestamp = self._heightData["timestamp"]
            self._currentHeight = self._heightData["currentHeight"]
            self._heightReference = self._heightData["heightReference"]
            self._currentVelocity = self._heightData["currentVelocity"].copy()
            if self._heightReference < 0.45:
                self._heightReference = 0.45
        self._heightControllerUpdateEvent.clear()

    def sendHeightControllerResults(self, rotorForce):
        while self._heightControllerResultsEvent.is_set():
            continue
        with self._heightActionLock:
            self._rotorForce = rotorForce
            print(f"Sending rotorForce (locked): {self._rotorForce}")
            self._heightControllerResultsEvent.set()

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
