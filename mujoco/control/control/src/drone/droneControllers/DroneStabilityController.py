import threading
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import List
import os
import time


class DroneStabilityController(threading.Thread):
    """Controller for drone's attitude: roll, pitch and yaw. Attitude controller at 100Hz."""

    def __init__(
        self,
        shutdownEvent: threading.Event,
        stabilityOrientationUpdateEvent: threading.Event,
        stabilityLock: threading.Lock,
        stabilityData: dict,
        stabilityActionLock: threading.Lock,
    ):
        super().__init__()
        self._shutdownEvent = shutdownEvent
        self._stabilityOrientationUpdateEvent = stabilityOrientationUpdateEvent
        self._stabilityActionLock = stabilityActionLock
        self._stabilityLock = stabilityLock
        self._stabilityData = stabilityData
        self._currentTimestamp = 0.0
        self._timestep = 0.001

        # Angle -> Rate P Controller
        self._pKp = np.array(
            [10, 10, 15]
        )  # [roll, pitch, yaw] attitude controller gain

        # Rate --> thrust PID Controller
        self._pidKp = np.array([20, 20, 20])  # [roll, pitch, yaw] rate controller gains
        self._pidKi = np.array([30, 30, 30])
        self._pidKd = np.array([0, 0, 0])

        # Attitude arrays
        self._currentOrientation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion
        self._currentAngularVelocity = np.zeros(3)  # Gyroscope
        # Normal range roll and pitch: [-0.52, 0.52] rad = ~[-30, 30] degrees
        # Normal range yaw: [-3.1416, 3.1416] rad = ~[-180, 180] degrees

        self._currentTargetAngles = np.zeros(3)  # Roll, pitch, yaw (rad)

        # Outputs
        self._rotorAdjustments = [0.0, 0.0, 0.0, 0.0]  # Adjustments for each rotor

        self._historyReferenceAngles = []
        self._historyCurrentAngles = []
        self._historyControlAction = []
        self._historyRotorAdjustments = []
        self._historyAngleError = []
        self._historyRateError = []
        self._historyTargetRate = []
        self._historyRate = []
        self._historyTime = []

        self._debugging = False  # Set to False to disable debugging output

    def run(self):
        print("[DroneStabilityController] Initialized.")
        previousRateError = np.zeros(3)
        integralRateError = np.zeros(3)
        iteration = 0
        dtPID = 0.001
        dtP = 0.004
        self._historyStartTime = time.time()

        while not self._shutdownEvent.is_set():
            debug = self._debugging and iteration % 100 == 0
            self.updateOrientation()

            if iteration == 1 or iteration % 4 == 0:
                # Attitude controller: P controller whose output is the target rate.
                # Will execute with a frequency of f = 250 Hz (given that timestep = 0.001). Therefore, dtP = 0.004.

                previousTimeP = self._currentTimestamp
                # Convert quaternion to Euler angles (roll, pitch, yaw) - in radians
                currentAngles = np.array(
                    self.quaternionToEuler(self._currentOrientation)
                )
                angleError = self._currentTargetAngles - currentAngles
                targetRate = self._pKp * angleError
            if debug:
                print(
                    f"[DroneStabilityController] iteration: {iteration}, current timestamp: {self._currentTimestamp}"
                )
                print(
                    f"[DroneStabilityController] currentAngles: {currentAngles}, current target angles: {self._currentTargetAngles},angleError: {angleError}"
                )
                print(
                    f"[DroneStabilityController] currentAngularVelocity: {self._currentAngularVelocity}, targetRate: {targetRate}"
                )

            # Rate controller: PID controller whose output is the torques to be applied.
            # Will execute with a frequency of f = 1000 Hz (given that timestep = 0.001). Therefore, dtPID = 0.001.

            rateError = targetRate - self._currentAngularVelocity
            derivativeError = (rateError - previousRateError) / dtPID
            integralRateError += rateError * dtPID
            previousRateError = rateError
            controlOutput = (
                self._pidKp * rateError
                + self._pidKi * integralRateError
                + self._pidKd * derivativeError
            )
            u_roll, u_pitch, u_yaw = controlOutput

            # Mixer for rotor forces based on roll/pitch/yaw adjustments
            rotorAdjustments = [
                -u_pitch - u_roll - u_yaw,  # Rotor 1: front-left - CW
                -u_pitch + u_roll + u_yaw,  # Rotor 2: front-right - CCW
                +u_pitch + u_roll - u_yaw,  # Rotor 3: back-right - CW
                +u_pitch - u_roll + u_yaw,  # Rotor 4: back-left - CCW
            ]
            if debug:
                print(f"[DroneStabilityController] rate error: {rateError}")
                print(
                    f"[DroneStabilityController] [u_roll, u_pitch, u_yaw]: {controlOutput}"
                )
                print(
                    f"[DroneStabilityController] rotorAdjustments: {rotorAdjustments}"
                )

            with self._stabilityActionLock:
                self._rotorAdjustments = rotorAdjustments

            # Store for analysis
            self._historyTime.append(self._currentTimestamp)
            self._historyAngleError.append(angleError.tolist())
            self._historyRateError.append(rateError.tolist())
            self._historyReferenceAngles.append(self._currentTargetAngles.tolist())
            self._historyCurrentAngles.append(currentAngles.tolist())
            self._historyRotorAdjustments.append(rotorAdjustments.copy())
            self._historyTargetRate.append(targetRate.tolist())
            self._historyRate.append(self._currentAngularVelocity.tolist())

            iteration += 1

        self._historyEndTime = time.time()
        self.graphErrors()

    def updateOrientation(self):
        self._stabilityOrientationUpdateEvent.wait()  # Sets thread frequency to 1000 Hz
        with self._stabilityLock:
            self._currentTimestamp = self._stabilityData["timestamp"]
            self._timestep = self._stabilityData["timestep"]
            self._currentOrientation = np.array(
                self._stabilityData["currentOrientation"]
            )
            self._currentTargetAngles = np.array(
                self._stabilityData["orientationReference"]
            )
            self._currentAngularVelocity = np.array(
                self._stabilityData["currentAngularVelocity"]
            )
        self._stabilityOrientationUpdateEvent.clear()

    def getRotorAdjustments(self):
        """Get the rotor adjustments to be applied to the drone."""
        with self._stabilityActionLock:
            rotorAdjustments = self._rotorAdjustments.copy()
        return rotorAdjustments

    def stop(self):
        self._shutdownEvent.set()
        self._stabilityOrientationUpdateEvent.set()
        self._stabilityOrientationUpdateEvent.set()
        print("[DroneStabilityController] Shutting Down.")

    def graphErrors(self):

        output_dir = "/home/nicolas/tfg/Memoria/GraficasAngulos"
        os.makedirs(output_dir, exist_ok=True)
        matplotlib.use("Agg")

        time_sim = np.array(self._historyTime)
        # Scaling: real time / simulated time
        real_duration = self._historyTime[-1] - self._historyTime[0]
        wall_clock_duration = self._historyEndTime - self._historyStartTime
        scale_factor = wall_clock_duration / real_duration if real_duration > 0 else 1.0
        time = (time_sim - time_sim[0]) * scale_factor

        if not self._historyTime or not self._historyRotorAdjustments:
            return

        rateErrors = np.array(self._historyRateError)
        angleErrors = np.array(self._historyAngleError)
        targetRate = np.array(self._historyTargetRate)
        rate = np.array(self._historyRate)
        labels = ["Roll", "Pitch", "Yaw"]

        # --- Angle Error Plot ---
        plt.figure()
        plt.clf()
        for i in range(3):
            plt.plot(time, angleErrors[:, i], label=labels[i])
        plt.title("Attitude Angle Control Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [rad]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "AttitudeAngleControl_Errors.png"))
        plt.close()
        print("[Drone Stability Controller] AttitudeAngleControl_Errors.png saved")

        # --- Rate Error Plot ---
        plt.figure()
        plt.clf()
        for i in range(3):
            plt.plot(time, rateErrors[:, i], label=labels[i])
        plt.title("Attitude Rate Control Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [rad/s]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "AttitudeRateControl_Errors.png"))
        plt.close()
        print("[Drone Stability Controller] AttitudeRateControl_Errors.png saved")

        reference_angles = np.array(self._historyReferenceAngles)
        current_angles = np.array(self._historyCurrentAngles)

        # --- Reference vs Current Angles Plot ---
        for i in range(3):
            plt.figure()
            plt.clf()
            plt.plot(time, reference_angles[:, i], "--", label=f"Ref {labels[i]}")
            plt.plot(time, current_angles[:, i], label=f"Actual {labels[i]}")
            plt.title(f"{labels[i]} Control")
            plt.xlabel("Time [s]")
            plt.ylabel("Angle [rad]")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{labels[i]}Control.png"))
            plt.close()
            print(f"[Drone Stability Controller] {labels[i]}Control.png saved")

        # --- Reference vs Current Rate Plot ---
        for i in range(3):
            plt.figure()
            plt.clf()
            plt.plot(time, targetRate[:, i], "--", label=f"{labels[i]} Reference Rate")
            plt.plot(time, rate[:, i], label=f"Actual {labels[i]} Rate")
            plt.title(f"{labels[i]} Rate Control")
            plt.xlabel("Time [s]")
            plt.ylabel("Angle [rad/s]")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{labels[i]}_Rate_Control.png"))
            plt.close()
            print(f"[Drone Stability Controller] {labels[i]}_Rate_Control.png saved")

        roll_output = [v[0] for v in self._historyRotorAdjustments]
        pitch_output = [v[1] for v in self._historyRotorAdjustments]
        yaw_output = [v[2] for v in self._historyRotorAdjustments]

        # --- Control Output Plot ---
        plt.figure()
        plt.clf()
        plt.plot(time, roll_output, label="Roll Output")
        plt.plot(time, pitch_output, label="Pitch Output")
        plt.plot(time, yaw_output, label="Yaw Output")
        plt.xlabel("Time [s]")
        plt.ylabel("Control Output")
        plt.title("Control Output vs Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ControlOutput.png"))
        plt.close()
        print("[Drone Stability Controller] ControlOutput.png saved")

    @staticmethod
    def quaternionToEuler(q):
        # Convert quaternion to roll, pitch, yaw
        w, x, y, z = q
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return roll, pitch, yaw
