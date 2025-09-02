# main.py
import threading
import tkinter as tk
import queue
from src.MujocoSimulation import MujocoSimulation
from src.GroundControlStation import GCS
from src.TaskManager import TaskManager
from src.drone.DroneControl import DroneControl
from src.manipulator.SwayControl import SwayControl

# from modules.sway_control import SwayControl
import time
import warnings
from glfw import GLFWError
import sys


def main():
    warnings.filterwarnings("ignore", category=GLFWError)

    # Verifies that there's an argument when executing the program.

    if len(sys.argv) > 1:
        model = sys.argv[1]
        if len(sys.argv) > 2:
            swayControllerEnabled = sys.argv[2].lower() in ["true", "1", "yes"]
        else:
            swayControllerEnabled = False

        if len(sys.argv) > 3:
            try:
                combination = int(sys.argv[3])
                if not (1 <= combination <= 7):
                    raise ValueError
            except ValueError:
                print(
                    "❌ Error: The third argument must be an integer number between 1 and 7."
                )
                sys.exit(1)
        else:
            combination = 0
    else:
        model = "drone"
        swayControllerEnabled = False
        print("The used model by default will be 'drone'.")

    shutdownEvent = threading.Event()
    maneuverEvent = threading.Event()
    droneControlQueue = queue.Queue()  # receives from TaskManager Thread

    # Instantiate modules
    swayController = SwayControl(shutdownEvent)

    # Initialize DroneControl in a separate thread.
    droneController = DroneControl(
        shutdownEvent,
        maneuverEvent,
        droneControlQueue,
    )

    # Initialize Task Manager in a separate thread.
    taskManager = TaskManager(shutdownEvent, maneuverEvent, droneControlQueue)
    taskManager.start()

    # Initialize GroundControlStation in the main thread. Tkinter isn't thread safe.
    root = tk.Tk()
    gcs = GCS(root, taskManager, shutdownEvent)

    mujocoSim = MujocoSimulation(
        shutdownEvent,
        model,
        droneController,
        swayController=swayController,
        swayControllerEnabled=swayControllerEnabled,
        qCombination=combination,
    )
    mujocoSim.start()
    droneController.start()
    swayController.start()

    try:
        gcs.checkShutdown()
        print("[main] GCS initialized.")
        root.mainloop()
    except KeyboardInterrupt:
        shutdownEvent.set()
        print("[main] Tkinter mainloop ended.")
        print("[main] GCS shutting down.")

    # When GCS' GUI closes, all other threads finish too
    taskManager.stop()
    taskManager.join()
    droneController.stop()
    droneController.join()
    swayController.stop()
    swayController.join()
    mujocoSim.stop()
    mujocoSim.join()


if __name__ == "__main__":
    main()
