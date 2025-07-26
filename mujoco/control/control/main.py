# main.py
import threading
import tkinter as tk
import queue
from src.MujocoSimulation import MujocoSimulation
from src.GroundControlStation import GCS
from src.TaskManager import TaskManager
from src.drone.DroneControl import DroneControl

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
    else:
        model = "drone"
        print("The used model by default will be 'drone'.")

    shutdownEvent = threading.Event()
    maneuverEvent = threading.Event()
    recvControlEvent = threading.Event()

    controlQueue = queue.Queue()  # sends to MuJoCo Simulation Thread
    droneControlQueue = queue.Queue()  # receives from TaskManager Thread

    # Instantiate modules
    # swayControl = SwayControl(pubsub, shutdownEvent)

    # Initialize DroneControl in a separate thread.
    sendControlMessageEvent = threading.Event()
    droneController = DroneControl(
        shutdownEvent,
        maneuverEvent,
        recvControlEvent,
        controlQueue,
        droneControlQueue,
    )

    # Initialize Task Manager in a separate thread.
    taskManager = TaskManager(shutdownEvent, maneuverEvent, droneControlQueue)
    taskManager.start()

    # Initialize GroundControlStation in the main thread. Tkinter isn't thread safe.
    root = tk.Tk()
    gcs = GCS(root, taskManager, shutdownEvent)

    mujocoSim = MujocoSimulation(
        controlQueue,
        recvControlEvent,
        shutdownEvent,
        model,
        droneController,
    )
    mujocoSim.start()
    droneController.start()

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
    mujocoSim.stop()
    mujocoSim.join()


if __name__ == "__main__":
    main()
