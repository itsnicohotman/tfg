import threading
import queue
import time
from include.Messages import Command, ControlMessage
from src.drone.DroneControl import DroneControl

class TaskManager(threading.Thread):
    def __init__(self, shutdownEvent:threading.Event, maneuverEvent:threading.Event, droneControlQueue:queue.Queue):
        super().__init__()
        self._gcsMessageQueue = queue.Queue()
        self._maneuverEvent = maneuverEvent   # Used to block TaskManager and incoming commands while executing a command, and to let know DroneControl thread of a new Control Message.
        self._gcsMessage: Command
        self._droneControlMessage = ControlMessage(grasp = False, referenceX = None, referenceY = None, referenceZ = None)
        self._lock = threading.Lock()
        self._shutdownEvent = shutdownEvent
        self._droneControlQueue = droneControlQueue

    def executeCommand(self):
        print(f"[TaskManager] Executing: {self._gcsMessage.command}")
        if (self._gcsMessage.command == "take-off"):
            self.takeOff(self._gcsMessage.coordinates[2])
        elif (self._gcsMessage.command == "land"):
            self.land()
        elif (self._gcsMessage.command == "go to"):
            self.goTo(self._gcsMessage.coordinates)
        elif (self._gcsMessage.command == "grasp"):
            self.grasp()
        elif (self._gcsMessage.command == "release"):
            self.release()
        else:
            print(f"{self._gcsMessage.command} is not a valid command.")
        

    def takeOff(self, height):
        if not self._maneuverEvent.is_set():
            self._droneControlMessage.referenceZ = height
            self.setDroneControlMessage()
            while self._maneuverEvent.is_set():
                time.sleep(1.5)
                print(f"[TaskManager] Taking off to a height of {height}m.")
            print("[TaskManager] Take-off completed.")
        else:
            print("[TaskManager] Busy. In the middle of executing another command.")

    def land(self):
        if not self._maneuverEvent.is_set():
            self._droneControlMessage.referenceZ = 0
            self.setDroneControlMessage()
            while self._maneuverEvent.is_set():
                time.sleep(0.5)
                print(f"[TaskManager] Landing")
            print("[TaskManager] Landing completed.")
        else:
            print("[TaskManager] Busy. In the middle of executing another command.")

    def goTo(self, coordinates):
        if not self._maneuverEvent.is_set():
            self._droneControlMessage.referenceX = coordinates[0]
            self._droneControlMessage.referenceY = coordinates[1]
            self._droneControlMessage.referenceZ = coordinates[2]
            self.setDroneControlMessage()
            while self._maneuverEvent.is_set():
                time.sleep(0.5)
                print(f"[TaskManager] Going to {coordinates}")
            print(f"[TaskManager] {coordinates} reached.")
        else:
            print("[TaskManager] Busy. In the middle of executing another command.")

    def grasp(self):
        if not self._maneuverEvent.is_set():
            self._droneControlMessage.grasp = True
            self.setDroneControlMessage()
            while self._maneuverEvent.is_set():
                time.sleep(0.5)
                print(f"[TaskManager] Grasping object.")
            print("[TaskManager] Object grasped.")
        else:
            print("[TaskManager] Busy. In the middle of executing another command.")

    def release(self):
        if not self._maneuverEvent.is_set():
            self._droneControlMessage.grasp = False
            self.setDroneControlMessage()
            while self._maneuverEvent.is_set():
                time.sleep(0.5)
                print(f"[TaskManager] Releasing object.")
            print("[TaskManager] Object released.")
        else:
            print("[TaskManager] Busy. In the middle of executing another command.")

    def getMessage(self):
        msg = self._gcsMessageQueue.get(timeout=0.5)
        return msg

    def setDroneControlMessage(self):
        self._droneControlQueue.put(self._droneControlMessage)
        self._maneuverEvent.set()

    def run(self):
        print("[TaskManager] Initialized.")
        while not self._shutdownEvent.is_set():
            if  not self._maneuverEvent.is_set():
                try:
                    self._gcsMessage = self.getMessage()
                    self.executeCommand()
                except queue.Empty:
                    continue
            else:
                try:
                    self._gcsMessageQueue.get_nowait()
                except queue.Empty:
                    continue
        
        print("[TaskManager] Shutting down.")
        
    def stop(self):
        print("[TaskManager] Shutting down.")
        self._shutdownEvent.set()