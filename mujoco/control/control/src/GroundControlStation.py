import threading
import tkinter as tk
from src.TaskManager import TaskManager
from include.Messages import Command

# Ground Control Station class
class GCS(): 
    def __init__(self, root: tk.Tk, taskManager: TaskManager, shutdownEvent:threading.Event):
        super().__init__()  # Start thread
        self._root = root
        self._root.title("Ground Control Station")
        self._message: Command
        self._taskManager = taskManager
        self._buildInterface()
        self._shutdownEvent = shutdownEvent
    
    def _buildInterface(self):
        # Text field to show commands.
        self._textDisplay = tk.Text(self._root, height=10, width=40)  
        self._textDisplay.pack(pady=10)                         

        # Validating only numbers
        validCmd = (self._root.register(self.checkOnlyNumbers), "%P")

        # X field
        tk.Label(self._root, text="X:").pack()
        self._entryX = tk.Entry(self._root, validate="key", validatecommand=validCmd, width=10)
        self._entryX.pack(pady=2)

        # Y field
        tk.Label(self._root, text="Y:").pack()
        self._entryY = tk.Entry(self._root, validate="key", validatecommand=validCmd, width=10)
        self._entryY.pack(pady=2)

        # Z field
        tk.Label(self._root, text="Z:").pack()
        self._entryZ = tk.Entry(self._root, validate="key", validatecommand=validCmd, width=10)
        self._entryZ.pack(pady=2)

        # Take-off button
        self._takeOffButton = tk.Button(self._root, text="Take-off", command=self.takeOffCallback)  # (graphic interface where the button should be placed, button's text, callback function)
        self._takeOffButton.pack(pady=5)                                                            # Adds button to graphic interface

        # Land button
        self._landButton = tk.Button(self._root, text="Land", command=self.landCallback)
        self._landButton.pack(pady=5)

        # GoTo button
        self._goToButton = tk.Button(self._root, text="Go To", command=self.goToCallback)
        self._goToButton.pack(pady=5)

        # Grasp button
        self._graspButton = tk.Button(self._root, text="Grasp", command=self.graspCallback)
        self._graspButton.pack(pady=5)

        # Release button
        self._releaseButton = tk.Button(self._root, text="Release", command=self.releaseCallback)
        self._releaseButton.pack(pady=5)

    def displayText(self, command):
        self._textDisplay.insert(tk.END, f"{command}\n")
        self._textDisplay.yview(tk.END)  # Desplazar el texto hacia abajo
    
    def takeOffCallback(self):
        try: 
            height = float(self._entryZ.get())
            if (height <= 0):
                self.displayText("Introduce a valid height.")
            else:
                msg = Command(command = "take-off", coordinates = [None, None, height])
                self._message = msg
                self.setMessage(msg)
        except ValueError:
            self.displayText("Introduce a valid numeric number.") 
    
    def landCallback(self):
        msg = Command(command = "land")
        self._message = msg
        self.setMessage(msg)

    def goToCallback(self):
        try:
            x = float(self._entryX.get())
            y = float(self._entryY.get())
            z = float(self._entryZ.get())
            if (z <= 0):
                self.displayText("Introduce a valid height.")
            else:
                msg = Command(command = "go to", coordinates = [x, y, z])
                self._message = msg
                self.setMessage(msg)
        except ValueError:
            self.displayText("Introduce valid numeric values.")
    
    def graspCallback(self):
        msg = Command(command = "grasp")
        self._message = msg
        self.setMessage(msg)


    def releaseCallback(self):
        msg = Command(command = "release")
        self._message = msg
        self.setMessage(msg)

    def checkOnlyNumbers(self, text):
        """Allows only integers or empty floats."""
        if text == "" or text == "-":
            return True
        try:
            float(text)
            return True
        except ValueError:
            return False
        
    def setMessage(self, msg):
        self._taskManager._gcsMessageQueue.put(msg)

    def checkShutdown(self):
        if self._shutdownEvent.is_set():
            self._root.quit()
        else:
            self._root.after(500, self.checkShutdown)

    def stop(self):
        print("[GCS] Shutting down.")
        if self._root:
            try: 
                self._root.quit()
            except tk.TclError:
                pass

if __name__ == "__main__":
    import time
    from TaskManager import TaskManager
    shutdownEvent = threading.Event()
    taskManager = TaskManager()

    gcs = GCS(taskManager, shutdownEvent)
    gcs.start() 

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Main] Shutdown signal received.")
        shutdownEvent.set()
        gcs.join()  
        print("[Main] GCS thread has been closed.")

