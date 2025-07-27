import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import queue
import threading


class RealTimePlotter:
    def __init__(self, root):
        self.root = root
        self.root.title("Height Controller Real-Time Plot")

        # Create plot queue
        self.plot_queue = queue.Queue()

        # Create figure and canvas
        self.fig = Figure(figsize=(10, 8))
        self.ax1 = self.fig.add_subplot(311)  # Height
        self.ax2 = self.fig.add_subplot(312)  # Velocity
        self.ax3 = self.fig.add_subplot(313)  # Force

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Initialize data lists
        self.times = []
        self.heights = []
        self.height_refs = []
        self.velocities = []
        self.vel_refs = []
        self.forces = []

        # Start update loop
        self.root.after(100, self.update_plots)

    def update_plots(self):
        # Get all available data from queue
        while not self.plot_queue.empty():
            data = self.plot_queue.get()
            self.times.append(data["time"])
            self.heights.append(data["height"])
            self.height_refs.append(data["height_ref"])
            self.velocities.append(data["velocity"])
            self.vel_refs.append(data["vel_ref"])
            self.forces.append(data["force"])

        # Update plots
        if self.times:  # Only plot if we have data
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()

            # Height plot
            self.ax1.plot(self.times, self.heights, "b-", label="Height")
            self.ax1.plot(self.times, self.height_refs, "r--", label="Reference")
            self.ax1.set_ylabel("Height (m)")
            self.ax1.legend()

            # Velocity plot
            self.ax2.plot(self.times, self.velocities, "b-", label="Velocity")
            self.ax2.plot(self.times, self.vel_refs, "r--", label="Reference")
            self.ax2.set_ylabel("Velocity (m/s)")
            self.ax2.legend()

            # Force plot
            self.ax3.plot(self.times, self.forces, "g-", label="Force")
            self.ax3.set_xlabel("Time (s)")
            self.ax3.set_ylabel("Force (N)")
            self.ax3.legend()

            self.canvas.draw()

        # Schedule next update
        self.root.after(100, self.update_plots)


# into each controller thread, you can add the following lines to send data to the plotter:
class DroneHeightController(threading.Thread):
    def __init__(self, shutdownEvent, heightControllerUpdateEvent, plot_queue=None):
        # ...existing init code...
        self._plot_queue = plot_queue

    def run(self):
        # ...existing run code...

        while not self._shutdownEvent.is_set():
            # ...existing control code...

            # Send data to plot queue if available
            if self._plot_queue is not None:
                plot_data = {
                    "time": self._currentTimestamp,
                    "height": self._currentHeight,
                    "height_ref": self._heightReference,
                    "velocity": self._currentVelocity[2],
                    "vel_ref": targetVelocity,
                    "force": rotorForce,
                }
                self._plot_queue.put(plot_data)


# into the main thread (GUI only works in the main thread), you can initialize the plotter and pass the queue:
def main():
    # Create root window
    root = tk.Tk()

    # Create plotter
    plotter = RealTimePlotter(root)

    # Initialize controllers with plot queue
    droneController = DroneHeightController(
        shutdownEvent, heightControllerUpdateEvent, plot_queue=plotter.plot_queue
    )

    # Start controller thread
    droneController.start()

    try:
        # Start GUI main loop
        root.mainloop()
    except KeyboardInterrupt:
        shutdownEvent.set()

    droneController.join()
