import tkinter as tk
from tkinter import ttk
from threading import Thread, Event
from network_simulation.network import NodeNetwork
from network_simulation.network_visualizer import NetworkVisualizer
import time


class NetworkControlApp:
    def __init__(self, root, num_nodes, initial_connections, alpha=1.7):
        self.root = root
        self.root.title("Network Control Panel")

        # Simulation parameters
        self.num_nodes = num_nodes
        self.initial_connections = initial_connections
        self.alpha = alpha

        # Shared variables
        self.epsilon = tk.DoubleVar(value=0.4)
        self.display_interval = tk.IntVar(value=1)
        self.stop_event = Event()

        # Network and Visualizer
        self.network = NodeNetwork(
            num_nodes=self.num_nodes,
            num_connections=self.initial_connections,
            alpha=self.alpha,
            epsilon=self.epsilon.get()
        )
        self.visualizer = NetworkVisualizer(self.network)

        # Create UI
        self.create_widgets()

        # Start simulation in a separate thread
        self.simulation_thread = Thread(target=self.run_simulation, daemon=True)
        self.simulation_thread.start()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky="NSEW")

        # Epsilon Slider
        ttk.Label(frame, text="Epsilon:").grid(row=0, column=0, sticky="W")
        epsilon_slider = ttk.Scale(
            frame, from_=0.1, to=1.0, variable=self.epsilon, orient="horizontal"
        )
        epsilon_slider.grid(row=0, column=1, sticky="EW")

        # Display Interval Slider
        ttk.Label(frame, text="Display Interval:").grid(row=1, column=0, sticky="W")
        interval_slider = ttk.Scale(
            frame, from_=1, to=1000, variable=self.display_interval, orient="horizontal"
        )
        interval_slider.grid(row=1, column=1, sticky="EW")

        # Quit Button
        quit_button = ttk.Button(frame, text="Quit", command=self.quit_application)
        quit_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Configure resizing
        frame.columnconfigure(1, weight=1)

    def run_simulation(self):
        step = 0
        while not self.stop_event.is_set():
            # Update epsilon dynamically
            self.network.epsilon = self.epsilon.get()

            # Update network state
            self.network.update_network()
            self.network.apply_forces(min(150, self.display_interval.get()))

            # Display network at intervals
            if step % self.display_interval.get() == 0:
                self.visualizer.show(step)

            time.sleep(0.001)  # Small delay to avoid busy looping
            step +=1

    def quit_application(self):
        self.stop_event.set()
        self.root.destroy()


# Main Application
if __name__ == "__main__":
    num_nodes = 200
    initial_connections = 1200
    alpha = 1.7

    root = tk.Tk()
    app = NetworkControlApp(root, num_nodes, initial_connections, alpha=alpha)
    root.mainloop()
