import tkinter as tk
from threading import Thread, Event
from network_simulation.network import NodeNetwork
from gui.visualization_panel import VisualizationPanel
from gui.control_panel import ControlPanel
import networkx as nx
import numpy as np
import time

class NetworkControlApp:
    def __init__(self, root, num_nodes=200, initial_connections=2000, alpha=1.7):
        self.root = root
        self.root.title("Network Control Panel")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)   # Closing the window triggers the quit_application handler
        self.clustering_coeffs = []  # To track clustering coefficients for calculating stddev

        # Simulation parameters
        self.num_nodes = num_nodes
        self.initial_connections = initial_connections
        self.alpha = alpha
        self.step = 0

        # Shared variables
        self.epsilon = tk.DoubleVar(value=0.4)
        self.display_interval = tk.IntVar(value=100)
        self.metrics_interval = tk.IntVar(value=100)
        self.running = Event()

        # Network and widget setup
        self.network = NodeNetwork(num_nodes=self.num_nodes, num_connections=self.initial_connections, alpha=self.alpha, epsilon=self.epsilon.get())
        self.control_panel = ControlPanel(root, self, self.epsilon, self.display_interval, self.metrics_interval, self.num_nodes, self.initial_connections, apply_changes_callback=self.apply_changes)
        self.visualization_panel  = VisualizationPanel(root, self.network)

        # Start simulation in a separate thread
        self.simulation_thread = Thread(target=self.run_simulation, daemon=True)
        self.simulation_thread.start()

    def run_simulation(self):
        while True:
            if self.running.is_set():
                # Update network state
                self.network.update_network()

                if self.step % self.display_interval.get() == 0:
                    self.update_visualization()

                if self.step % self.metrics_interval.get() == 0:
                    self.control_panel.update_metrics(self.network, self.step)

                time.sleep(0.00000001)  # Small delay to avoid busy looping
                self.step += 1
            else:
                time.sleep(0.1)  # Sleep briefly when paused

    def update_visualization(self):
        """Update the visualization with current network state."""
        self.network.apply_forces(min(50, self.display_interval.get()))
        self.visualization_panel.update(self.network, self.step)

    def on_input_change(self, event):
        """Highlight text fields when their values differ from current settings."""
        self.changes_pending = False

        # Compare current inputs with simulation parameters
        inputs = {
            "epsilon": (self.epsilon_input, float(self.epsilon.get())),
            "display_interval": (self.display_interval_input, int(self.display_interval.get())),
            "metrics_interval": (self.metrics_interval_input, int(self.metrics_interval.get())),
            "node_count": (self.node_count_input, self.num_nodes),
            "connection_count": (self.connection_count_input, self.initial_connections),
        }

        for key, (input_var, current_value) in inputs.items():
            input_widget = event.widget
            try:
                if float(input_var.get()) != current_value:
                    input_widget.config(background="lightyellow")
                    self.changes_pending = True
                else:
                    input_widget.config(background="white")
            except ValueError:
                # If the input is invalid (e.g., empty or non-numeric), keep it highlighted
                input_widget.config(background="lightyellow")
                self.changes_pending = True

    def apply_changes(self, num_nodes, num_connections):
        self.network.update_node_count(num_nodes)
        self.network.update_connection_count(num_connections)
        self.visualization_panel.update_network(self.network)

    def cancel_changes(self):
        """Reset inputs to current settings."""
        self.epsilon_input.set(str(self.epsilon.get()))
        self.display_interval_input.set(str(self.display_interval.get()))
        self.metrics_interval_input.set(str(self.metrics_interval.get()))
        self.node_count_input.set(str(self.num_nodes))
        self.connection_count_input.set(str(self.initial_connections))

    def start_simulation(self):
        self.running.set()

    def pause_simulation(self):
        self.running.clear()

    def quit_application(self):
        """Terminate the simulation and close the application."""
        self.running.clear()  # Stop the simulation
        self.simulation_thread.join(timeout=1)  # Ensure the simulation thread exits
        self.root.quit()  # Quit the Tkinter main loop
        self.root.destroy()  # Destroy the root window


# Main Application
if __name__ == "__main__":
    num_nodes = 200
    initial_connections = int(0.1 * (num_nodes * (num_nodes - 1) / 2))
    alpha = 1.7

    root = tk.Tk()
    app = NetworkControlApp(root, num_nodes, initial_connections, alpha=alpha)
    root.mainloop()
