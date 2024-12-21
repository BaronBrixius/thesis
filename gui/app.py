import tkinter as tk
from threading import Thread, Event
from network_simulation.network import NodeNetwork
from gui.visualization_panel import VisualizationPanel
from gui.control_panel import ControlPanel
import networkx as nx
import numpy as np
import time

class NetworkControlApp:
    def __init__(self, num_nodes=200, initial_connections=2000, alpha=1.7):
        self.root = tk.Tk()
        self.root.title("Network Control Panel")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)   # Closing the window triggers the quit_application handler

        # Simulation parameters
        self.num_nodes = num_nodes
        self.initial_connections = initial_connections
        self.alpha = alpha
        self.step = 0

        # Shared variables for simulation intervals
        self.display_interval = 1000
        self.metrics_interval = 1000
        self.running = Event()

        # Initialize Network and GUI
        self.network = NodeNetwork(num_nodes=self.num_nodes, num_connections=self.initial_connections, alpha=self.alpha, random_seed=42)
        self.control_panel = ControlPanel(self.root, network=self.network, apply_changes_callback=self.apply_changes, toggle_simulation_callback=self.toggle_simulation, physics_callback=self.update_visualization)
        self.visualization_panel = VisualizationPanel(self.root, self.network)

        # Start simulation thread
        self.simulation_thread = Thread(target=self.run_simulation, daemon=True)
        self.simulation_thread.start()

        self.root.mainloop()

    def toggle_simulation(self):
        """Play/pause the simulation."""
        if self.running.is_set():
            self.running.clear()
            self.control_panel.playpause_button.config(text="Play")
        else:
            self.running.set()
            self.control_panel.playpause_button.config(text="Pause")

    def run_simulation(self):
        while True:
            if self.running.is_set():
                # Update network state
                self.network.update_network()

                if self.step % self.display_interval == 0:
                    self.update_visualization()

                if self.step % self.metrics_interval == 0:
                    self.control_panel.update_metrics(self.network, self.step)

                time.sleep(0.00000001)  # Small delay to avoid busy looping
                self.step += 1
            else:
                time.sleep(0.1)  # Sleep briefly when paused

    def update_visualization(self):
        """Update the visualization with current network state."""
        self.network.apply_forces(min(50, self.display_interval))
        self.visualization_panel.update(self.network, self.step)

    def apply_changes(self, **kwargs):
        """
        Apply changes to the network and simulation settings.
        """
        was_running = self.running.is_set()
        if self.running.is_set():
            self.running.clear()

        # Extract values from kwargs
        num_nodes = kwargs.get("num_nodes")
        num_connections = kwargs.get("num_connections")

        # Update network parameters
        self.network.alpha = kwargs.get("alpha")
        self.network.epsilon = kwargs.get("epsilon")
        self.display_interval = kwargs.get("display_interval")
        self.metrics_interval = kwargs.get("metrics_interval")

        # Update network structure if node or connection count changes
        if num_nodes != self.network.num_nodes or num_connections != self.network.num_connections:
            self.network.update_network_structure(num_nodes, num_connections)
            self.update_visualization()

        if was_running:
            self.running.set()

    def quit_application(self):
        """Terminate the simulation and close the application."""
        self.running.clear()  # Stop the simulation
        self.simulation_thread.join(timeout=1)  # Ensure the simulation thread exits
        self.root.quit()  # Quit the Tkinter main loop
        self.root.destroy()  # Destroy the root window