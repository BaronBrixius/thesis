import tkinter as tk
from tkinter import ttk
from threading import Thread, Event
from network_simulation.network import NodeNetwork
from network_simulation.visualization import Visualization, ColorBy
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np
import time

matplotlib.use("TkAgg")

class NetworkControlApp:
    def __init__(self, root, num_nodes, initial_connections, alpha=1.7):
        self.root = root
        self.root.title("Network Control Panel")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)
        self.step = 0
        self.clustering_coeffs = []  # To track clustering coefficients for calculating stddev

        # Simulation parameters
        self.num_nodes = num_nodes
        self.initial_connections = initial_connections
        self.alpha = alpha

        # Shared variables
        self.epsilon = tk.DoubleVar(value=0.4)
        self.display_interval = tk.IntVar(value=1)
        self.running = Event()  # Simulation running state (paused initially)
        self.stop_event = Event()

        # Input fields to track changes
        self.epsilon_input = tk.StringVar(value=str(self.epsilon.get()))
        self.display_interval_input = tk.StringVar(value=str(self.display_interval.get()))
        self.node_count_input = tk.StringVar(value=str(self.num_nodes))
        self.connection_count_input = tk.StringVar(value=str(self.initial_connections))
        self.changes_pending = False

        # Network
        self.network = NodeNetwork(
            num_nodes=self.num_nodes,
            num_connections=self.initial_connections,
            alpha=self.alpha,
            epsilon=self.epsilon.get()
        )

        # Visualization setup
        self.visualizer = Visualization(
            positions=self.network.positions,
            activities=self.network.activities,
            adjacency_matrix=self.network.adjacency_matrix,
            color_by=ColorBy.ACTIVITY,
            draw_lines=True,
            show=False
        )

        # Create UI
        self.create_widgets()

        # Start simulation in a separate thread
        self.simulation_thread = Thread(target=self.run_simulation, daemon=True)
        self.simulation_thread.start()

    def create_widgets(self):
        # Main layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Frame for controls and display
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="NSEW")
        main_frame.columnconfigure(1, weight=1)

        # Control Panel
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(row=0, column=0, sticky="NS")
        control_frame.columnconfigure(0, weight=1)

        # Epsilon Control
        ttk.Label(control_frame, text="Epsilon:").grid(row=0, column=0, sticky="W")
        epsilon_entry = ttk.Entry(control_frame, textvariable=self.epsilon_input, width=10)
        epsilon_entry.grid(row=0, column=1, sticky="EW")
        epsilon_entry.bind("<KeyRelease>", self.on_input_change)

        # Display Interval Control
        ttk.Label(control_frame, text="Display Interval:").grid(row=1, column=0, sticky="W")
        interval_entry = ttk.Entry(control_frame, textvariable=self.display_interval_input, width=10)
        interval_entry.grid(row=1, column=1, sticky="EW")
        interval_entry.bind("<KeyRelease>", self.on_input_change)

        # Node Count Control
        ttk.Label(control_frame, text="Node Count:").grid(row=2, column=0, sticky="W")
        node_count_entry = ttk.Entry(control_frame, textvariable=self.node_count_input, width=10)
        node_count_entry.grid(row=2, column=1, sticky="EW")
        node_count_entry.bind("<KeyRelease>", self.on_input_change)

        # Connection Count Control
        ttk.Label(control_frame, text="Connection Count:").grid(row=3, column=0, sticky="W")
        connection_count_entry = ttk.Entry(control_frame, textvariable=self.connection_count_input, width=10)
        connection_count_entry.grid(row=3, column=1, sticky="EW")
        connection_count_entry.bind("<KeyRelease>", self.on_input_change)

        # Apply/Cancel Buttons
        action_buttons_frame = ttk.Frame(control_frame)
        action_buttons_frame.grid(row=4, column=0, columnspan=2, pady=10)
        apply_button = ttk.Button(action_buttons_frame, text="Apply Changes", command=self.apply_changes)
        apply_button.grid(row=0, column=0, padx=5)
        cancel_button = ttk.Button(action_buttons_frame, text="Cancel Changes", command=self.cancel_changes)
        cancel_button.grid(row=0, column=1, padx=5)

        # Play and Pause Buttons
        ttk.Label(control_frame, text="Simulation Control:").grid(row=5, column=0, sticky="W")
        control_buttons_frame = ttk.Frame(control_frame)
        control_buttons_frame.grid(row=5, column=1, sticky="EW")
        play_button = ttk.Button(control_buttons_frame, text="Play", command=self.start_simulation)
        play_button.grid(row=0, column=0, padx=5)
        pause_button = ttk.Button(control_buttons_frame, text="Pause", command=self.pause_simulation)
        pause_button.grid(row=0, column=1, padx=5)

        # Quit Button
        quit_button = ttk.Button(control_frame, text="Quit", command=self.quit_application)
        quit_button.grid(row=6, column=0, columnspan=2, pady=10)

        # Network Visualization
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=0, column=1, sticky="NSEW")
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)

        # Matplotlib Canvas for Visualization
        self.canvas = FigureCanvasTkAgg(self.visualizer.fig, master=display_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="NSEW")
        self.canvas.draw()

    def run_simulation(self):
        while True:
            if self.running.is_set():
                # Update network state
                self.network.update_network()

                if self.step % self.display_interval.get() == 0:
                    self.update_visualization()

                time.sleep(0.00000001)  # Small delay to avoid busy looping
                self.step += 1
            else:
                time.sleep(0.1)  # Sleep briefly when paused

    def update_visualization(self):
        """Update the visualization with current network state."""
        self.network.apply_forces(min(25, self.display_interval.get()))
                    
        # Calculate metrics
        clustering_coeff = self.network.metrics.calculate_clustering_coefficient(nx.from_numpy_array(self.network.adjacency_matrix))
        self.clustering_coeffs.append(clustering_coeff)
        if len(self.clustering_coeffs) > 100:  # Keep only the last 100 values for stddev
            self.clustering_coeffs.pop(0)

        cc_stddev = np.std(self.clustering_coeffs) if self.clustering_coeffs else 0
        rewiring_chance = self.network.metrics.calculate_rewiring_chance(
            self.network.adjacency_matrix, self.network.activities
        )
        rewiring_rate = self.network.successful_rewirings / self.display_interval.get()
        self.network.successful_rewirings = 0

        title = (f"Step: {self.step} CC: {clustering_coeff:.3f}, CC StdDev: {cc_stddev:.5f}, Rewiring Chance: {rewiring_chance:.3f}, Rewiring Rate: {rewiring_rate:.3f}")
        self.visualizer.update_plot(
            positions=self.network.positions,
            activities=self.network.activities,
            adjacency_matrix=self.network.adjacency_matrix,
            title=title,
            draw_lines=True
        )
        self.canvas.draw()


    def on_input_change(self, event):
        """Highlight text fields when their values differ from current settings."""
        self.changes_pending = False

        # Compare current inputs with simulation parameters
        inputs = {
            "epsilon": (self.epsilon_input, float(self.epsilon.get())),
            "display_interval": (self.display_interval_input, int(self.display_interval.get())),
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

    def apply_changes(self):
        self.epsilon.set(float(self.epsilon_input.get()))
        self.display_interval.set(int(self.display_interval_input.get()))

        # Update node and connection counts
        new_node_count = int(self.node_count_input.get())
        new_connection_count = int(self.connection_count_input.get())
        if new_node_count != self.num_nodes:
            self.network.update_node_count(new_node_count)
            self.num_nodes = new_node_count
        if new_connection_count != self.initial_connections:
            self.network.update_connection_count(new_connection_count)
            self.initial_connections = new_connection_count

        self.update_visualization()

    def cancel_changes(self):
        """Reset inputs to current settings."""
        self.epsilon_input.set(str(self.epsilon.get()))
        self.display_interval_input.set(str(self.display_interval.get()))
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
