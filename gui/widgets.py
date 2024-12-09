import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from network_simulation.visualization import Visualization, ColorBy
import networkx as nx
import numpy as np

class ControlPanel:
    def __init__(self, root, app, epsilon, display_interval, metrics_interval, num_nodes, num_connections, apply_changes_callback):
        self.root = root
        self.app = app
        self.apply_changes_callback = apply_changes_callback

        self.epsilon = epsilon
        self.display_interval = display_interval
        self.metrics_interval = metrics_interval

        self.num_nodes = tk.StringVar(value=str(num_nodes))
        self.num_connections = tk.StringVar(value=str(num_connections))

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky="NS")
        frame.columnconfigure(0, weight=1)

        # Epsilon
        ttk.Label(frame, text="Epsilon:").grid(row=0, column=0, sticky="W")
        epsilon_entry = ttk.Entry(frame, textvariable=self.epsilon, width=10)
        epsilon_entry.grid(row=0, column=1, sticky="EW")

        # Display Interval
        ttk.Label(frame, text="Display Interval:").grid(row=1, column=0, sticky="W")
        interval_entry = ttk.Entry(frame, textvariable=self.display_interval, width=10)
        interval_entry.grid(row=1, column=1, sticky="EW")

        # Metrics Interval
        ttk.Label(frame, text="Metrics Interval:").grid(row=2, column=0, sticky="W")
        metrics_entry = ttk.Entry(frame, textvariable=self.metrics_interval, width=10)
        metrics_entry.grid(row=2, column=1, sticky="EW")

        # Node Count
        ttk.Label(frame, text="Node Count:").grid(row=3, column=0, sticky="W")
        node_entry = ttk.Entry(frame, textvariable=self.num_nodes, width=10)
        node_entry.grid(row=3, column=1, sticky="EW")

        # Connection Count
        ttk.Label(frame, text="Connection Count:").grid(row=4, column=0, sticky="W")
        connection_entry = ttk.Entry(frame, textvariable=self.num_connections, width=10)
        connection_entry.grid(row=4, column=1, sticky="EW")

        # Apply Button
        apply_button = ttk.Button(frame, text="Apply Changes", command=self.apply_changes)
        apply_button.grid(row=5, column=0, columnspan=2)

        # Metrics Display
        self.metrics_text = tk.Text(frame, height=10, width=30, wrap="word", state="disabled", bg="lightgray")
        self.metrics_text.grid(row=6, column=0, columnspan=2, sticky="EW")

        # Play/Pause Buttons
        ttk.Label(frame, text="Simulation Control:").grid(row=7, column=0, sticky="W")
        control_buttons_frame = ttk.Frame(frame)
        control_buttons_frame.grid(row=7, column=1, sticky="EW")
        play_button = ttk.Button(control_buttons_frame, text="Play", command=self.app.start_simulation)
        play_button.grid(row=0, column=0, padx=5)
        pause_button = ttk.Button(control_buttons_frame, text="Pause", command=self.app.pause_simulation)
        pause_button.grid(row=0, column=1, padx=5)

        # Quit Button
        quit_button = ttk.Button(frame, text="Quit", command=self.app.quit_application)
        quit_button.grid(row=8, column=0, columnspan=2, pady=10)

    def apply_changes(self):
        num_nodes = int(self.num_nodes.get())
        num_connections = int(self.num_connections.get())
        self.apply_changes_callback(num_nodes, num_connections)

    def update_metrics(self, network, step):
        clustering_coeff = network.metrics.calculate_clustering_coefficient(nx.from_numpy_array(network.adjacency_matrix))
        rewiring_chance = network.metrics.calculate_rewiring_chance(network.adjacency_matrix, network.activities)
        rewiring_rate = network.successful_rewirings / self.metrics_interval.get()
        cluster_assignments = network.metrics.detect_communities(network.adjacency_matrix)

        metrics_text = (
            f"Step: {step}\n"
            f"Clustering Coefficient: {clustering_coeff:.3f}\n"
            f"Rewiring Chance: {rewiring_chance:.3f}\n"
            f"Rewiring Rate: {rewiring_rate:.3f}\n"
            f"Cluster Count: {np.max(cluster_assignments) + 1}\n"
        )

        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, metrics_text)
        self.metrics_text.config(state="disabled")


class VisualizationPanel:
    def __init__(self, root, network):
        self.root = root
        self.network = network
        self.visualizer = Visualization(
            positions=network.positions,
            activities=network.activities,
            adjacency_matrix=network.adjacency_matrix,
            color_by=ColorBy.ACTIVITY,
            draw_lines=True,
            show=False
        )
        self.create_canvas()

    def create_canvas(self):
        frame = ttk.Frame(self.root)
        frame.grid(row=0, column=1, sticky="NSEW")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.canvas = FigureCanvasTkAgg(self.visualizer.fig, master=frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="NSEW")
        self.canvas.draw()

    def update(self, network, step):
        self.visualizer.update_plot(
            positions=network.positions,
            activities=network.activities,
            adjacency_matrix=network.adjacency_matrix,
            title=f"Step: {step}",
            draw_lines=True
        )
        self.canvas.draw()

    def update_network(self, network):
        self.network = network
