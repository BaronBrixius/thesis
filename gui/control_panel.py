import tkinter as tk
from tkinter import ttk
import networkx as nx
import numpy as np
from network_simulation.network import NodeNetwork

class ControlPanel:
    def __init__(self, root, network:NodeNetwork, apply_changes_callback, toggle_simulation_callback):
        self.apply_changes_callback = apply_changes_callback
        self.toggle_simulation_callback = toggle_simulation_callback

        # Variables for inputs
        self.num_nodes_var = tk.StringVar(value=str(network.num_nodes))
        self.num_connections_var = tk.StringVar(value=str(network.num_connections))
        self.epsilon_var = tk.StringVar(value=str(network.epsilon))
        self.display_interval_var = tk.StringVar(value="100")
        self.metrics_interval_var = tk.StringVar(value="100")

        # Create widgets
        self.frame = ttk.Frame(root, padding="10")
        self.frame.grid(row=0, column=0, sticky="NS")
        self.create_widgets()

    def create_widgets(self):
        # Widget definitions
        widget_configs = [
            {"label": "Node Count:", "var": self.num_nodes_var},
            {"label": "Connection Count:", "var": self.num_connections_var},
            {"label": "Epsilon:", "var": self.epsilon_var},
            {"label": "Display Interval:", "var": self.display_interval_var},
            {"label": "Metrics Interval:", "var": self.metrics_interval_var},
        ]

        for row, config in enumerate(widget_configs):
            self.create_labeled_entry(self.frame, config["label"], config["var"], row)

        # Apply button
        ttk.Button(self.frame, text="Apply Changes", command=self.apply_changes).grid(row=len(widget_configs), column=0, columnspan=2, pady=5)

        # Metrics Display
        self.metrics_text = tk.Text(self.frame, height=10, width=30, wrap="word", state="disabled", bg="lightgray")
        self.metrics_text.grid(row=len(widget_configs) + 1, column=0, columnspan=2, sticky="EW")

        # Play/Pause button
        ttk.Button(self.frame, text="Play/Pause", command=self.toggle_simulation_callback).grid(row=len(widget_configs) + 2, column=0, columnspan=2, pady=5)

    def create_labeled_entry(self, parent, label_text, variable, row):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="W")
        ttk.Entry(parent, textvariable=variable, width=10).grid(row=row, column=1, sticky="EW")

    def apply_changes(self):
        """Extract values and pass them to the callback."""
        num_nodes = int(self.num_nodes_var.get())
        num_connections = int(self.num_connections_var.get())
        epsilon = float(self.epsilon_var.get())
        display_interval = int(self.display_interval_var.get())
        metrics_interval = int(self.metrics_interval_var.get())
        self.apply_changes_callback(num_nodes, num_connections, epsilon, display_interval, metrics_interval)

    def toggle_simulation(self):
        if self.simulation_button.cget("text") == "Play":
            self.simulation_button.config(text="Pause")
            self.toggle_simulation_callback(start=True)
        else:
            self.simulation_button.config(text="Play")
            self.toggle_simulation_callback(start=False)

    def update_metrics(self, network, step):
        """Update the metrics display."""
        clustering_coeff = network.metrics.calculate_clustering_coefficient(nx.from_numpy_array(network.adjacency_matrix))
        rewiring_chance = network.metrics.calculate_rewiring_chance(network.adjacency_matrix, network.activities)
        rewiring_rate = network.successful_rewirings / int(self.metrics_interval_var.get())
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
