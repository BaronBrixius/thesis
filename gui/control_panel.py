import tkinter as tk
from tkinter import ttk
import networkx as nx
import numpy as np
from network_simulation.network import NodeNetwork

class ControlPanel:
    def __init__(self, root, network:NodeNetwork, apply_changes_callback, toggle_simulation_callback):
        self.apply_changes_callback = apply_changes_callback
        self.toggle_simulation_callback = toggle_simulation_callback

        # Configuration for variables and their labels
        self.configs = {
            "num_nodes": {"label": "Node Count:", "default": network.num_nodes, "type": int},
            "num_connections": {"label": "Connection Count:", "default": network.num_connections, "type": int},
            "alpha": {"label": "Alpha:", "default": network.alpha, "type": float},
            "epsilon": {"label": "Epsilon:", "default": network.epsilon, "type": float},
            "display_interval": {"label": "Display Interval:", "default": 1000, "type": int},
            "metrics_interval": {"label": "Metrics Interval:", "default": 1000, "type": int},
        }

        # Initialize variables dynamically
        self.variables = {
            key: tk.StringVar(value=str(config["default"]))
            for key, config in self.configs.items()
        }

        # Create widgets
        self.frame = ttk.Frame(root, padding="10")
        self.frame.grid(row=0, column=0, sticky="NS")
        self.create_widgets()

    def create_widgets(self):
        for row, (key, config) in enumerate(self.configs.items()):
            self.create_labeled_entry(self.frame, config["label"], self.variables[key], row)

        # Apply button
        ttk.Button(self.frame, text="Apply", command=self.apply_changes).grid(row=len(self.configs), column=0, columnspan=2, pady=5)

        # Metrics Display
        self.metrics_text = tk.Text(self.frame, height=10, width=30, wrap="word", state="disabled", bg="lightgray")
        self.metrics_text.grid(row=len(self.configs) + 1, column=0, columnspan=2, sticky="EW")

        # Play/Pause button
        ttk.Button(self.frame, text="Play/Pause", command=self.toggle_simulation_callback).grid(row=len(self.configs) + 2, column=0, columnspan=2, pady=5)

    def create_labeled_entry(self, parent, label_text, variable, row):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="W")
        ttk.Entry(parent, textvariable=variable, width=10).grid(row=row, column=1, sticky="EW")

    def apply_changes(self):
        """Extract values dynamically and pass them to the callback."""
        try:
            # Iterate over configs and variables in parallel
            values = {
                key: self._parse_value(config["type"], self.variables[key].get())
                for key, config in self.configs.items()
            }
            self.apply_changes_callback(**values)
        except ValueError as e:
            print("Invalid input detected:", e)

    def _parse_value(self, var_type, value):
        return var_type(value)

    def update_metrics(self, network, step):
        """Update the metrics display."""
        clustering_coeff = network.metrics.calculate_clustering_coefficient(nx.from_numpy_array(network.adjacency_matrix))
        rewiring_chance = network.metrics.calculate_rewiring_chance(network.adjacency_matrix, network.activities)
        rewiring_rate = network.successful_rewirings / int(self.variables["metrics_interval"].get())
        cluster_assignments = nx.algorithms.community.louvain_communities(nx.from_numpy_array(network.adjacency_matrix))
        cluster_sizes = [len(cluster) for cluster in cluster_assignments]
        num_clusters = len(cluster_sizes)

        metrics_text = (
            f"Step: {step}\n"
            f"Clustering Coefficient: {clustering_coeff:.3f}\n"
            f"Rewiring Chance: {rewiring_chance:.3f}\n"
            f"Rewiring Rate: {rewiring_rate:.3f}\n"
            f"Cluster Count: {num_clusters}\n"
            f"Cluster Sizes: {cluster_sizes}\n"
        )

        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, metrics_text)
        self.metrics_text.config(state="disabled")
