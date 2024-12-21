import tkinter as tk
from tkinter import ttk
import networkx as nx
import numpy as np
from network_simulation.network import NodeNetwork

class ControlPanel:
    def __init__(self, root, network:NodeNetwork, apply_changes_callback, toggle_simulation_callback, physics_callback):
        self.apply_changes_callback = apply_changes_callback
        self.previous_cluster_assignments = None
        self.physics_callback = physics_callback

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
        self.create_widgets(toggle_simulation_callback, physics_callback)

    def create_widgets(self, toggle_simulation_callback, physics_callback):
        for row, (key, config) in enumerate(self.configs.items()):
            self.create_labeled_entry(self.frame, config["label"], self.variables[key], row)

        # Apply button
        ttk.Button(self.frame, text="Apply", command=self.apply_changes).grid(row=len(self.configs), column=0, columnspan=2, pady=5)

        # Metrics Display
        metrics_frame = ttk.Frame(self.frame)
        metrics_frame.grid(row=len(self.configs) + 1, column=0, columnspan=2, sticky="EW")
        self.metrics_text = tk.Text(metrics_frame, height=40, width=30, wrap="word", state="disabled", bg="lightgray")
        self.metrics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.metrics_text.insert("1.0", " ")
        scrollbar = ttk.Scrollbar(metrics_frame, orient=tk.VERTICAL, command=self.metrics_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.metrics_text.config(yscrollcommand=scrollbar.set)

        # Play/Pause button
        self.playpause_button = ttk.Button(self.frame, text="Play", command=toggle_simulation_callback)
        self.playpause_button.grid(row=len(self.configs) + 2, column=0, columnspan=1, pady=5)

        # Physics button with hold functionality
        self.physics_button = ttk.Button(self.frame, text="Physics")
        self.physics_button.grid(row=len(self.configs) + 2, column=1, columnspan=1, pady=5)
        self.physics_button.bind("<ButtonPress>", self.start_physics)
        self.physics_button.bind("<ButtonRelease>", self.stop_physics)
        self.physics_running = False

    def start_physics(self, event):
        self.physics_running = True
        self.run_physics()

    def stop_physics(self, event):
        self.physics_running = False

    def run_physics(self):
        if self.physics_running:
            self.physics_callback()
            self.frame.after(50, self.run_physics)

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

    def update_metrics(self, network:NodeNetwork, step):
        graph = nx.from_numpy_array(network.adjacency_matrix)
        clustering_coeff = network.metrics.calculate_clustering_coefficient(graph)
        rewiring_chance = network.metrics.calculate_rewiring_chance(network.adjacency_matrix, network.activities)
        rewiring_rate = network.successful_rewirings / int(self.variables["metrics_interval"].get())

        # Detect communities with optional previous assignments
        cluster_assignments = network.metrics.detect_communities(network.adjacency_matrix)

        # Calculate metrics
        cluster_sizes = [len(cluster) for cluster in cluster_assignments]
        num_clusters = len(cluster_sizes)
        intra_cluster_densities = [network.metrics.calculate_intra_cluster_density(network.adjacency_matrix, cluster) for cluster in cluster_assignments]
        activity_variance = [np.var(network.activities[list(cluster)]) for cluster in cluster_assignments]

        # Calculate cluster colors
        color_names = ["Red", "Blue", "Green", "Purple", "Orange", "Yellow", "Brown", "Pink", "Grey"]
        color_indices = np.linspace(0, len(color_names) - 1, num_clusters, dtype=int)
        colors = [color_names[i] for i in color_indices]

        metrics_text = (
            f"Step: {step}\n"
            f"Clustering Coefficient: {clustering_coeff:.3f}\n"
            f"Rewiring Chance: {rewiring_chance:.3f}\n"
            f"Rewiring Rate: {rewiring_rate:.3f}\n"
            f"Cluster Count: {num_clusters}\n"
        )

        for i, (size, density, variance, color) in enumerate(zip(cluster_sizes, intra_cluster_densities, activity_variance, colors)):
            metrics_text += (
                f"Cluster {i}: ({color})\n"
                f"  Size: {size}\n"
                f"  Density: {density:.3f}\n"
                f"  Activity Variance: {variance:.3f}\n"
            )

        metrics_text += f"Cluster Assignments: {cluster_assignments}\n"

        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert("1.0", metrics_text)
        self.metrics_text.config(state="disabled")

        self.previous_cluster_assignments = cluster_assignments
