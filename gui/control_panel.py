import tkinter as tk
from tkinter import ttk
import networkx as nx
import numpy as np

class ControlPanel:
    def __init__(self, root, app, epsilon, display_interval, metrics_interval, num_nodes, num_connections, apply_changes_callback):
        self.root = root
        self.app = app
        self.apply_changes_callback = apply_changes_callback

        # Widgets
        self.frame = ttk.Frame(root, padding="10")
        self.frame.grid(row=0, column=0, sticky="NS")
        self.frame.columnconfigure(0, weight=1)

        # Control variables
        self.epsilon = epsilon
        self.display_interval = display_interval
        self.metrics_interval = metrics_interval
        self.num_nodes = num_nodes
        self.num_connections = num_connections

        # Create control widgets
        self.create_widgets()

    def create_widgets(self):
        # Widget definitions
        widget_configs = [
            {"label": "Epsilon:", "var": self.epsilon},
            {"label": "Display Interval:", "var": self.display_interval},
            {"label": "Metrics Interval:", "var": self.metrics_interval},
            {"label": "Node Count:", "var": self.num_nodes},
            {"label": "Connection Count:", "var": self.num_connections},
        ]

        for row, config in enumerate(widget_configs):
            self.create_labeled_entry(self.frame, config["label"], config["var"], row)

        # Apply Button
        apply_button = ttk.Button(self.frame, text="Apply Changes", command=self.apply_changes)
        apply_button.grid(row=len(widget_configs), column=0, columnspan=2, pady=5)

        # Metrics Display
        self.metrics_text = tk.Text(self.frame, height=10, width=30, wrap="word", state="disabled", bg="lightgray")
        self.metrics_text.grid(row=6, column=0, columnspan=2, sticky="EW")

        # PlayPause Button
        self.simulation_button = ttk.Button(self.frame, text="Play", command=self.toggle_simulation)
        self.simulation_button.grid(row=len(widget_configs) + 2, column=0, columnspan=2, pady=5)

    def create_labeled_entry(self, parent, label_text, variable, row):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="W")
        ttk.Entry(parent, textvariable=variable, width=10).grid(row=row, column=1, sticky="EW")

    def apply_changes(self):
        num_nodes = int(self.num_nodes.get())
        num_connections = int(self.num_connections.get())
        self.apply_changes_callback(num_nodes, num_connections)

    def toggle_simulation(self):
        if self.app.running.is_set():
            self.app.pause_simulation()
            self.simulation_button.config(text="Play")
        else:
            self.app.start_simulation()
            self.simulation_button.config(text="Pause")

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
