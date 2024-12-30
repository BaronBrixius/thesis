import csv
import logging
import os
from network_simulation.metrics import Metrics
from network_simulation.network import NodeNetwork
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Output:
    def __init__(self, project_dir, num_nodes=None, num_connections=None):
        self.logger = logging.getLogger(__name__)
        self.num_nodes = num_nodes
        self.num_connections = num_connections

        self.project_dir = project_dir
        os.makedirs(project_dir, exist_ok=True)

        self.metrics_file_path = os.path.join(self.project_dir, f"summary_metrics_nodes_{self.num_nodes}_edges_{self.num_connections}.csv")
        print(self.metrics_file_path)
        self.metrics_file = open(self.metrics_file_path, mode="w", newline="")
        self.csv_writer = None

    # Runtime Metrics Writing
    def write_metrics_line(self, step, network: NodeNetwork):
        row = self._compute_row(step, network)

        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.metrics_file, fieldnames=row.keys())
            self.csv_writer.writeheader()

        self.csv_writer.writerow(row)

    def _compute_row(self, step, network: NodeNetwork):
        # Shared Computations
        graph = network.graph
        metrics = network.metrics
        activities = network.activities.a

        # Get cluster metrics
        cluster_metrics = metrics.get_cluster_metrics(graph, step)

        # Compute row data
        row = {
            "Step": step,
            "Clustering Coefficient": metrics.get_clustering_coefficient(graph),
            "Average Path Length": metrics.calculate_average_path_length(graph),
            "Rewiring Chance": metrics.calculate_rewiring_chance(graph, activities),
        }

        # Update with cluster metrics
        row.update(cluster_metrics)

        # Add rewiring metrics
        row.update(
            {f"Rewirings ({key})": value for key, value in metrics.rewirings.items()}
        )
        return row
