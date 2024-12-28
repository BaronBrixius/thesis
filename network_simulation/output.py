import csv
import logging
import os
from network_simulation.metrics import Metrics
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Output:
    def __init__(self, project_dir, num_nodes=None, num_connections=None):
        self.logger = logging.getLogger(__name__)
        self.num_nodes = num_nodes
        self.num_connections = num_connections

        output_dir = os.path.join("output", project_dir)
        self.directories = {
            "base": output_dir,
            "images": os.path.join(output_dir, "images"),
        }
        self._initialize_directories()

        self.metrics_file_path = os.path.join(self.directories["base"], f"summary_metrics_nodes_{self.num_nodes}_edges_{self.num_connections}.csv")
        self.metrics_file = open(self.metrics_file_path, mode="w", newline="")
        self.csv_writer = None


    def _initialize_directories(self):
        for _, path in self.directories.items():
            os.makedirs(path, exist_ok=True)

    def save_network_image(self, visualization, step):
        image_path = os.path.join(self.directories["images"], f"network_nodes_{self.num_nodes}_edges_{self.num_connections}_step_{step}.jpg")
        visualization.fig.savefig(image_path)
        self.logger.info(f"Saved network visualization for step {step} to {image_path}")

    # Runtime Metrics Writing
    def write_metrics_line(self, step, adjacency_matrix, activities, metrics):
        row = self._compute_row(step, adjacency_matrix, activities, metrics)

        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.metrics_file, fieldnames=row.keys())
            self.csv_writer.writeheader()

        self.csv_writer.writerow(row)

    def _compute_row(self, step, adjacency_matrix, activities, metrics: Metrics):
        # Shared Computations
        adjacency_matrix_nx = nx.from_numpy_array(adjacency_matrix)
        previous_cluster_assignments = metrics.current_cluster_assignments
        cluster_assignments = metrics.get_cluster_assignments(adjacency_matrix, step)
        cluster_sizes = {i: len(cluster) for i, cluster in enumerate(cluster_assignments)}
        cluster_densities = {i: metrics.calculate_intra_cluster_density(adjacency_matrix, cluster) for i, cluster in enumerate(cluster_assignments)}
        avg_cluster_size = np.mean(list(cluster_sizes.values()))
        avg_cluster_density = np.mean(list(cluster_densities.values()))

        # Compute
        row = {
            "Step": step,
            "Clustering Coefficient": metrics.calculate_clustering_coefficient(adjacency_matrix_nx),
            "Average Path Length": metrics.calculate_average_path_length(adjacency_matrix_nx),
            "Rewiring Chance": metrics.calculate_rewiring_chance(adjacency_matrix, activities),
            "Rich-Club Coefficient": metrics.calculate_rich_club_coefficient(adjacency_matrix_nx),

            # Cluster Metrics
            "Cluster Membership": {i: cluster for i, cluster in enumerate(cluster_assignments)},
            "Cluster Count": len(cluster_assignments),
            "Cluster Membership Stability": metrics.calculate_cluster_membership_stability(cluster_assignments, previous_cluster_assignments),
            "Cluster Sizes": cluster_sizes,
            "Average Cluster Size": avg_cluster_size,
            "Cluster Densities": cluster_densities,
            "Average Cluster Density": avg_cluster_density,
            "Cluster Size Variance": metrics.calculate_cluster_size_variance([len(cluster) for cluster in cluster_assignments]),
        }

        row.update({f"Rewirings ({key})": value for key, value in metrics.rewirings.items()})
        return row
