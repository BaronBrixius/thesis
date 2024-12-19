import csv
import logging
import os
import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame
import h5py
import matplotlib.pyplot as plt
from network_simulation.calculator import Calculator

class Output:
    def __init__(self, project_dir, num_nodes=None, num_connections=None):
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

        self.previous_adjacency_matrix = None
        self.previous_cluster_assignments = None

        self.calculator = Calculator()
        self.logger = self._initialize_logger()

    def _initialize_directories(self):
        for _, path in self.directories.items():
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def _initialize_logger():
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        return logging.getLogger(__name__)

    ### Snapshot Methods ###

    def save_snapshot(self, step, activities, adjacency_matrix, successful_rewirings):
        snapshot_file = os.path.join(self.directories["snapshots"], f"snapshot_nodes_{self.num_nodes}_edges_{self.num_connections}_step_{step}.h5")
        with h5py.File(snapshot_file, "w") as h5file:
            h5file.create_dataset("activities", data=activities)
            h5file.create_dataset("adjacency_matrix", data=adjacency_matrix)
            h5file.create_dataset("successful_rewirings", data=successful_rewirings)    # TODO I think this can be calculated post-run at some point, but need permission
        self.logger.info(f"Saved snapshot for step {step} to {snapshot_file}")

    def load_snapshot(self, snapshot_file):
        with h5py.File(os.path.join(self.directories["snapshots"], snapshot_file), "r") as h5file:
            return {
                "activities": h5file["activities"][:],
                "adjacency_matrix": h5file["adjacency_matrix"][:],
                "successful_rewirings": h5file["successful_rewirings"][()]
            }

    def save_network_image(self, visualization, step):
        image_path = os.path.join(self.directories["images"], f"network_nodes_{self.num_nodes}_edges_{self.num_connections}_step_{step}.jpg")
        visualization.fig.savefig(image_path)
        self.logger.info(f"Saved network visualization for step {step} to {image_path}")

    # Runtime Metrics Writing
    def write_metrics_line(self, step, adjacency_matrix, activities, successful_rewirings):
        metrics = self._compute_metrics(step, adjacency_matrix, activities, successful_rewirings, self.previous_adjacency_matrix, self.previous_cluster_assignments)

        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.metrics_file, fieldnames=metrics.keys())
            self.csv_writer.writeheader()

        self.csv_writer.writerow(metrics)
    
    def _compute_metrics(self, step, adjacency_matrix, activities, successful_rewirings, previous_adjacency_matrix=None, previous_cluster_assignments=None):
        # Shared Computations
        adjacency_matrix_nx = nx.from_numpy_array(adjacency_matrix)
        cluster_assignments = self.calculator.detect_communities(adjacency_matrix_nx)

        # Compute
        metrics = {
            "Step": step,
            "Clustering Coefficient": self.calculator.calculate_clustering_coefficient(adjacency_matrix_nx),
            "Average Path Length" : self.calculator.calculate_average_path_length(adjacency_matrix_nx),
            # "Assortativity" : self.calculator.calculate_assortativity(adjacency_matrix_nx),
            # "Betweenness Centrality" : self.calculator.calculate_betweenness_centrality(adjacency_matrix_nx)
            # "Modularity" : self.calculator.calculate_modularity(adjacency_matrix_nx),

            ### Rewiring Metrics ###
            "Rewirings (interval)" : successful_rewirings,
            "Rewiring Chance" : self.calculator.calculate_rewiring_chance(adjacency_matrix, activities),
            "Edge Persistence" : self.calculator.calculate_edge_persistence(adjacency_matrix, previous_adjacency_matrix),

            ### Cluster Metrics ###
            "Cluster Membership" : cluster_assignments,
            "Cluster Count" : np.max(cluster_assignments) + 1,
            "Cluster Membership Stability" : self.calculator.calculate_cluster_membership_stability(cluster_assignments, previous_cluster_assignments),
            "Cluster Size Variance" : self.calculator.calculate_cluster_size_variance(cluster_assignments),
        }

        # Update old states for temporal metrics
        self.previous_adjacency_matrix = adjacency_matrix.copy()
        self.previous_cluster_assignments = cluster_assignments.copy()

        return metrics

    ### Post-Run Metrics Calculation ###

    def output_line_graph(self, metrics, xlim=None, ylim=None, title=None, xlabel="Step Number", ylabel="Value"):
        """
        Generate a line graph for specified metrics over time.
        """
        if not os.path.exists(self.metrics_file_path):
            self.logger.warning(f"No metrics summary file found at {self.metrics_file_path}")
            return

        data = pd.read_csv(self.metrics_file_path)

        # Validate the existence of metrics in the data
        missing_metrics = [metric for metric in metrics if metric not in data.columns]
        if missing_metrics:
            self.logger.warning(f"Missing required metrics in metrics summary: {missing_metrics}")
            return

        plt.figure(figsize=(10, 6))
        for metric in metrics:
            plt.plot(data["Step"], data[metric], label=metric)

        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        if not title:
            title = f"{', '.join(metrics)} Over Time"

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        # Save the graph
        metrics_graph_path = os.path.join(self.directories["plots"], f"{'_'.join(metrics)}_nodes_{self.num_nodes}_edges_{self.num_connections}.jpg")
        plt.savefig(metrics_graph_path)
        plt.close()
        self.logger.info(f"Saved line graph for metrics {metrics} to {metrics_graph_path}")
