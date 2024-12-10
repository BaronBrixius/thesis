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
            "snapshots": os.path.join(output_dir, "state_snapshots"),
            "images": os.path.join(output_dir, "images"),
            "plots": os.path.join(output_dir, "plots"),
        }
        self._initialize_directories()

        self.metrics_file_path = os.path.join(self.directories["base"], f"summary_metrics_nodes_{self.num_nodes}_edges_{self.num_connections}.csv")

        self.calculator = Calculator()
        self.logger = self._initialize_logger()

    def _initialize_directories(self):
        for _, path in self.directories.items():
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def _initialize_logger():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        return logging.getLogger(__name__)

    ### Runtime Snapshot/Image Methods ###

    def save_snapshot(self, step, activities, adjacency_matrix, successful_rewirings):
        # Save activities and adjacency matrix for a given step
        snapshot_file = os.path.join(self.directories["snapshots"], f"snapshot_nodes_{self.num_nodes}_edges_{self.num_connections}_step_{step}.h5")
        with h5py.File(snapshot_file, "w") as h5file:
            h5file.create_dataset("activities", data=activities)
            h5file.create_dataset("adjacency_matrix", data=adjacency_matrix)
            h5file.create_dataset("successful_rewirings", data=successful_rewirings)    # TODO I think this can be calculated post-run at some point, but need permission
        self.logger.info(f"Saved snapshot for step {step} to {snapshot_file}")

    def save_network_image(self, visualization, step):
        image_path = os.path.join(self.directories["images"], f"network_nodes_{self.num_nodes}_edges_{self.num_connections}_step_{step}.jpg")
        visualization.fig.savefig(image_path)
        self.logger.info(f"Saved network visualization for step {step} to {image_path}")

    ### Post-Run Metrics Calculation ###

    def post_run_output(self):
        if not os.path.exists(self.directories["snapshots"]):
            print("No snapshots available for post-run outputs.")
            return

        self.output_metrics_from_snapshots()
        self.logger.info("Generated summary metrics.")

        self.output_line_graph(metrics=["Clustering Coefficient", "Average Path Length"], title=f"CC and APL, {self.num_nodes} Nodes, {self.num_connections} Connections", ylabel="Value")
        self.logger.info("Generated CC and APL graph.")

    def output_metrics_from_snapshots(self):
        previous_adjacency_matrix = None
        previous_cluster_assignments = None

        with open(self.metrics_file_path, mode="w", newline="") as csv_file:
            # Define the CSV writer
            writer = None

            for step, snapshot_file in self._get_sorted_snapshots():
                self.logger.info(f"Analyzing snapshot: {snapshot_file}")
                snapshot_data = self._load_snapshot(snapshot_file)

                # Calculate Metrics
                metrics = self._compute_metrics(step, snapshot_data, previous_adjacency_matrix, previous_cluster_assignments)

                # Update temporal states
                previous_adjacency_matrix = snapshot_data["adjacency_matrix"]
                previous_cluster_assignments = metrics["Cluster Membership"]

                # Lazy initialization of the writer, so all the fields are known when it's created
                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=metrics.keys())
                    writer.writeheader()
                writer.writerow(metrics)

        self.logger.info(f"Summary metrics saved to {self.metrics_file_path}")

    def _get_sorted_snapshots(self):
        # Retrieve snapshots sorted by step number.
        snapshot_files = [file for file in os.listdir(self.directories["snapshots"]) if file.endswith(".h5")]
        sorted_snapshots = sorted(
            [(int(file.split("_")[file.split("_").index("step") + 1].split(".")[0]), file)
             for file in snapshot_files],
            key=lambda x: x[0]  # Sort by step number
        )
        return sorted_snapshots

    def _load_snapshot(self, snapshot_file):
        with h5py.File(os.path.join(self.directories["snapshots"], snapshot_file), "r") as h5file:
            return {
                "activities": h5file["activities"][:],
                "adjacency_matrix": h5file["adjacency_matrix"][:],
                "successful_rewirings": h5file["successful_rewirings"][()]
            }

    def _compute_metrics(self, step, snapshot_data, previous_adjacency_matrix, previous_cluster_assignments):
        adjacency_matrix_nx = nx.from_numpy_array(snapshot_data["adjacency_matrix"])
        cluster_assignments = self.calculator.detect_communities(adjacency_matrix_nx)
        return {
            "Step": step,
            "Clustering Coefficient": self.calculator.calculate_clustering_coefficient(adjacency_matrix_nx),
            "Average Path Length" : self.calculator.calculate_average_path_length(adjacency_matrix_nx),
            # "Assortativity" : self.calculator.calculate_assortativity(adjacency_matrix_nx),
            # "Betweenness Centrality" : self.calculator.calculate_betweenness_centrality(adjacency_matrix_nx)
            # "Modularity" : self.calculator.calculate_modularity(adjacency_matrix_nx),

            ### Rewiring Metrics ###
            "Rewirings (interval)" : snapshot_data["successful_rewirings"],
            "Rewiring Chance" : self.calculator.calculate_rewiring_chance(snapshot_data["adjacency_matrix"], snapshot_data["activities"]),
            "Edge Persistence" : self.calculator.calculate_edge_persistence(snapshot_data["adjacency_matrix"], previous_adjacency_matrix),

            ### Cluster Metrics ###
            "Cluster Membership" : cluster_assignments,
            "Cluster Count" : np.max(cluster_assignments) + 1,
            "Cluster Membership Stability" : self.calculator.calculate_cluster_membership_stability(cluster_assignments, previous_cluster_assignments),
            "Cluster Size Variance" : self.calculator.calculate_cluster_size_variance(cluster_assignments),
        }

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

    @staticmethod
    def aggregate_metrics(root_dir, starting_step = 500_000, snapshot_output_filepath=None, run_level_output_filepath=None):
        """
        Aggregates metrics from all metrics_summary_nodes_{num_nodes}_edges_{num_edges}.csv files
        in subfolders of the specified root directory into a single CSV file.
        """
        if snapshot_output_filepath is None:
            snapshot_output_filepath = os.path.join(root_dir, "aggregated_snapshot_metrics.csv")
        if run_level_output_filepath is None:
            run_level_output_filepath = os.path.join(root_dir, "run_level_metrics.csv")

        snapshot_data = []
        run_level_data = []

        for subfolder_path in Output._get_subfolders(root_dir):
            for file_path in Output._get_metric_files(subfolder_path):
                num_nodes, num_edges = Output._extract_node_edge_info(file_path)
                df = pd.read_csv(file_path)

                df["Nodes"] = num_nodes
                df["Edges"] = num_edges

                # Append snapshot line, with "Nodes" and "Edges" as the leftmost columns
                snapshot_data.append(df[["Nodes", "Edges"] + [col for col in df.columns if col not in ["Nodes", "Edges"]]])

                # Append run-level metrics
                run_metrics = Output._compute_run_level_metrics(df=df, starting_step=starting_step, num_nodes=num_nodes, num_edges=num_edges)
                run_level_data.append(run_metrics)

        # Combine snapshot-level data and save to CSV
        if snapshot_data:
            snapshot_df = pd.concat(snapshot_data, ignore_index=True)
            snapshot_df.to_csv(snapshot_output_filepath, index=False)
            print("Snapshot data aggregated")

        # Combine run-level metrics and save to CSV
        if run_level_data:
            run_level_df = pd.DataFrame(run_level_data)
            run_level_df.to_csv(run_level_output_filepath, index=False)
            print("Run-level data output")

    @staticmethod
    def _get_subfolders(root_dir):
        return [os.path.join(root_dir, subfolder) for subfolder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subfolder))]

    @staticmethod
    def _get_metric_files(subfolder_path):
        """Retrieve all metric files in a subfolder."""
        return [
            os.path.join(subfolder_path, file) for file in os.listdir(subfolder_path)
            if file.startswith("summary_metrics_") and file.endswith(".csv")
        ]

    @staticmethod
    def _extract_node_edge_info(file_path):
        """Extract num_nodes and num_edges from the file name."""
        try:
            parts = os.path.basename(file_path).split("_")
            num_nodes = int(parts[3])
            num_edges = int(parts[5].split(".")[0])
            return num_nodes, num_edges
        except (IndexError, ValueError) as e:
            raise ValueError(f"Failed to extract node/edge info from {file_path}: {e}")

    @staticmethod
    def _compute_run_level_metrics(df:DataFrame, starting_step, num_nodes, num_edges):
        """Compute aggregated run-level metrics."""
        # Filter out rows where step is less than starting_step
        df = df[df["Step"] >= starting_step]

        run_metrics = {
            "Nodes": num_nodes,
            "Edges": num_edges,
            "Mean CC": df["Clustering Coefficient"].mean(),
            "StdDev CC": df["Clustering Coefficient"].std(),
            "Max CC": df["Clustering Coefficient"].max(),
            "Min CC": df["Clustering Coefficient"].min(),
            "Cluster Count Mean": df["Cluster Count"].mean(),
            "Cluster Count Min": df["Cluster Count"].min(),
            "Cluster Count Max": df["Cluster Count"].max(),
            "Cluster Count StdDev": df["Cluster Count"].std(),
            "Amplitude CC": df["Clustering Coefficient"].max() - df["Clustering Coefficient"].min(),
            "Mean Rewiring Chance": df["Rewiring Chance"].mean(),
            "StdDev Rewiring Chance": df["Rewiring Chance"].std(),
            "Mean Edge Persistence": df["Edge Persistence"].mean(),
            "StdDev Edge Persistence": df["Edge Persistence"].std(),
            "Mean Rewirings (interval)": df["Rewirings (interval)"].mean(),
            "StdDev Rewirings (interval)": df["Rewirings (interval)"].std(),
        }

        # Fourier analysis on Clustering Coefficient (CC)
        try:
            from scipy.signal import periodogram
            cc_values = df["Clustering Coefficient"].to_numpy()
            frequencies, power = periodogram(cc_values)
            run_metrics["Dominant Frequency"] = frequencies[np.argmax(power)]
            run_metrics["Spectral Power"] = np.sum(power)
        except Exception:
            run_metrics["Dominant Frequency"] = None
            run_metrics["Spectral Power"] = None

        return run_metrics