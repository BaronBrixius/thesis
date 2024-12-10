import csv
import logging
import os
import networkx as nx
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from network_simulation.calculator import Calculator

class Output:
    def __init__(self, base_dir, num_nodes=None, num_connections=None):
        self.base_dir = os.path.join("output", base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

        self.snapshots_dir = os.path.join(self.base_dir, "state_snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)

        self.network_images_dir = os.path.join(self.base_dir, "images")
        os.makedirs(self.network_images_dir, exist_ok=True)

        self.plots_dir = os.path.join(self.base_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        self.num_nodes = num_nodes
        self.num_connections = num_connections
        self.metrics_file_path = os.path.join(self.base_dir, f"summary_metrics_nodes_{self.num_nodes}_edges_{self.num_connections}.csv")

        self.calculator = Calculator()
        self.logger = self._initialize_logger()

    @staticmethod
    def _initialize_logger():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        return logging.getLogger(__name__)

    ### Runtime Snapshot Methods ###

    def save_snapshot(self, step, activities, adjacency_matrix, successful_rewirings):
        # Save activities and adjacency matrix for a given step
        snapshot_file = os.path.join(self.snapshots_dir, f"snapshot_nodes_{self.num_nodes}_edges_{self.num_connections}_step_{step}.h5")
        with h5py.File(snapshot_file, "w") as h5file:
            h5file.create_dataset("activities", data=activities)
            h5file.create_dataset("adjacency_matrix", data=adjacency_matrix)
            h5file.create_dataset("successful_rewirings", data=successful_rewirings)    # TODO I think this can be calculated post-run at some point, but need permission
        self.logger.info(f"Saved snapshot for step {step} to {snapshot_file}")

    def save_network_image(self, visualization, step):
        image_path = os.path.join(self.network_images_dir, f"network_nodes_{self.num_nodes}_edges_{self.num_connections}_step_{step}.jpg")
        visualization.fig.savefig(image_path)
        self.logger.info(f"Saved network visualization for step {step} to {image_path}")

    ### Post-Run Metrics Calculation ###

    def post_run_output(self):
        if not os.path.exists(self.snapshots_dir):
            print("No snapshots available for post-run outputs.")
            return

        self.output_metrics_from_snapshots()
        self.logger.info("Generated summary metrics.")

        self.output_line_graph(metrics=["Clustering Coefficient", "Average Path Length"], title=f"CC and APL, {self.num_nodes} Nodes, {self.num_connections} Connections", ylabel="Value")
        self.logger.info("Generated CC and APL graph.")

    def get_sorted_snapshots(self):
        # Retrieve snapshots sorted by step number.
        snapshot_files = [file for file in os.listdir(self.snapshots_dir) if file.endswith(".h5")]
        sorted_snapshots = sorted(
            [(int(file.split("_")[file.split("_").index("step") + 1].split(".")[0]), file)
             for file in snapshot_files],
            key=lambda x: x[0]  # Sort by step number
        )
        return sorted_snapshots

    def output_metrics_from_snapshots(self):
        previous_adjacency_matrix = None
        previous_cluster_assignments = None

        with open(self.metrics_file_path, mode="w", newline="") as csv_file:
            # Define the CSV writer
            writer = None

            for step, snapshot_file in self.get_sorted_snapshots():
                self.logger.info(f"Analyzing snapshot: {snapshot_file}")

                # Load snapshot data
                snapshot_path = os.path.join(self.snapshots_dir, snapshot_file)
                with h5py.File(snapshot_path, "r") as h5file:
                    activities = h5file["activities"][:]
                    adjacency_matrix = h5file["adjacency_matrix"][:]
                    successful_rewirings = h5file["successful_rewirings"][()]

                # Initialize metrics dictionary
                metrics = {"Step": step}

                # Convert adjacency matrix to NetworkX graph
                adjacency_matrix_nx = nx.from_numpy_array(adjacency_matrix)

                ### Network Connectivity and Structure Metrics ###
                metrics["Clustering Coefficient"] = self.calculator.calculate_clustering_coefficient(adjacency_matrix_nx)
                metrics["Average Path Length"] = self.calculator.calculate_average_path_length(adjacency_matrix_nx)
                metrics["Assortativity"] = self.calculator.calculate_assortativity(adjacency_matrix_nx)
                # metrics["Betweenness Centrality"] = self.calculator.calculate_betweenness_centrality(adjacency_matrix_nx)
                metrics["Modularity"] = self.calculator.calculate_modularity(adjacency_matrix_nx)

                ### Rewiring Metrics ###
                metrics["Rewirings (interval)"] = successful_rewirings
                metrics["Rewiring Chance"] = self.calculator.calculate_rewiring_chance(adjacency_matrix, activities)

                metrics["Edge Persistence"] = self.calculator.calculate_edge_persistence(adjacency_matrix, previous_adjacency_matrix)

                ### Temporal Community and Cluster Metrics ###
                cluster_assignments = self.calculator.detect_communities(adjacency_matrix_nx)
                metrics["Cluster Membership"] = cluster_assignments
                metrics["Cluster Count"] = np.max(cluster_assignments) + 1
                metrics["Cluster Membership Stability"] = self.calculator.calculate_cluster_membership_stability(cluster_assignments, previous_cluster_assignments)
                metrics["Cluster Size Variance"] = self.calculator.calculate_cluster_size_variance(cluster_assignments)

                # Update temporal states
                previous_adjacency_matrix = adjacency_matrix
                previous_cluster_assignments = cluster_assignments

                # Lazy initialization of the writer, so all the fields are known when it's created
                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=metrics.keys())
                    writer.writeheader()
                writer.writerow(metrics)

        self.logger.info(f"Summary metrics saved to {self.metrics_file_path}")

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
        metrics_graph_path = os.path.join(self.plots_dir, f"{'_'.join(metrics)}_nodes_{self.num_nodes}_edges_{self.num_connections}.jpg")
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

        for subfolder in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            for file in os.listdir(subfolder_path):
                if file.startswith("summary_metrics_nodes_") and file.endswith(".csv"):
                    # Extract num_nodes and num_edges from the file name
                    try:
                        parts = file.split("_")
                        num_nodes = int(parts[3])
                        num_edges = int(parts[5].split(".")[0])
                    except (IndexError, ValueError):
                        continue

                    # Read the metrics file and add num_nodes and num_edges columns
                    file_path = os.path.join(subfolder_path, file)
                    df = pd.read_csv(file_path)
                    if "Cluster Count" not in df.columns:
                        continue
                    df["Nodes"] = num_nodes
                    df["Edges"] = num_edges

                    # Ensure "Nodes" and "Edges" are the leftmost columns
                    columns_order = ["Nodes", "Edges"] + [col for col in df.columns if col not in ["Nodes", "Edges"]]
                    df = df[columns_order]
                    snapshot_data.append(df)

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
                        "Mean Edge Persistence": df["Edge Persistence"].mean(),
                        "StdDev Edge Persistence": df["Edge Persistence"].std(),
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

                    # Append run-level metrics to the list
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

