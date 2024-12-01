import logging
import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from network_simulation.metrics import Metrics

class Output:
    def __init__(self, base_dir, runtime_outputs=None, post_run_outputs=None):
        self.base_dir = os.path.join("output", base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

        self.snapshots_dir = os.path.join(self.base_dir, "state_snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)

        self.network_images_dir = os.path.join(self.base_dir, "images")
        os.makedirs(self.network_images_dir, exist_ok=True)

        self.plots_dir = os.path.join(self.base_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        self.metrics_file_path = os.path.join(self.base_dir, "metrics_summary.csv")

        self.metrics = Metrics()

        self.runtime_outputs = runtime_outputs or {
            "state_snapshot",
            "network_image",
        }

        self.post_run_outputs = post_run_outputs or {
            "metrics_summary",
            "average_metrics",
            "cc_apl_graph",
        }

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        self.logger = logging.getLogger(__name__)

    ### Runtime Snapshot Methods ###

    def output_state_snapshot(self, step, activities, adjacency_matrix):
        # Save activities and adjacency matrix for a given step
        if "state_snapshot" not in self.runtime_outputs:
            return
        snapshot_file = os.path.join(self.snapshots_dir, f"snapshot_{step}.h5")
        with h5py.File(snapshot_file, "w") as h5file:
            h5file.create_dataset("activities", data=activities)
            h5file.create_dataset("adjacency_matrix", data=adjacency_matrix)
        self.logger.info(f"Saved snapshot for step {step} to {snapshot_file}")

    def output_network_image(self, visualization, step):
        if "network_image" not in self.runtime_outputs:
            return
        image_path = os.path.join(self.network_images_dir, f"network_{step}.jpg")
        visualization.fig.savefig(image_path, format="jpg")
        self.logger.info(f"Saved network visualization for step {step} to {image_path}")

    ### Post-Run Metrics Calculation ###

    def post_run_output(self, last_steps=200000, xlim=None, ylim=None, ylim_cc=None, ylim_apl=None):
        if not os.path.exists(self.snapshots_dir):
            print("No snapshots available for post-run outputs.")
            return

        self.output_metrics_from_snapshots()
        self.logger.info("Generated metrics summar.")

        self.output_average_metrics(last_steps=last_steps, xlim=xlim, ylim=ylim)
        self.logger.info("Generated average metrics.")

        self.output_line_graph(metrics=["Clustering Coefficient", "Average Path Length"], title="CC and APL Over Time", ylabel="Value")
        self.logger.info("Generated CC and APL graph.")

    def output_metrics_from_snapshots(self):
        metrics_summary = []
        previous_adjacency_matrix = None
        previous_cluster_assignments = None

        for snapshot_file in sorted(os.listdir(self.snapshots_dir)):
            if not snapshot_file.endswith(".h5"):
                continue

            step = int(snapshot_file.split("_")[1].split(".")[0])
            snapshot_path = os.path.join(self.snapshots_dir, snapshot_file)

            with h5py.File(snapshot_path, "r") as h5file:
                adjacency_matrix = h5file["adjacency_matrix"][:]

            # Calculate metrics
            metrics = {"Step": step}    # So "Step" is on the left in the output
            metrics.update(self.metrics.calculate_all(adjacency_matrix))

            # Detect communities for cluster stability
            current_cluster_assignments = self.metrics.detect_communities(adjacency_matrix)

            # Calculate temporal metrics
            if previous_adjacency_matrix is not None:
                metrics["Edge Turnover"] = self.metrics.calculate_edge_turnover(
                    adjacency_matrix, previous_adjacency_matrix
                )
            else:
                metrics["Edge Turnover"] = None

            if previous_cluster_assignments is not None:
                metrics["Cluster Membership Stability"] = self.metrics.calculate_cluster_membership_stability(
                    current_cluster_assignments, previous_cluster_assignments
                )
            else:
                metrics["Cluster Membership Stability"] = None

            # Update temporal states
            previous_adjacency_matrix = adjacency_matrix
            previous_cluster_assignments = current_cluster_assignments

            # Add metrics to the summary
            metrics_summary.append(metrics)

        # Save metrics summary
        metrics_summary_df = pd.DataFrame(metrics_summary)

        # Ensure "Step" is the leftmost column
        cols = ["Step"] + [col for col in metrics_summary_df.columns if col != "Step"]
        metrics_summary_df = metrics_summary_df[cols]

        metrics_summary_df.to_csv(self.metrics_file_path, index=False)
        self.logger.info(f"Metrics summary saved to {self.metrics_file_path}")

    def output_average_metrics(self, last_steps=200000, xlim=None, ylim=None):
        histogram_sum = None
        bin_edges = None
        adjacency_sum = None
        count = 0
        recurrence_rates = []  # To store recurrence rates for each snapshot

        for snapshot_file in sorted(os.listdir(self.snapshots_dir)):
            if not snapshot_file.endswith(".h5"):
                continue

            step = int(snapshot_file.split("_")[1].split(".")[0])
            if step < max(0, step - last_steps):
                continue

            snapshot_path = os.path.join(self.snapshots_dir, snapshot_file)
            with h5py.File(snapshot_path, "r") as h5file:
                adjacency_matrix = h5file["adjacency_matrix"][:]
                connections_per_node = np.sum(adjacency_matrix, axis=1)

                # Update adjacency sum
                adjacency_sum = (adjacency_matrix if adjacency_sum is None else adjacency_sum + adjacency_matrix)

                # Determine bin edges dynamically
                max_connections = int(connections_per_node.max()) + 1
                new_bin_edges = np.arange(0, max_connections + 1)
                if bin_edges is None or len(new_bin_edges) > len(bin_edges):
                    # Update bin edges and adjust histogram_sum to match new bins
                    if histogram_sum is not None:
                        expanded_histogram_sum = np.zeros(len(new_bin_edges) - 1)
                        expanded_histogram_sum[: len(histogram_sum)] = histogram_sum
                        histogram_sum = expanded_histogram_sum
                    bin_edges = new_bin_edges

                histogram, _ = np.histogram(connections_per_node, bins=bin_edges)
                histogram_sum = (histogram if histogram_sum is None else histogram_sum + histogram)
                count += 1

        # Calculate and save average adjacency matrix
        if adjacency_sum is not None and count > 0:
            avg_adjacency_matrix = adjacency_sum / count
            avg_matrix_path = os.path.join(self.base_dir, "average_adjacency_matrix.csv")
            np.savetxt(avg_matrix_path, avg_adjacency_matrix, delimiter=",")
            self.logger.info(f"Saved average adjacency matrix to {avg_matrix_path}")

        # Calculate and save average histogram
        if histogram_sum is not None:
            avg_histogram = histogram_sum / count
            plt.figure()
            plt.bar(bin_edges[:-1], avg_histogram, color="gray", align="center", width=1)
            plt.xlabel("Connections per Node")
            plt.ylabel("Frequency")
            plt.title("Average Connection Distribution")
            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)
            output_path = os.path.join(self.plots_dir, "connection_distribution.jpg")
            plt.savefig(output_path)
            plt.close()
            self.logger.info(f"Saved average histogram to {output_path}")

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
        metrics_graph_path = os.path.join(self.plots_dir, f"{'_'.join(metrics)}.jpg")
        plt.savefig(metrics_graph_path)
        plt.close()
        self.logger.info(f"Saved line graph for metrics {metrics} to {metrics_graph_path}")
