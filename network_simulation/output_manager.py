import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from network_simulation.metrics import Metrics

class Output:
    def __init__(self, base_dir, runtime_outputs=None, post_run_outputs=None):
        self.base_dir = os.path.join("output", base_dir)
        self.metrics_file_path = os.path.join(self.base_dir, "metrics_summary.csv")

        self.snapshots_dir = os.path.join(self.base_dir, "state_snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)

        self.network_images_dir = os.path.join(self.base_dir, "images")
        os.makedirs(self.network_images_dir, exist_ok=True)

        self.plots_dir = os.path.join(self.base_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        self.metrics = Metrics()

        self.runtime_outputs = runtime_outputs or {
            "state_snapshot": self.output_state_snapshot,
            "network_image": self.output_network_image,
        }

        self.post_run_outputs = post_run_outputs or {
            "metrics_summary": self.output_metrics_from_snapshots,
            "histogram": self.output_histogram_connection_distribution,
            "cc_apl_graph": self.output_cc_apl_graph,
        }

    ### Runtime Snapshot Methods ###

    def output_state_snapshot(self, step, activities, adjacency_matrix):
        # Save activities and adjacency matrix for a given step
        if "state_snapshot" not in self.runtime_outputs:
            return
        snapshot_file = os.path.join(self.snapshots_dir, f"snapshot_{step}.h5")
        with h5py.File(snapshot_file, "w") as h5file:
            h5file.create_dataset("activities", data=activities)
            h5file.create_dataset("adjacency_matrix", data=adjacency_matrix)
        print(f"Saved snapshot for step {step} to {snapshot_file}")

    def output_network_image(self, visualization, step):
        image_path = os.path.join(self.network_images_dir, f"network_{step}.jpg")
        visualization.fig.savefig(image_path, format="jpg")
        print(f"Saved network visualization for step {step} to {image_path}")

    ### Post-Run Metrics Calculation ###

    def post_run_output(self, last_steps=200000, xlim=None, ylim=None, ylim_cc=None, ylim_apl=None):
        if not os.path.exists(self.snapshots_dir):
            print("No snapshots available for post-run outputs.")
            return

        # Generate metrics summary first (required for other outputs)
        if "metrics_summary" in self.post_run_outputs:
            print("Generating metrics summary...")
            self.output_metrics_from_snapshots()

        # Generate histogram if enabled
        if "histogram" in self.post_run_outputs:
            print("Generating average histogram...")
            self.output_histogram_connection_distribution(last_steps=last_steps, xlim=xlim, ylim=ylim)

        # Generate CC and APL graph if enabled
        if "cc_apl_graph" in self.post_run_outputs:
            print("Generating CC and APL graph...")
            self.output_cc_apl_graph(xlim=xlim, ylim_cc=ylim_cc, ylim_apl=ylim_apl)

    def output_metrics_from_snapshots(self):
        metrics_summary = []

        for snapshot_file in sorted(os.listdir(self.snapshots_dir)):
            if not snapshot_file.endswith(".h5"):
                continue

            step = int(snapshot_file.split("_")[1].split(".")[0])
            snapshot_path = os.path.join(self.snapshots_dir, snapshot_file)

            with h5py.File(snapshot_path, "r") as h5file:
                activities = h5file["activities"][:]
                adjacency_matrix = h5file["adjacency_matrix"][:]

            # Use MetricsManager to calculate all metrics
            metrics = {"Step": step}  # Add "Step" as the first entry
            metrics.update(self.metrics.calculate_all(adjacency_matrix))
            metrics_summary.append(metrics)

        # Save metrics summary to a file
        metrics_summary_df = pd.DataFrame(metrics_summary)
        metrics_summary_df.to_csv(self.metrics_file_path, index=False)
        print(f"Metrics summary saved to {self.metrics_file_path}")

    def output_histogram_connection_distribution(self, last_steps=200000, xlim=None, ylim=None):
        histogram_sum = None
        bin_edges = None
        count = 0

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
                histogram_sum = (
                    histogram if histogram_sum is None else histogram_sum + histogram
                )
                count += 1

        # Calculate average histogram
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
            print(f"Saved average histogram to {output_path}")

    def output_cc_apl_graph(self, xlim=None, ylim_cc=None, ylim_apl=None):    # Line graph of CC and APL over the run
        if not os.path.exists(self.metrics_file_path):
            print(f"No metrics summary file found at {self.metrics_file_path}")
            return

        data = pd.read_csv(self.metrics_file_path)

        # Ensure required columns exist
        if not {"Step", "CC", "APL"}.issubset(data.columns):
            print(f"Metrics summary file is missing required columns.")
            return

        plt.figure(figsize=(10, 6))
        # CC Plot
        plt.plot(
            data["Step"],
            data["CC"],
            label="Clustering Coefficient (CC)",
            color="green",
        )
        if ylim_cc:
            plt.ylim(ylim_cc)

        # APL Plot
        plt.plot(
            data["Step"],
            data["APL"],
            label="Average Path Length (APL)",
            color="blue",
        )
        if ylim_apl:
            plt.ylim(ylim_apl)

        plt.title("CC and APL Over Time")
        plt.xlabel("Step Number")
        plt.ylabel("Value")
        if xlim:
            plt.xlim(xlim)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        metrics_graph_path = os.path.join(self.plots_dir, "cc_apl.jpg")
        plt.savefig(metrics_graph_path)
        plt.close()
        print(f"Saved CC and APL graph to {metrics_graph_path}")
