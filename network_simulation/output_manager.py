import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from network_simulation.metrics import Metrics

class Output:
    def __init__(self, base_dir):
        self.base_dir = os.path.join("output", base_dir)
        self.snapshots_dir = os.path.join(self.base_dir, "state_snapshots")
        self.metrics_file_path = os.path.join(self.base_dir, "metrics_summary.csv")
        self.network_images_dir = os.path.join(self.base_dir, "images")  # For network visualization images
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.network_images_dir, exist_ok=True)
        self.metrics_manager = Metrics()

    ### Runtime Snapshot Methods ###

    def save_state_snapshot(self, step, activities, adjacency_matrix):
        # Save activities and adjacency matrix for a given step
        snapshot_file = os.path.join(self.snapshots_dir, f"snapshot_{step}.h5")
        with h5py.File(snapshot_file, "w") as h5file:
            h5file.create_dataset("activities", data=activities)
            h5file.create_dataset("adjacency_matrix", data=adjacency_matrix)
        print(f"Saved snapshot for step {step} to {snapshot_file}")

    def save_network_image(self, visualization, step):
        image_path = os.path.join(self.network_images_dir, f"network_{step}.jpg")
        visualization.fig.savefig(image_path, format="jpg")
        print(f"Saved network visualization for step {step} to {image_path}")

    ### Post-Run Metrics Calculation ###

    def calculate_metrics_from_snapshots(self):
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
            metrics = self.metrics_manager.calculate_all_metrics(adjacency_matrix)
            metrics["Step"] = step
            metrics_summary.append(metrics)

        # Save metrics summary to a file
        metrics_summary_df = pd.DataFrame(metrics_summary)
        metrics_summary_df.to_csv(self.metrics_file_path, index=False)
        print(f"Metrics summary saved to {self.metrics_file_path}")

    # --- Optional Visualization of Metrics ---
    def plot_metrics_graph(self):
        """Plot metrics graph from metrics summary CSV."""
        if not os.path.exists(self.metrics_file_path):
            print(f"No metrics summary file found at {self.metrics_file_path}")
            return

        data = pd.read_csv(self.metrics_file_path)

        # Ensure required columns exist
        if "Step" not in data.columns or len(data.columns) < 2:
            print(f"Metrics summary file is missing required columns.")
            return

        plt.figure(figsize=(10, 6))
        for column in data.columns:
            if column == "Step":
                continue
            plt.plot(data["Step"], data[column], label=column)

        plt.title("Metrics Over Time")
        plt.xlabel("Step Number")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        metrics_graph_path = os.path.join(self.base_dir, "metrics_graph.png")
        plt.savefig(metrics_graph_path, dpi=300)
        plt.close()
        print(f"Metrics graph saved to {metrics_graph_path}")
