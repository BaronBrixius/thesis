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

class PostRunAnalyzer:
    def __init__(self, project_dir):
        self.project_dir = project_dir

    def aggregate_metrics(self, root_dir, starting_step=500_000, snapshot_output_filepath=None, run_level_output_filepath=None, replace=False):
        """
        Aggregates metrics from all metrics_summary_nodes_{num_nodes}_edges_{num_edges}.csv files
        in subfolders of the specified root directory into a single CSV file.
        Handles new folder structure with `seed_X` subfolders containing `edges_X` sub-subfolders.
        """
        if snapshot_output_filepath is None:
            snapshot_output_filepath = os.path.join(root_dir, "aggregated_snapshot_metrics.csv")
        if run_level_output_filepath is None:
            run_level_output_filepath = os.path.join(root_dir, "run_level_metrics.csv")

        # Process each seed folder
        for seed_folder in self._get_subfolders(root_dir):
            seed_snapshot_output = os.path.join(seed_folder, "aggregated_snapshot_metrics.csv")
            seed_run_level_output = os.path.join(seed_folder, "run_level_metrics.csv")

            # Skip processing if output files already exist and `replace` is False
            if not replace and os.path.exists(seed_snapshot_output) and os.path.exists(seed_run_level_output):
                continue

            # Process all edges within the seed folder
            self._process_seed_folder(
                seed_folder, starting_step, seed_snapshot_output, seed_run_level_output
            )

        # Aggregate at the top level
        self._aggregate_top_level(root_dir, snapshot_output_filepath, run_level_output_filepath)

    def _process_seed_folder(self, seed_folder, starting_step, snapshot_output_path, run_level_output_path):
        """
        Process metrics within a single seed folder.
        """
        seed_snapshot_data = []
        seed_run_level_data = []

        seed = int(seed_folder.split("_")[1].split(".")[0])

        for edges_folder in self._get_subfolders(seed_folder):
            for file_path in self._get_metric_files(edges_folder):
                num_nodes, num_edges = self._extract_node_edge_info(file_path)
                df = pd.read_csv(file_path)

                df["Seed"] = seed
                df["Nodes"] = num_nodes
                df["Edges"] = num_edges

                # Append snapshot data
                seed_snapshot_data.append(df[["Seed", "Nodes", "Edges"] + [col for col in df.columns if col not in ["Seed", "Nodes", "Edges"]]])

                # Append run-level metrics
                run_metrics = self._compute_run_level_metrics(df=df, starting_step=starting_step, num_nodes=num_nodes, num_edges=num_edges, seed=seed)
                seed_run_level_data.append(run_metrics)

        # Save seed-level outputs
        if seed_snapshot_data:
            snapshot_df = pd.concat(seed_snapshot_data, ignore_index=True)
            snapshot_df.to_csv(snapshot_output_path, index=False)
            print(f"Snapshot data aggregated for {seed_folder}")

        if seed_run_level_data:
            run_level_df = pd.DataFrame(seed_run_level_data)
            run_level_df.to_csv(run_level_output_path, index=False)
            print(f"Run-level data output for {seed_folder}")

    def _aggregate_top_level(self, root_dir, snapshot_output_path, run_level_output_path):
        """
        Aggregates all seed-level results into top-level files.
        Reads seed-level aggregated files dynamically.
        """
        snapshot_data = []
        run_level_data = []

        for seed_folder in self._get_subfolders(root_dir):
            seed_snapshot_output = os.path.join(seed_folder, "aggregated_snapshot_metrics.csv")
            seed_run_level_output = os.path.join(seed_folder, "run_level_metrics.csv")

            # Read seed-level aggregated files
            if os.path.exists(seed_snapshot_output):
                snapshot_data.append(pd.read_csv(seed_snapshot_output))
            if os.path.exists(seed_run_level_output):
                run_level_data.append(pd.read_csv(seed_run_level_output))

        # Combine and save top-level results
        if snapshot_data:
            snapshot_df = pd.concat(snapshot_data, ignore_index=True)

            snapshot_df = snapshot_df.astype({"Seed": "int", "Nodes": "int", "Edges": "int", "Step": "int"})
            snapshot_df.sort_values(by=["Seed", "Nodes", "Edges", "Step"], inplace=True)

            snapshot_df.to_csv(snapshot_output_path, index=False)
            print("Snapshot data aggregated at the top level")

        if run_level_data:
            run_level_df = pd.concat(run_level_data, ignore_index=True)

            run_level_df = run_level_df.astype({"Seed": "int", "Nodes": "int", "Edges": "int"})
            run_level_df.sort_values(by=["Seed", "Nodes", "Edges"], inplace=True)

            run_level_df.to_csv(run_level_output_path, index=False)
            print("Run-level data output at the top level")

    def _get_subfolders(self, root_dir):
        """
        Retrieve subfolders within a directory.
        """
        return [
            os.path.join(root_dir, subfolder)
            for subfolder in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, subfolder))
        ]

    def _get_metric_files(self, subfolder_path):
        """
        Retrieve all metric files in a subfolder.
        """
        return [
            os.path.join(subfolder_path, file)
            for file in os.listdir(subfolder_path)
            if file.startswith("summary_metrics_") and file.endswith(".csv")
        ]

    def _extract_node_edge_info(self, file_path):
        """Extract num_nodes and num_edges from the file name."""
        try:
            parts = os.path.basename(file_path).split("_")
            num_nodes = int(parts[3])
            num_edges = int(parts[5].split(".")[0])
            return num_nodes, num_edges
        except (IndexError, ValueError) as e:
            raise ValueError(f"Failed to extract node/edge info from {file_path}: {e}")

    def _compute_run_level_metrics(self, df: DataFrame, starting_step, num_nodes, num_edges, seed):
        """Compute aggregated run-level metrics."""
        # Filter out rows where step is less than starting_step
        df = df[df["Step"] >= starting_step]

        run_metrics = {
            "Seed": seed,
            "Nodes": num_nodes,
            "Edges": num_edges,
            "Mean CC": df["Clustering Coefficient"].mean(),
            "StdDev CC": df["Clustering Coefficient"].std(),
            "Max CC": df["Clustering Coefficient"].max(),
            "Min CC": df["Clustering Coefficient"].min(),
            "Mean APL": df["Average Path Length"].mean(),
            "StdDev APL": df["Average Path Length"].std(),
            "Max APL": df["Average Path Length"].max(),
            "Min APL": df["Average Path Length"].min(),
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
