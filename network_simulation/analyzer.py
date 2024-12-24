import csv
import logging
import os
import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame
import h5py
import matplotlib.pyplot as plt
from network_simulation.metrics import Metrics

class PostRunAnalyzer:
    def __init__(self, project_dir):
        self.project_dir = project_dir

    def aggregate_metrics(self, root_dir, starting_step=500_000, snapshot_output_filepath=None, run_level_output_filepath=None):
        """
        Aggregates metrics from all metrics_summary_nodes_{num_nodes}_edges_{num_edges}.csv files
        in subfolders of the specified root directory into a single CSV file.
        """
        if snapshot_output_filepath is None:
            snapshot_output_filepath = os.path.join(root_dir, "aggregated_snapshot_metrics.csv")
        if run_level_output_filepath is None:
            run_level_output_filepath = os.path.join(root_dir, "run_level_metrics.csv")

        folder_data = []

        for dirpath, dirnames, filenames in os.walk(root_dir):
            variables = self._extract_variables_from_path(dirpath)
            for file in filenames:
                if file.endswith(".csv") and file.startswith("summary_metrics"):
                    file_path = os.path.join(dirpath, file)
                    df = pd.read_csv(file_path)

                    # Add extracted variables as columns
                    for var, val in variables.items():
                        df[var] = val

                    folder_data.append(df)

        if folder_data:
            aggregated_df = pd.concat(folder_data, ignore_index=True)
            aggregated_df.to_csv(snapshot_output_filepath, index=False)
            print(f"Aggregated snapshot metrics saved to {snapshot_output_filepath}")

            run_level_data = self._compute_run_level_aggregations(aggregated_df, starting_step)
            run_level_data.to_csv(run_level_output_filepath, index=False)
            print(f"Aggregated run-level metrics saved to {run_level_output_filepath}")

    def _compute_run_level_aggregations(self, aggregated_df: DataFrame, starting_step):
        """
        Compute run-level aggregations from the combined dataframe.
        """
        def summary_stats(group, column_name):
            """Helper function to compute min, mean, max, and std for a given column."""
            return {
                f"Mean {column_name}": group[column_name].mean(),
                f"StdDev {column_name}": group[column_name].std(),
                f"Max {column_name}": group[column_name].max(),
                f"Min {column_name}": group[column_name].min(),
            }

        run_level_data = []
        grouped = aggregated_df.groupby(["Seed", "Nodes", "Edges"])

        for (seed, nodes, edges), group in grouped:
            group = group[group["Step"] >= starting_step]
            run_metrics = {
                "Seed": seed,
                "Nodes": nodes,
                "Edges": edges,
            }

            columns_to_summarize = [
                "Clustering Coefficient",
                "Average Path Length",
                "Rewiring Chance",
                "Rewirings (intra_cluster)",
                "Rewirings (inter_cluster_change)",
                "Rewirings (inter_cluster_same)",
                "Rewirings (intra_to_inter)",
                "Rewirings (inter_to_intra)",
                # "Edge Persistence",
            ]

            for column in columns_to_summarize:
                if column in group.columns:
                    run_metrics.update(summary_stats(group, column))

            if "Clustering Coefficient" in group.columns:
                run_metrics["Amplitude CC"] = (group["Clustering Coefficient"].max() - group["Clustering Coefficient"].min())

            run_level_data.append(run_metrics)

        return pd.DataFrame(run_level_data)

    def _extract_variables_from_path(self, path):
        """
        Extract variables and values from folder names in the path.
        Example: seed_5/nodes_200/edges_3000 -> {'Seed': 5, 'Nodes': 200, 'Edges': 3000}
        """
        variables = {}
        for folder in path.split(os.sep):
            if "_" in folder:
                try:
                    var, val = folder.split("_", 1)
                    variables[var.capitalize()] = int(val)
                except ValueError:
                    continue
        return variables