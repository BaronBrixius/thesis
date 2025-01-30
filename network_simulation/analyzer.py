import logging
import os
import pandas as pd
import csv

class PostRunAnalyzer:
    def __init__(self, project_dir):
        self.logger = logging.getLogger(__name__)
        self.project_dir = project_dir

    def aggregate_metrics(self, root_dir, snapshot_output_filepath=None, run_level_output_filepath=None):
        """
        Aggregates metrics from all metrics_summary_nodes_{num_nodes}_edges_{num_edges}.csv files
        in subfolders of the specified root directory into a single CSV file, processing each file one at a time.
        """
        snapshot_output_filepath = self.prepare_file(snapshot_output_filepath, os.path.join(root_dir, "aggregated_snapshot_metrics.csv"))
        run_level_output_filepath = self.prepare_file(run_level_output_filepath, os.path.join(root_dir, "run_level_metrics.csv"))

        # Prepare for run-level metrics
        run_writer = None
        with open(run_level_output_filepath, mode='w', newline='', encoding='utf-8') as run_level_file:
            for dirpath, _, filenames in os.walk(root_dir):
                variables = self._extract_variables_from_path(dirpath)
                for file in filenames:
                    if file.startswith("summary_metrics") and file.endswith(".csv"):
                        file_path = os.path.join(dirpath, file)
                        self.logger.info(f"Processing {file_path}")

                        try:
                            # Read file
                            df = self.parse_file_to_df(file_path, variables)
                            # Write snapshot-level metrics
                            self._write_snapshot_metrics(df, snapshot_output_filepath)
                            # Write run-level metrics
                            run_writer = self._write_run_metrics(df, run_writer, run_level_file)

                        except Exception as e:
                            self.logger.error(f"Error processing file {file_path}: {e}")
                            continue

    def prepare_file(self, filepath, default_filepath):
        if filepath is None:
            filepath = default_filepath
        if os.path.exists(filepath):
            os.remove(filepath)

        return filepath

    def parse_file_to_df(self, file_path, variables_to_add):
        df = pd.read_csv(file_path)
        for var, val in variables_to_add.items():
            df[var] = val  # Add extracted variables as columns

        # Computed columns
        cluster_sizes = df['Cluster Sizes'].apply(eval)
        df['Ideal Edges'] = cluster_sizes.apply(lambda cluster_sizes: sum((size * (size - 1)) // 2 for size in cluster_sizes.values()))
        df['Structure'] = cluster_sizes.apply(lambda cluster_sizes: ",".join(map(str, sorted(cluster_sizes.values()))))

        df['Delta Cluster Count'] = df['Cluster Count'].diff()

        df['Intra-cluster Edge Ratio'] = df['Intra-cluster Edges'] / df['Edges']
        df['Inter-cluster Edges'] = df['Edges'] - df['Intra-cluster Edges']
        df['Intra-cluster Edge Ratio Delta'] = df['Intra-cluster Edge Ratio'].diff()
        df['Cluster Size Variance Delta'] = df['Cluster Size Variance'].diff()

        return df

    def _write_snapshot_metrics(self, df, snapshot_output_filepath):
        """Writes the snapshot-level metrics to the specified CSV file."""
        with open(snapshot_output_filepath, mode='a', newline='', encoding='utf-8') as snapshot_file:
            writer = csv.DictWriter(snapshot_file, fieldnames=df.columns)
            if snapshot_file.tell() == 0:  # Write header only if file is empty
                writer.writeheader()
            df.to_csv(snapshot_file, index=False, header=False)

    def _write_run_metrics(self, df, run_writer, run_level_file):
        """Compute run-level metrics and write them to the run-level output file."""

        # Convert step to millions and group by relevant columns
        df["Step (Millions)"] = ((df["Step"] - 1) // 1_000_000 + 1).clip(lower=0).astype(int)
        grouped = df.groupby(["Seed", "Family", "Edges", "Step (Millions)"])

        # Aggregate metrics for each group
        aggregated = grouped.agg({
            "Seed": "first",
            "Family": "first",
            "Edges": "first",
            "Step (Millions)": "first",
            "Cluster Count": "mean",
            "Intra-cluster Edges": "mean",
            "Intra-cluster Edge Ratio": "mean",
            "Intra-cluster Edge Ratio Delta": "mean",
            "Inter-cluster Edges": "mean",
            "Clustering Coefficient": ["mean", "std"],
            "Average Path Length": ["mean", "std"],
            "Average Cluster Density Weighted": "mean",
            "Cluster Size Variance": "mean",
            "Cluster Size Variance Delta": "mean",
            "Rewirings (intra_cluster)": "mean",
            "Rewirings (inter_cluster_change)": "mean",
            "Rewirings (inter_cluster_same)": "mean",
            "Rewirings (intra_to_inter)": "mean",
            "Rewirings (inter_to_intra)": "mean",
            "SBM Entropy Normalized": "mean",
        })

        # Add computed metrics
        aggregated["Nodes"] = 300
        aggregated["Density"] = aggregated["Edges"] / (aggregated["Nodes"] * (aggregated["Nodes"] - 1) / 2)
        aggregated["Cluster Count Round"] = aggregated["Cluster Count"].round()
        aggregated["Cluster Count DeciRound"] = aggregated["Cluster Count"].round(1)

        # Flatten the aggregated data into a dictionary
        for idx, row in aggregated.iterrows():
            run_metrics = row.to_dict()

            # Write run-level metrics
            if run_writer is None:
                run_writer = csv.DictWriter(run_level_file, fieldnames=run_metrics.keys())
                run_writer.writeheader()

            run_writer.writerow(run_metrics)

        return run_writer

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
