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
                    if file == "metrics.csv":
                        file_path = os.path.join(dirpath, file)
                        self.logger.info(f"Processing {file_path}")

                        # Read file
                        df = self.parse_file_to_df(file_path, variables)
                        # Write snapshot-level metrics
                        self._write_snapshot_metrics(df, snapshot_output_filepath)
                        # Write run-level metrics
                        run_writer = self._write_run_metrics(df, run_writer, run_level_file)

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
        # community_sizes = df['Community Sizes'].apply(eval)
        # df['Ideal Edges'] = community_sizes.apply(lambda community_sizes: sum((size * (size - 1)) // 2 for size in community_sizes.values()))
        # df['Structure'] = community_sizes.apply(lambda community_sizes: ",".join(map(str, sorted(community_sizes.values()))))

        df['Delta Community Count'] = df['Community Count'].diff()

        df['Intra-Community Edge Ratio'] = df['Intra-Community Edges'] / df['Edges']
        df['Inter-Community Edges'] = df['Edges'] - df['Intra-Community Edges']
        df['Intra-Community Edge Ratio Delta'] = df['Intra-Community Edge Ratio'].diff()
        df['Community Size Variance Delta'] = df['Community Size Variance'].diff()

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
        df["Step (Millions)"] = ((df["Step"] - 1) // 1_000_000 + 1).clip(lower=0)
        grouped = df.groupby(["Seed", "Nodes", "Edges", "Step (Millions)"])

        # Aggregate metrics for each group
        aggregated = grouped.agg({
            "Seed": "first",
            "Nodes": "first",
            "Edges": "first",
            "Step (Millions)": "first",
            "Clustering Coefficient": ["mean", "std"],
            "Average Path Length": ["mean", "std"],
            "Community Count": "mean",
            "Intra-Community Edges": "mean",
            "Intra-Community Edge Ratio": "mean",
            "Intra-Community Edge Ratio Delta": "mean",
            "Inter-Community Edges": "mean",
            "Community Size Variance": "mean",
            "Community Size Variance Delta": "mean",
            "SBM Entropy Normalized": "mean",
        })

        # Add computed metrics
        aggregated["Density"] = aggregated["Edges"] / (aggregated["Nodes"] * (aggregated["Nodes"] - 1) / 2)
        aggregated["Community Count Round"] = aggregated["Community Count"].round()
        aggregated["Community Count DeciRound"] = aggregated["Community Count"].round(1)

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
