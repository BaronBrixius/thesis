import logging
import os
import pandas as pd
from network_simulation.csvwriter import CSVWriter

class PostRunAnalyzer:
    def __init__(self, project_dir):
        self.logger = logging.getLogger(__name__)
        self.project_dir = project_dir

    def aggregate_metrics(self, root_dir, snapshot_output_filepath="aggregated_metrics.csv", run_level_output_filepath="analysis.csv"):
        """
        Aggregates metrics from all metrics_summary_nodes_{num_nodes}_edges_{num_edges}.csv files
        in subfolders of the specified root directory into a single CSV file, processing each file one at a time.
        """
        snapshot_writer = CSVWriter(root_dir, snapshot_output_filepath, mode="a")
        run_writer = CSVWriter(root_dir, run_level_output_filepath, mode="a")

        for dirpath, _, filenames in os.walk(root_dir):
            variables = self._extract_variables_from_path(dirpath)
            for file in filenames:
                if file == "metrics.csv":
                    file_path = os.path.join(dirpath, file)
                    self.logger.info(f"Processing {file_path}")

                    df = self._parse_file_to_df(file_path, variables)

                    # Write snapshot-level metrics
                    for record in df.to_dict(orient="records"):
                        snapshot_writer.write_metrics_line(record)

                    # Write run-level metrics
                    aggregated = self._aggregate_metrics(df)
                    for record in aggregated.to_dict(orient="records"):
                        run_writer.write_metrics_line(record)

        snapshot_writer.close()
        run_writer.close()

    def _parse_file_to_df(self, file_path, variables_to_add):
        df = pd.read_csv(file_path)
        for var, val in variables_to_add.items():
            df[var] = val  # Add extracted variables as columns

        # Computed columns
        # community_sizes = df['Community Sizes'].apply(eval)
        # df['Ideal Edges'] = community_sizes.apply(lambda community_sizes: sum((size * (size - 1)) // 2 for size in community_sizes.values()))
        # df['Structure'] = community_sizes.apply(lambda community_sizes: ",".join(map(str, sorted(community_sizes.values()))))

        df['Intra-Community Edge Ratio'] = df['Intra-Community Edges'] / df['Edges']
        df['Inter-Community Edges'] = df['Edges'] - df['Intra-Community Edges']
        df['Intra-Community Edge Ratio Delta'] = df['Intra-Community Edge Ratio'].diff()
        df['Community Size Variance Delta'] = df['Community Size Variance'].diff()

        return df

    def _aggregate_metrics(self, df):
        """Aggregate metrics by millions of steps."""

        # Convert step to millions and group by relevant columns
        df["Step (Millions)"] = ((df["Step"] - 1) // 1_000_000 + 1).clip(lower=0)

        # Aggregate metrics for each group
        aggregated = df.groupby(["Seed", "Nodes", "Edges", "Step (Millions)"]).agg({
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
        }).reset_index()

        # Rounded versions of community count let me do some quick and dirty bucketing/comparisons, may remove for final
        aggregated["Community Count Round"] = aggregated["Community Count"].round()
        aggregated["Community Count DeciRound"] = aggregated["Community Count"].round(1)

        return aggregated

    def _extract_variables_from_path(self, path):
        """Extract values from directory names in the path (e.g., `seed_5/nodes_200/edges_3000`)."""
        variables = {}
        for folder in path.split(os.sep):
            if "_" in folder:
                try:
                    var, val = folder.split("_", 1)
                    variables[var.capitalize()] = int(val)
                except ValueError:
                    continue
        return variables
