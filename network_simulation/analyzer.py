import logging
import os
import pandas as pd

class PostRunAnalyzer:
    def __init__(self, project_dir):
        self.logger = logging.getLogger(__name__)
        self.project_dir = project_dir

    def aggregate_metrics(self, root_dir, snapshot_output_file="aggregated_metrics.csv", run_level_output_file="analysis.csv"):
        """
        Aggregates metrics from all metrics_summary_nodes_{num_nodes}_edges_{num_edges}.csv files
        in subfolders of the specified root directory into a single CSV file, processing each file one at a time.
        """
        snapshot_output_file = os.path.join(root_dir, snapshot_output_file)
        run_level_output_file = os.path.join(root_dir, run_level_output_file)

        # **Step 1: Efficiently merge all CSVs into snapshot file**
        # self._merge_snapshot_metrics(root_dir, snapshot_output_file)

        # **Step 2: Compute run-level metrics from merged file**
        self._aggregate_metrics(snapshot_output_file, run_level_output_file)

    def _merge_snapshot_metrics(self, root_dir, output_filepath):
        """Quickly merges all `metrics.csv` files into one, only adding Seed, Nodes, Edges from file path."""
        first_file = True
        with open(output_filepath, "w", newline="", encoding="utf-8") as outfile:
            for dirpath, _, filenames in os.walk(root_dir):
                variables = self._extract_variables_from_path(dirpath)  # Extract Seed, Nodes, Edges
                for file in filenames:
                    if file == "metrics.csv":
                        file_path = os.path.join(dirpath, file)
                        self.logger.info(f"Processing {file_path}")

                        df = pd.read_csv(file_path)
                        
                        # Add extracted metadata (Seed, Nodes, Edges)
                        for var, val in variables.items():
                            df[var] = val

                        # Write to output file
                        if first_file:
                            df.to_csv(outfile, index=False)
                            first_file = False
                        else:
                            df.to_csv(outfile, index=False, header=False)

    def _parse_file_to_df(self, file_path):
        # Read CSV file
        df = pd.read_csv(file_path)

        # Computed columns
        df['Intra-Community Edge Ratio'] = df['Intra-Community Edges'] / df['Edges']
        df['Inter-Community Edges'] = df['Edges'] - df['Intra-Community Edges']
        df['Intra-Community Edge Ratio Delta'] = df['Intra-Community Edge Ratio'].diff()
        df['Community Size Variance Delta'] = df['Community Size Variance'].diff()

        return df

    def _aggregate_metrics(self, snapshot_filepath, run_level_filepath):
        """Computes aggregated run-level metrics from the snapshot file, applying additional processing."""
        df = self._parse_file_to_df(snapshot_filepath)

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

        # Flatten multi-index column names
        aggregated.columns = ["_".join(col).strip("_") for col in aggregated.columns.values]

        # Rounded versions for quick comparisons
        aggregated["Community Count Round"] = aggregated["Community Count_mean"].round()
        aggregated["Community Count DeciRound"] = aggregated["Community Count_mean"].round(1)

        # Write final processed results
        aggregated.to_csv(run_level_filepath, index=False)

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
