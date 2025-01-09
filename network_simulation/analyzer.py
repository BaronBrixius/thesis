import logging
import os
import pandas as pd
import csv


class PostRunAnalyzer:
    def __init__(self, project_dir):
        self.logger = logging.getLogger(__name__)
        self.project_dir = project_dir

    def aggregate_metrics(self, root_dir, starting_step=2_000_000, snapshot_output_filepath=None, run_level_output_filepath=None):
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
                        self.logger.debug(f"Processing {file_path}")

                        try:
                            # Read file
                            df = self.parse_file_to_df(file_path, variables)
                            # Write snapshot-level metrics
                            self._write_snapshot_metrics(df, snapshot_output_filepath)
                            # Write run-level metrics
                            run_writer = self._write_run_metrics(df, variables, starting_step, run_writer, run_level_file)

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
        df['Cluster Sizes'] = df['Cluster Sizes'].apply(eval)   # ensure formatting
        df['Ideal Edges'] = df['Cluster Sizes'].apply(lambda cluster_sizes: sum((size * (size - 1)) // 2 for size in cluster_sizes))
        df['Structure'] = df['Cluster Sizes'].apply(lambda cluster_sizes: ",".join(map(str, sorted(cluster_sizes.values()))))

        return df

    def _write_snapshot_metrics(self, df, snapshot_output_filepath):
        """Writes the snapshot-level metrics to the specified CSV file."""
        with open(snapshot_output_filepath, mode='a', newline='', encoding='utf-8') as snapshot_file:
            writer = csv.DictWriter(snapshot_file, fieldnames=df.columns)
            if snapshot_file.tell() == 0:  # Write header only if file is empty
                writer.writeheader()
            df.to_csv(snapshot_file, index=False, header=False)

    def _write_run_metrics(self, df, variables, starting_step, run_writer, run_level_file):
        """Compute run-level metrics for a single file and write them to the run-level output file."""
        df = df[df["Step"] >= starting_step]
        run_metrics = {
            "Seed": variables.get("Seed", None),
            "Nodes": variables.get("Nodes", None),
            "Edges": variables.get("Edges", None),
            "Density": round(variables.get("Edges", 0) / (variables.get("Nodes", 0) * (variables.get("Nodes", 0) - 1) / 2), 3),
            "Cluster Count Round": round(df["Cluster Count"].mean()),
            "Cluster Count DeciRound": round(df["Cluster Count"].mean(), 1),
        }

        # Add summary stats for relevant columns
        columns_to_summarize = [
                "Clustering Coefficient",
                "Average Path Length",
                "Rewiring Chance",
                "Cluster Count",
                "Cluster Membership Stability",
                "Average Cluster Size",
                "Average Cluster Density",
                "Cluster Size Variance",
                "Rewirings (intra_cluster)",
                "Rewirings (inter_cluster_change)",
                "Rewirings (inter_cluster_same)",
                "Rewirings (intra_to_inter)",
                "Rewirings (inter_to_intra)",
                "SBM Entropy",
                # "SBM Mean Posterior",
                # "SBM StdDev Posterior",
                # "SBM Best Posterior"
            ]

        for column in columns_to_summarize:
            if column in df.columns:
                run_metrics[f"{column} Mean"] = df[column].mean()
                run_metrics[f"{column} StdDev"] = df[column].std()
                run_metrics[f"{column} Max"] = df[column].max()
                run_metrics[f"{column} Min"] = df[column].min()

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
