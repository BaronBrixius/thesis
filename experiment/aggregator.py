import os
import pandas as pd
import logging

def aggregate_metrics(root_dirs, output_filename="aggregated_metrics.csv"):
    """Merges all `metrics.csv` files into one, adding Seed, Nodes, and Edges from file paths."""
    # Ensure root_dirs is a list (support single string input)
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]
    
    output_filepath = os.path.join(root_dirs[0], output_filename)
    logging.info(f"Aggregating snapshot metrics from {root_dirs[0]} into {output_filepath}")

    columns = None
    with open(output_filepath, "w", newline="", encoding="utf-8") as outfile:
        for root_dir in root_dirs:
            for dirpath, _, filenames in os.walk(root_dir):
                variables = _extract_variables_from_path(dirpath)  # Extract metadata

                for file in filenames:
                    if "summary_metrics" in file or file == "metrics.csv":
                        # Read metrics file
                        file_path = os.path.join(dirpath, file)
                        logging.info(f"Processing {file_path}")

                        df = pd.read_csv(file_path)
                        
                        # Add extracted metadata (Seed, Nodes, Edges)
                        for var, val in variables.items():
                            df[var] = val

                        # Write to output file (handle header only once)
                        if columns is None:
                            columns = list(df.columns)
                            df.to_csv(outfile, index=False, header=True)
                        else:
                            df = df.reindex(columns=columns, fill_value=None)  # Align columns
                            df.to_csv(outfile, index=False, header=False)

    logging.info(f"Aggregated metrics saved to {output_filepath}")

def _extract_variables_from_path(path):
    """Extracts Seed, Nodes, Edges from directory names (e.g., `seed_5/nodes_200/edges_3000`)."""
    variables = {}
    for folder in path.split(os.sep):
        if "_" in folder:
            try:
                var, val = folder.split("_", 1)
                variables[var.capitalize()] = int(val)
            except ValueError:
                continue
    return variables