import os
import pandas as pd
import logging

def aggregate_metrics(root_dir, output_filename="aggregated_metrics.csv"):
    """Merges all `metrics.csv` files into one, adding Seed, Nodes, and Edges from file paths."""
    output_filepath = os.path.join(root_dir, output_filename)
    logging.info(f"Aggregating snapshot metrics from {root_dir} into {output_filepath}")

    existing_scenarios = set()
    # If aggregated_metrics.csv exists, load it and find existing scenarios
    if os.path.exists(output_filepath):
        for chunk in pd.read_csv(output_filepath, usecols=["Seed", "Nodes", "Edges"], chunksize=500_000):
            logging.info("parsin...")
            existing_scenarios.update(zip(chunk["Seed"], chunk["Nodes"], chunk["Edges"]))
        file_mode = "a"
        write_header = False
    else:
        file_mode = "w"
        write_header = True

    with open(output_filepath, file_mode, newline="", encoding="utf-8") as outfile:
        for dirpath, _, filenames in os.walk(root_dir):
            variables = _extract_variables_from_path(dirpath)  # Extract metadata

            # If we already have this scenario in the existing CSV, skip it
            scenario = (variables.get("Seed", None), variables.get("Nodes", None), variables.get("Edges", None))
            if scenario in existing_scenarios:
                continue

            for file in filenames:
                if "summary_metrics" in file:
                    # Read metrics file
                    file_path = os.path.join(dirpath, file)
                    logging.info(f"Processing {file_path}")
                    df = pd.read_csv(file_path)

                    # Add extracted metadata (Seed, Nodes, Edges)
                    for var, val in variables.items():
                        df[var] = val

                    # Write to output file (handle header only once)
                    df.to_csv(outfile, index=False, header=write_header)
                    write_header = False

                    # Mark this scenario as processed
                    existing_scenarios.add(scenario)

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