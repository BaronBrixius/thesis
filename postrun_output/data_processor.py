import os
import pandas as pd
import logging

def process_metrics(root_dir, aggregated_metrics_file="aggregated_metrics.csv", output_filename="processed_data.csv"):
    """Processes the data from the aggregated metrics file to a new csv, ready to be plotted."""
    output_filepath = os.path.join(root_dir, output_filename)
    logging.info(f"Processing metrics from {aggregated_metrics_file} to {output_filepath}")

    first_chunk = True
    for chunk in pd.read_csv(os.path.join(root_dir, aggregated_metrics_file), chunksize=10_001):    #TODO more flexible chunksize
        df = _parse_metrics(chunk)  # Apply transformations for analysis

        # Group by millions of steps for each run
        df["Step (Millions)"] = ((df["Step"] - 1) // 1_000_000 + 1).clip(lower=0)

        # Aggregate metrics for each group
        grouped = df.groupby(["Seed", "Nodes", "Edges", "Step (Millions)"]).agg({
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

        # Flatten multi-index column names that were generated
        grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]

        # Rounded versions are useful for quick checks, may want to remove for final
        grouped["Community Count Round"] = grouped["Community Count_mean"].round()
        grouped["Community Count DeciRound"] = grouped["Community Count_mean"].round(1)

        # Save results, appending after the first chunk
        if first_chunk:
            grouped.to_csv(output_filepath, mode='w', header=True, index=False)
            first_chunk = False
        else:
            grouped.to_csv(output_filepath, mode='a', header=False, index=False)

    logging.info(f"Analysis saved to {output_filepath}")

def _parse_metrics(df):
    """Processes a DataFrame, computing additional derived metrics for analysis."""
    df['Intra-Community Edge Ratio'] = df['Intra-Community Edges'] / df['Edges']
    df['Inter-Community Edges'] = df['Edges'] - df['Intra-Community Edges']
    df['Intra-Community Edge Ratio Delta'] = df['Intra-Community Edge Ratio'].diff()
    df['Community Size Variance Delta'] = df['Community Size Variance'].diff()
    return df