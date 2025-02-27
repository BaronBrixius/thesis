import os
import pandas as pd
import logging

def analyze_metrics(root_dir, aggregated_metrics_file="aggregated_metrics.csv", output_filename="analysis.csv"):
    """Computes aggregated run-level metrics from the snapshot file."""
    output_filepath = os.path.join(root_dir, output_filename)
    logging.info(f"Aggregating run-level metrics from {aggregated_metrics_file} to {output_filepath}")

    df = pd.read_csv(os.path.join(root_dir, aggregated_metrics_file))
    df = _parse_metrics(df)  # Apply transformations after merging all snapshots

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

    # Save results
    aggregated.to_csv(output_filepath, index=False)
    logging.info(f"Run-level analysis saved to {output_filepath}")

def _parse_metrics(df):
    """Processes a DataFrame, computing additional derived metrics for run-level analysis."""
    df['Intra-Community Edge Ratio'] = df['Intra-Community Edges'] / df['Edges']
    df['Inter-Community Edges'] = df['Edges'] - df['Intra-Community Edges']
    df['Intra-Community Edge Ratio Delta'] = df['Intra-Community Edge Ratio'].diff()
    df['Community Size Variance Delta'] = df['Community Size Variance'].diff()
    return df