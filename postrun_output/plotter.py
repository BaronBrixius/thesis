import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict

def generate_scatterplots(root_dir, input_filename="processed_data.csv", output_dir="scatterplots"):
    input_filepath = os.path.join(root_dir, input_filename)
    output_dir = os.path.join(root_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_filepath)
    df_filtered = df[df["Step (Millions)"] == 10]
    num_edges = df_filtered["Edges"]

    vertical_lines = _generate_colors_for_lines([19900, 9900, 6567, 4900, 3900, 3233, 2757, 2400, 2122, 1900])
    community_count_colors = df_filtered["Community Count_mean"]

    _plot_scatter(x=num_edges, y=df_filtered["Community Count_mean"], ylabel="Community Count", output_dir=output_dir, filename="community_count.png", color=community_count_colors, colorbar_label="Communities", y_limit=(0, 10), x_limit=(0, 20100),)
    _plot_scatter(x=num_edges, y=df_filtered["Community Count_mean"], ylabel="Community Count", output_dir=output_dir, filename="community_count.png", color=community_count_colors, colorbar_label="Communities", y_limit=(0, 10), x_limit=(0, 20100),)
    _plot_scatter(x=num_edges, y=df_filtered["SBM Entropy Normalized_mean"], ylabel="SBM Entropy Normalized", output_dir=output_dir, filename="sbm_entropy_normalized.png", color=community_count_colors, colorbar_label="Communities", y_limit=(0, 5), x_limit=(0, 20100),)
    _plot_scatter(x=num_edges, y=df_filtered["Community Size Variance_mean"], ylabel="Community Size Variance", output_dir=output_dir, filename="community_size_variance.png", color=community_count_colors, colorbar_label="Communities", vertical_lines=vertical_lines, y_limit=(0, 600), x_limit=(1500, 11000),)
    _plot_scatter(x=num_edges, y=df_filtered["Community Density Variance_mean"], ylabel="Community Density Variance", output_dir=output_dir, filename="community_density_variance.png", color=community_count_colors, colorbar_label="Communities", vertical_lines=vertical_lines, y_limit=(0, 0.025), x_limit=(1500, 11000),)
    _plot_scatter(x=num_edges, y=df_filtered["Weighted Average Community Density_mean"], ylabel="Weighted Community Density", output_dir=output_dir, filename="weighted_community_density.png", color=community_count_colors, colorbar_label="Communities", vertical_lines=vertical_lines, y_limit=(0, 1), x_limit=(1500, 11000),)
    _plot_scatter(x=num_edges, y=df_filtered["Clustering Coefficient_mean"], ylabel="Clustering Coefficient", output_dir=output_dir, filename="clustering_coefficient.png", color=community_count_colors, colorbar_label="Communities", vertical_lines=vertical_lines, y_limit=(0, 1), x_limit=(0, 20100),)
    _plot_scatter(x=num_edges, y=df_filtered["Average Path Length_mean"], ylabel="Average Path Length", output_dir=output_dir, filename="average_path_length.png", color=community_count_colors, colorbar_label="Communities", vertical_lines=vertical_lines, y_limit=(0, 3), x_limit=(0, 20100),)

def _generate_colors_for_lines(vertical_edges):
    """Assigns evenly spaced colors from a colormap for vertical lines."""
    cmap = cm.get_cmap("rainbow", len(vertical_edges))
    color_map = {edge: mcolors.rgb2hex(cmap(i)) for i, edge in enumerate(vertical_edges)}
    return color_map

def _setup_figure(figsize=(12, 6)):
    plt.figure(figsize=figsize)
    plt.style.use("dark_background")

def _plot_scatter(x, y, ylabel, output_dir, filename, color=None, colorbar_label=None, y_limit=None, x_limit=None, vertical_lines=None):
    _setup_figure(figsize=(12, 6))

    scatter = plt.scatter(x, y, c=color if color is not None else "cyan", cmap="rainbow" if color is not None else None, alpha=0.8, edgecolors='none', s=2)

    plt.xticks(color="white")
    plt.yticks(color="white")
    plt.xlabel("Edges", fontsize=14, color="white")
    plt.ylabel(ylabel, fontsize=14, color="white")
    # plt.grid(color="gray", linestyle="dotted", linewidth=0.5, axis='y')

    if color is not None:
        cbar = plt.colorbar(scatter)
        cbar.set_label(colorbar_label, fontsize=12, color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.clim(1, 10)

    if y_limit is not None:
        plt.ylim(y_limit)
    if x_limit is not None:
        plt.xlim(x_limit)

    if vertical_lines:
        for edge, color in vertical_lines.items():
            plt.axvline(x=edge, color=color, linestyle='-', linewidth=1.0, alpha=0.7)
            plt.axvline(x=edge, color=color, linestyle='-', linewidth=1.0, alpha=0.1)

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def generate_degree_distribution_histogram(input_file, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_file)
    df_filtered = df[df["Step"] == 10_000_000]
    row = df_filtered.iloc[0]

    # Extract degrees & communities
    degrees = np.fromstring((row["Node Degrees"]).strip("[]"), sep=' ', dtype=int) / 2
    communities = np.fromstring((row["Community Membership"]).strip("[]"), sep=' ', dtype=int)

    # Uncolored frequency chart
    _setup_figure(figsize=(10, 6))
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    plt.bar(unique_degrees, counts, color='white')
    plt.xlabel("Degrees")
    plt.ylabel("Frequency")
    plt.title("Node Degree Distribution", color="white")
    plt.grid(True, linestyle='dotted', alpha=0.5)
    save_path = os.path.join(output_dir, "degree_distribution_uncolored.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Saved: {save_path}")
    
    # Colored by community
    community_degree_counts = defaultdict(lambda: defaultdict(int))
    for deg, com in zip(degrees, communities):
        community_degree_counts[com][deg] += 1

    unique_degrees = sorted(set(degrees))
    community_ids = sorted(community_degree_counts.keys())
    cmap = cm.get_cmap("rainbow", len(community_ids))
    colors = {com: cmap(i) for i, com in enumerate(community_ids)}

    bottom_stack = np.zeros(len(unique_degrees))
    _setup_figure(figsize=(10, 6))
    for com in community_ids:
        freq = [community_degree_counts[com].get(deg, 0) for deg in unique_degrees]
        plt.bar(unique_degrees, freq, bottom=bottom_stack, color=colors[com], label=f"Community {com}")
        bottom_stack += np.array(freq)

    plt.xlabel("Degrees")
    plt.ylabel("Frequency")
    plt.title("Node Degree Distribution (Colored by Community)", color="white")
    plt.grid(True, linestyle='dotted', alpha=0.5)
    save_path = os.path.join(output_dir, "degree_distribution_colored.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved: {save_path}")

def scatterplot_clustering_vs_edges(input_file, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_file)

    _setup_figure(figsize=(12, 6))
    scatter = plt.scatter(
        df["Edges"], 
        df["Clustering Coefficient"], 
        c=df["Step"], 
        cmap="inferno",
        alpha=0.6,
        edgecolors='none',
        s=2
    )
    plt.xlabel("Edges", fontsize=14, color="white")
    plt.ylabel("Clustering Coefficient", fontsize=14, color="white")
    plt.xticks(color="white")
    plt.yticks(color="white")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Step", fontsize=12, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")

    plt.ylim((0, 1))
    plt.xlim((0, 20100))

    save_path = os.path.join(output_dir, "clustering_vs_edges.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def long_long_plots(root_dir):
    input_csv = os.path.join(root_dir, "run_level_metrics.csv")
    df = pd.read_csv(input_csv)
    df = df.rename(columns={
        "('Seed', 'first')": "Seed",
        "('Nodes', 'first')": "Nodes",
        "('Edges', 'first')": "Edges",
        "('Step (Millions)', 'first')": "Step_Millions",
        "('Clustering Coefficient', 'mean')": "Clustering Coefficient",
        "('Clustering Coefficient', 'std')": "Clustering Coefficient_std",
        "('Rewirings (intra_to_inter)', 'mean')": "Rew_intra_to_inter_mean",
        "('Rewirings (inter_to_intra)', 'mean')": "Rew_inter_to_intra_mean",
    })

    # Plot 1
    df_cluster = df[
        (df["Edges"] >= 6000) & (df["Edges"] <= 8000)
    ].groupby(["Edges", "Step_Millions"], as_index=False)["Clustering Coefficient"].mean()

    _setup_figure(figsize=(12, 6))
    all_edges = sorted(df_cluster["Edges"].unique(), reverse=True)
    cmap = cm.get_cmap("rainbow", len(all_edges))

    for i, e in enumerate(all_edges):
        subset = df_cluster[df_cluster["Edges"] == e].sort_values("Step_Millions")
        plt.plot(
            subset["Step_Millions"],
            subset["Clustering Coefficient"],
            color=cmap(i),
            label=f"Edges={int(e)}",
            linewidth=2,
            alpha=0.8
        )

    plt.xlabel("Steps (Millions)", fontsize=14, color="white")
    plt.ylabel("Clustering Coefficient", fontsize=14, color="white")
    plt.tick_params(colors="white")
    plt.title("Clustering Coefficient vs. Steps (Millions) by Edges", color="white")
    plt.legend(loc="center right", fontsize=9)
    plt.ylim((0, 1))
    plt.tight_layout()

    output_filename = os.path.join(root_dir, "longlongtime.png")
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_filename}")

    # Plot 2
    df_8000 = df[df["Edges"] == 8000].copy()
    df_8000 = df_8000.groupby("Step_Millions", as_index=False).agg({
        "Clustering Coefficient": "mean",
        "Rew_intra_to_inter_mean": "mean",
        "Rew_inter_to_intra_mean": "mean"
    }).sort_values("Step_Millions")

    _setup_figure(figsize=(12, 6))
    ax1 = plt.gca()

    # Left Y-axis
    ax1.plot(
        df_8000["Step_Millions"],
        df_8000["Rew_intra_to_inter_mean"],
        color="orange",
        linewidth=2,
        alpha=0.8,
        label="Rewiring (intra->inter)"
    )
    ax1.plot(
        df_8000["Step_Millions"],
        df_8000["Rew_inter_to_intra_mean"],
        color="silver",
        linewidth=2,
        alpha=0.8,
        label="Rewiring (inter->intra)"
    )
    ax1.set_xlabel("Steps (Millions)", fontsize=14, color="white")
    ax1.set_ylabel("Rewirings Count", fontsize=14, color="white")
    ax1.tick_params(axis='x', colors="white")
    ax1.tick_params(axis='y', colors="white")
    ax1.set_xlim((0, 250))
    ax1.legend(loc=(0.005, 0.0625), fontsize=9)

    # Right Y-axis
    ax2 = ax1.twinx()
    ax2.plot(
        df_8000["Step_Millions"],
        df_8000["Clustering Coefficient"],
        color=cm.rainbow(0.0),  # 8000 is first (index 0) in the reversed sort
        linewidth=2,
        alpha=0.8,
        linestyle="--",
        label="Clustering Coefficient"
    )
    ax2.set_ylabel("Clustering Coefficient", fontsize=14, color="white")
    ax2.tick_params(axis='y', colors="white")
    ax2.set_ylim((0, 1))
    ax2.legend(loc=(0.825, 0.725), fontsize=9)

    plt.title("8000 Edges: Clustering and Rewiring", color="white")
    plt.tight_layout()

    output_filename2 = os.path.join(root_dir, "longlongrewiring.png")
    plt.savefig(output_filename2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_filename2}")

def real_big_plots(root_dir):
    df = pd.read_csv(os.path.join(root_dir, "metrics.csv"))

    _setup_figure(figsize=(12, 6))
    plt.plot(
        df["Step"], 
        df["Community Count"],
        color="cyan", 
        linewidth=2,
        alpha=0.8,
        label="Community Count"
    )
    plt.axhline(y=75, color="red", linestyle="--", linewidth=1)

    plt.xlabel("Step", fontsize=14, color="white")
    plt.ylabel("Community Count", fontsize=14, color="white")
    plt.tick_params(colors="white")
    plt.title("Community Count vs. Step", color="white")
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(root_dir, "real_big_community_count.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")
