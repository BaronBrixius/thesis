import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict

def generate_scatterplots(root_dir, input_filename="processed_data.csv", output_dir="scatterplots"):
    """
    Generates and saves stylized scatterplots for Clustering Coefficient and Average Path Length vs. Edges,
    including color-coded versions by Community Count.
    """
    input_filepath = os.path.join(root_dir, input_filename)
    output_dir = os.path.join(root_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_filepath)
    df_filtered = df[df["Step (Millions)"] == 10]
    num_edges = df_filtered["Edges"]

    vertical_lines = generate_colors_for_lines([19900, 9900, 6567, 4900, 3900, 3233, 2757, 2400, 2122, 1900])

    scatter_variables = [
        ("SBM Entropy Normalized_mean", "SBM Entropy Normalized", (0, 5), (0, 20100)),
        ("Community Count_mean", "Community Count", (0, 10), (0, 20100)),
        ("Clustering Coefficient_mean", "Clustering Coefficient", (0, 1), (0, 20100)),
        ("Average Path Length_mean", "Average Path Length", (0, 3), (0, 20100)),
        ("Intra-Community Edge Ratio_mean", "Intra-Community Edge Ratio", (0, 1), (1500, 11000)),
        ("Weighted Average Community Density_mean", "Weighted Community Density", (0, 1), (1500, 11000)),
        ("Community Size Variance_mean", "Community Size Variance", (0, 600), (1500, 11000)),
        ("Community Density Variance_mean", "Community Density Variance", (0, .025), (1500, 11000)),
        ("Community Density Variance_last", "Community Density Variance (last)", (0, .025), (1500, 11000)),
        ("Community Density Variance_min", "Community Density Variance (min)", (0, .025), (1500, 11000)),
    ]

    for metric_col, label, y_lim, x_lim in scatter_variables:
        scatterplot_variable_vs_edges(num_edges, df_filtered[metric_col], label, output_dir, vertical_lines, df_filtered["Community Count_mean"], y_lim, x_lim)

def generate_colors_for_lines(vertical_edges):
    """Automatically assigns evenly spaced colors from a colormap for vertical lines."""
    cmap = cm.get_cmap("rainbow", len(vertical_edges))  # Generate evenly spaced colors
    color_map = {edge: mcolors.rgb2hex(cmap(i)) for i, edge in enumerate(vertical_edges)}
    return color_map

def scatterplot_variable_vs_edges(x, y, ylabel, output_dir, vertical_lines=None, color_data=None, y_limit=None, x_limit=None):
    """Creates and saves scatterplots of a variable vs. Edges, with and without Community Count coloring and vertical lines."""
    # Plot without color
    filename = f"{ylabel.replace(' ', '_').lower()}.png"
    _plot_scatter(x, y, ylabel, output_dir, filename, y_limit=y_limit, x_limit=x_limit)
    
    # Plot with Community Count coloring
    filename_colored = f"{ylabel.replace(' ', '_').lower()}_colored.png"
    _plot_scatter(x, y, ylabel, output_dir, filename_colored, color=color_data, colorbar_label="Communities", y_limit=y_limit, x_limit=x_limit)
    
    # Plot with Community Count coloring and vertical lines
    filename_colored_ideal = f"{ylabel.replace(' ', '_').lower()}_colored_ideal.png"
    _plot_scatter(x, y, ylabel, output_dir, filename_colored_ideal, color=color_data, colorbar_label="Communities", vertical_lines=vertical_lines, y_limit=y_limit, x_limit=x_limit)


def _plot_scatter(x, y, ylabel, output_dir, filename, color=None, colorbar_label=None, y_limit=None, x_limit=None, vertical_lines=None):
    """Helper function to create and save a stylized scatterplot with optional vertical lines."""
    plt.figure(figsize=(12, 6))

    plt.style.use("dark_background")
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
        plt.clim(1, 10)  # Ensure consistent color range
    
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
    df_filtered = df[df["Step"] == 10_000_000]  # Only use step 10,000,000

    # Extract node degrees and community membership
    row = df_filtered.iloc[0]
    degrees = np.fromstring((row["Node Degrees"]).strip("[]"), sep=' ', dtype=int) / 2 # Divide by 2 to account for adjacency matrix being doubled
    communities = np.fromstring((row["Community Membership"]).strip("[]"), sep=' ', dtype=int)

    # Generate uncolored frequency chart
    plt.figure(figsize=(10, 6))
    plt.style.use("dark_background")
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    plt.bar(unique_degrees, counts, color='white')
    plt.xlabel("Degrees")
    plt.ylabel("Frequency")
    plt.title("Node Degree Distribution")
    plt.grid(True, linestyle='dotted', alpha=0.5)
    save_path = os.path.join(output_dir, "degree_distribution_uncolored.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Saved: {save_path}")
    
    # Group degrees by community
    community_degree_counts = defaultdict(lambda: defaultdict(int))
    for deg, com in zip(degrees, communities):
        community_degree_counts[com][deg] += 1  # Increment frequency

    # Prepare data for stacked bar plot
    unique_degrees = sorted(set(degrees))
    community_ids = sorted(community_degree_counts.keys())
    
    # Create a color map for communities
    cmap = cm.get_cmap("rainbow", len(community_ids))
    colors = {com: cmap(i) for i, com in enumerate(community_ids)}

    # Stack values for each community
    bottom_stack = np.zeros(len(unique_degrees))
    plt.figure(figsize=(10, 6))
    plt.style.use("dark_background")

    for com in community_ids:
        counts = [community_degree_counts[com].get(deg, 0) for deg in unique_degrees]
        plt.bar(unique_degrees, counts, bottom=bottom_stack, color=colors[com], label=f"Community {com}")
        bottom_stack += np.array(counts)

    # Labels and formatting
    plt.xlabel("Degrees")
    plt.ylabel("Frequency")
    plt.title("Node Degree Distribution (Colored by Community)")
    plt.grid(True, linestyle='dotted', alpha=0.5)

    # Save and display
    save_path = os.path.join(output_dir, "degree_distribution_colored.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved: {save_path}")

def scatterplot_clustering_vs_edges(input_file, output_dir=None):
    """Creates a scatter plot of Clustering Coefficient vs. Edges, colored by Step."""
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_file)

    plt.figure(figsize=(12, 6))
    plt.style.use("dark_background")

    # Scatter plot, using Step as color
    scatter = plt.scatter(df["Edges"], df["Clustering Coefficient"], c=df["Step"], 
                          cmap="inferno", alpha=0.6, edgecolors='none', s=2)

    # Formatting
    plt.xlabel("Edges", fontsize=14, color="white")
    plt.ylabel("Clustering Coefficient", fontsize=14, color="white")
    plt.xticks(color="white")
    plt.yticks(color="white")

    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Step", fontsize=12, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")

    plt.ylim((0, 1))
    plt.xlim((0, 20100))

    # Save the figure
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

    # Plot 1 "longlongtime.png"
    df_cluster = df[(df["Edges"] >= 6000) & (df["Edges"] <= 8000)].groupby(["Edges", "Step_Millions"], as_index=False)["Clustering Coefficient"].mean()

    # Create the figure and style
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Distinct color per Edges value
    all_edges = sorted(df_cluster["Edges"].unique())
    cmap = cm.get_cmap("rainbow", len(all_edges))

    for i, e in enumerate(all_edges):
        subset = df_cluster[df_cluster["Edges"] == e]
        subset = subset.sort_values("Step_Millions")

        # Plot lines
        ax.plot(
            subset["Step_Millions"],
            subset["Clustering Coefficient"],
            color=cmap(i),
            label=f"Edges={int(e)}",
            linewidth=2,
            alpha=0.8
        )

    ax.set_xlabel("Steps (Millions)", fontsize=14, color="white")
    ax.set_ylabel("Clustering Coefficient", fontsize=14, color="white")
    ax.tick_params(colors="white")
    ax.set_title("Clustering Coefficient vs. Steps (Millions) by Edges", color="white")
    ax.legend(loc="center right", fontsize=9)
    plt.ylim((0, 1))
    plt.tight_layout()

    # Save the figure
    output_filename = os.path.join(root_dir, "longlongtime.png")
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_filename}")

    # Plot 2 "longlongrewiring.png"
    df_8000 = df[df["Edges"] == 8000].copy()
    df_8000 = df_8000.groupby("Step_Millions", as_index=False).agg({
        "Clustering Coefficient": "mean",
        "Rew_intra_to_inter_mean": "mean",
        "Rew_inter_to_intra_mean": "mean"
    })
    df_8000 = df_8000.sort_values("Step_Millions")

    plt.style.use("dark_background")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left Y-axis: Rewiring (intra->inter) and Rewiring (inter->intra)
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

    # Right Y-axis: Clustering Coefficient
    ax2 = ax1.twinx()
    ax2.plot(
        df_8000["Step_Millions"],
        df_8000["Clustering Coefficient"],
        color="blue",
        linewidth=2,
        alpha=0.8,
        linestyle="--",
        label="Clustering Coefficient"
    )
    ax2.set_ylabel("Clustering Coefficient", fontsize=14, color="white")
    ax2.tick_params(axis='y', colors="white")
    ax2.set_ylim((0, 1))
    ax2.legend(loc=(0.825, 0.725), fontsize=9)


    plt.title(f"8000 Edges: Clustering and Rewiring", color="white")
    plt.tight_layout()

    # Save second plot
    output_filename2 = os.path.join(root_dir, "longlongrewiring.png")
    plt.savefig(output_filename2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_filename2}")
