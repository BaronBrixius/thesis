import matplotlib.pyplot as plt
import os
import pandas as pd

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

    _plot_scatter(num_edges, df_filtered["Clustering Coefficient_mean"], "Clustering Coefficient", output_dir, "cc_vs_edges.png", style=True)
    _plot_scatter(num_edges, df_filtered["Average Path Length_mean"], "Average Path Length", output_dir, "apl_vs_edges.png", style=True, y_limit=3.0)
    _plot_scatter(num_edges, df_filtered["Clustering Coefficient_mean"], "Clustering Coefficient", output_dir, "cc_vs_edges_colored.png", color=df_filtered["Community Count_mean"], colorbar_label="Communities", style=True)
    _plot_scatter(num_edges, df_filtered["Average Path Length_mean"], "Average Path Length", output_dir, "apl_vs_edges_colored.png", color=df_filtered["Community Count_mean"], colorbar_label="Communities", style=True, y_limit=3.0)

    generate_intra_community_edge_ratio_plot(df_filtered, output_dir)
    generate_sbm_entropy_plot(df_filtered, output_dir)

def _plot_scatter(x, y, ylabel, output_dir, filename, color=None, colorbar_label=None, style=False, y_limit=None):
    """Helper function to create and save a stylized scatterplot."""
    plt.figure(figsize=(12, 6))
    
    if style:
        plt.style.use("dark_background")
        scatter = plt.scatter(x, y, c=color if color is not None else "white", cmap="rainbow" if color is not None else None, alpha=0.8, edgecolors='none', s=10)
        plt.xticks(color="white")
        plt.yticks(color="white")
        plt.xlabel("Edges", fontsize=14, color="white")
        plt.ylabel(ylabel, fontsize=14, color="white")
        plt.grid(color="gray", linestyle="dotted", linewidth=0.5)
        if color is not None:
            cbar = plt.colorbar(scatter)
            cbar.set_label(colorbar_label, fontsize=12, color="white")
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.clim(1, 10)  # Ensure consistent color range
    else:
        plt.scatter(x, y, alpha=0.6)
        plt.xlabel("Edges")
        plt.ylabel(ylabel)
        plt.grid(True)
    
    if y_limit is not None:
        plt.ylim(0, y_limit)
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def generate_intra_community_edge_ratio_plot(df, output_dir):
    """Generates a plot of Intra-Community Edge Ratio vs. Edge Count."""
    plt.figure(figsize=(12, 6))
    plt.scatter(df["Edges"], df["Intra-Community Edge Ratio_mean"], alpha=0.8, edgecolors='none', s=10, color="cyan")
    plt.xlabel("Edges")
    plt.ylabel("Intra-Community Edge Ratio")
    plt.title("Intra-Community Edge Ratio vs. Edge Count")
    plt.grid(True)
    save_path = os.path.join(output_dir, "intra_community_edge_ratio.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def generate_sbm_entropy_plot(df, output_dir):
    """Generates a plot of SBM Entropy vs. Edge Count."""
    plt.figure(figsize=(12, 6))
    plt.scatter(df["Edges"], df["SBM Entropy Normalized_mean"], alpha=0.8, edgecolors='none', s=10, color="magenta")
    plt.xlabel("Edges")
    plt.ylabel("SBM Entropy Normalized")
    plt.title("SBM Entropy vs. Edge Count")
    plt.grid(True)
    save_path = os.path.join(output_dir, "sbm_entropy_vs_edges.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
