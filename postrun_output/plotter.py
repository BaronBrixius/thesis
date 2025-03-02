
import matplotlib.pyplot as plt

def generate_scatterplots(df):
    """
    Generates scatterplots for Clustering Coefficient and Average Path Length vs. Edges,
    including color-coded versions by Community Count.
    """
    df_filtered = df[df["Step (Millions)_first"] == 10]
    num_edges = df_filtered["Intra-Community Edges_mean"]
    
    # Scatterplot for Clustering Coefficient vs. Edges
    plt.figure(figsize=(10, 6))
    plt.scatter(num_edges, df_filtered["Clustering Coefficient_mean"], alpha=0.6)
    plt.xlabel("Number of Edges (Intra-Community)")
    plt.ylabel("Clustering Coefficient")
    plt.title("Clustering Coefficient vs. Number of Edges")
    plt.grid(True)
    plt.show()

    # Scatterplot for Average Path Length vs. Edges
    plt.figure(figsize=(10, 6))
    plt.scatter(num_edges, df_filtered["Average Path Length_mean"], alpha=0.6)
    plt.xlabel("Number of Edges (Intra-Community)")
    plt.ylabel("Average Path Length")
    plt.title("Average Path Length vs. Number of Edges")
    plt.grid(True)
    plt.show()

    # Scatterplot for Clustering Coefficient colored by Community Count
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        num_edges,
        df_filtered["Clustering Coefficient_mean"],
        c=df_filtered["Community Count_mean"],
        cmap="rainbow",
        alpha=0.7
    )
    plt.colorbar(label="Community Count")
    plt.xlabel("Number of Edges (Intra-Community)")
    plt.ylabel("Clustering Coefficient")
    plt.title("Clustering Coefficient vs. Edges (Colored by Community Count)")
    plt.grid(True)
    plt.show()

    # Scatterplot for APL colored by Community Count
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        num_edges,
        df_filtered["Average Path Length_mean"],
        c=df_filtered["Community Count_mean"],
        cmap="rainbow",
        alpha=0.7
    )
    plt.colorbar(label="Community Count")
    plt.xlabel("Number of Edges (Intra-Community)")
    plt.ylabel("Average Path Length")
    plt.title("Average Path Length vs. Edges (Colored by Community Count)")
    plt.grid(True)
    plt.show()
