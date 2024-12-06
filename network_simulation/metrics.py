import networkx as nx
import numpy as np
from scipy.signal import periodogram
from sklearn.metrics.cluster import adjusted_rand_score

class Metrics:
    def __init__(self):
        pass

    ## Individual Metric Calculation Methods ##
    
    def calculate_clustering_coefficient(self, graph):
        """
        Clustering Coefficient (CC):
        Tendency of nodes to form tightly knit groups (triangles).
        """
        return nx.average_clustering(graph)

    def calculate_average_path_length(self, graph):
        """
        Average Path Length (APL):
        Average shortest path length between all pairs of nodes in the network.
        """
        try:
            return nx.average_shortest_path_length(graph)
        except nx.NetworkXError:
            return None  # Disconnected graph

    def calculate_rewiring_chance(self, adjacency_matrix, activities):
        """
        Rewiring Chance:
        Ratio of nodes that are not connected to their most similar node in the activity space.
        """
        num_nodes = activities.shape[0]
        not_connected_count = 0

        for i in range(num_nodes):
            # Compute similarity with all other nodes
            activity_diff = np.abs(activities - activities[i])
            activity_diff[i] = np.inf  # Ignore self-similarity

            # Find the most similar node
            most_similar_node = np.argmin(activity_diff)

            # Check if not connected
            if adjacency_matrix[i, most_similar_node] == 0:
                not_connected_count += 1

        # Calculate and return the ratio
        return not_connected_count / num_nodes

    def calculate_modularity(self, graph):
        """
        Modularity:
        Strength of division of the graph into clusters. Higher modularity indicates stronger community structures.
        """
        communities = nx.algorithms.community.louvain_communities(graph)
        return nx.algorithms.community.quality.modularity(graph, communities)

    def calculate_assortativity(self, graph):
        """
        Assortativity:
        Correlation between the degrees of connected nodes.
        Positive value: Nodes tend to connect to nodes with similar degree.
        Negative value: High-degree nodes connect to low-degree nodes.
        """
        return nx.degree_assortativity_coefficient(graph)

    def calculate_betweenness_centrality(self, graph):
        """
        Betweenness Centrality:
        How often a node appears on the shortest paths between other nodes.
        """
        betweenness = nx.betweenness_centrality(graph)
        return np.mean(list(betweenness.values()))

    def calculate_network_entropy(self, adjacency_matrix):
        """
        Network Entropy:
        Degree distribution's unpredictability.
        """
        degrees = np.sum(adjacency_matrix, axis=1)
        prob = degrees / degrees.sum()
        return -np.sum(prob * np.log2(prob + 1e-10))  # Avoid log(0).

    def calculate_rich_club_coefficient(self, graph):
        """
        Rich-Club Coefficient:
        Tendency of high-degree nodes to form tightly interconnected subgraphs.
        """
        return nx.rich_club_coefficient(graph, normalized=False)    # normalization=True gives divide by zero errors in the code for generating the random graph (weird)

    def calculate_cliques(self, graph):
        """
        Hierarchical Metrics:
        Analyzes the hierarchical structure of the network, focusing on cliques and nested communities.
        """
        # Find all cliques in the graph
        cliques = list(nx.find_cliques(graph))
        max_clique_size = max(len(clique) for clique in cliques)

        # Clique distribution
        clique_distribution = {}
        for clique in cliques:
            size = len(clique)
            clique_distribution[size] = clique_distribution.get(size, 0) + 1

        return {
            "Max Clique Size": max_clique_size,
            "Clique Distribution": clique_distribution,
        }

    # Edge Persistence Metrics
    def calculate_edge_persistence(self, current_adjacency, previous_adjacency):
        """
        Edge Persistence:
        Ratio of persistent edges to the total number of edges over time.
        """
        if previous_adjacency is None:
            return None
        overlap = np.sum((previous_adjacency > 0) & (current_adjacency > 0))
        total_edges = np.sum((previous_adjacency > 0) | (current_adjacency > 0))
        return overlap / total_edges if total_edges > 0 else 0

    def calculate_edge_turnover_rate(self, current_adjacency, previous_adjacency):
        """Edge Turnover Rate: Fraction of edges that disappear or reappear between snapshots."""
        if previous_adjacency is None:
            return None
        new_edges = np.sum((current_adjacency > 0) & (previous_adjacency == 0))
        removed_edges = np.sum((previous_adjacency > 0) & (current_adjacency == 0))
        total_edges = np.sum((previous_adjacency > 0) | (current_adjacency > 0))
        return (new_edges + removed_edges) / total_edges if total_edges > 0 else 0

    def calculate_edge_recurrence(self, current_adjacency, previous_adjacency):
        """Edge Recurrence: Fraction of previously removed edges that reappear."""
        if previous_adjacency is None:
            return None
        removed_edges = (previous_adjacency > 0) & (current_adjacency == 0)
        reappeared_edges = np.sum((current_adjacency > 0) & removed_edges)
        total_removed_edges = np.sum(removed_edges)
        return reappeared_edges / total_removed_edges if total_removed_edges > 0 else 0

    # Cluster Metrics
    def detect_communities(self, adjacency_matrix):
        """
        Assigns nodes to communities for modularity and cluster stability calculations.
        """
        graph = nx.from_numpy_array(adjacency_matrix)
        clusters = nx.algorithms.community.louvain_communities(graph)

        cluster_assignments = {}
        for i, community in enumerate(clusters):
            for node in community:
                cluster_assignments[node] = i
        return np.array([cluster_assignments[node] for node in graph.nodes])

    def calculate_cluster_membership_stability(self, current_assignments, previous_assignments):
        """Cluster Membership Stability: Similarity between cluster assignments across time steps."""
        if previous_assignments is None:
            return None
        return adjusted_rand_score(previous_assignments, current_assignments)

    def calculate_cluster_size_variance(self, cluster_assignments):
        """Cluster Size Variance: Variability in cluster sizes."""
        _, counts = np.unique(cluster_assignments, return_counts=True)
        return np.var(counts)

    def calculate_cluster_formation_rate(self, current_assignments, previous_assignments):
        """Cluster Formation Rate: Number of new clusters formed since the last snapshot."""
        if previous_assignments is None:
            return None
        current_clusters = set(current_assignments)
        previous_clusters = set(previous_assignments)
        new_clusters = current_clusters - previous_clusters
        return len(new_clusters)

    def calculate_cluster_lifespan(self, cluster_assignments, step):
        """Cluster Lifespan: Tracks the duration (in steps) of clusters."""
        # Requires external tracking of clusters over time. Placeholder implementation.
        return None  # Can be filled with logic if cluster history is tracked externally.

    # Activity Metrics
    def calculate_activity_spread(self, adjacency_matrix, activities):
        """Activity Spread: Average activity difference across edges."""
        total_diff = 0
        edge_count = 0
        for i in range(adjacency_matrix.shape[0]):
            for j in range(i + 1, adjacency_matrix.shape[0]):
                if adjacency_matrix[i, j] > 0:
                    total_diff += abs(activities[i] - activities[j])
                    edge_count += 1
        return total_diff / edge_count if edge_count > 0 else None

    def calculate_activity_similarity_distribution(self, adjacency_matrix, activities):
        """Activity Similarity Distribution: Distribution of activity differences across edges."""
        similarities = []
        for i in range(adjacency_matrix.shape[0]):
            for j in range(i + 1, adjacency_matrix.shape[0]):
                if adjacency_matrix[i, j] > 0:
                    similarities.append(abs(activities[i] - activities[j]))
        return np.array(similarities)

    # Advanced Fourier Analysis
    def calculate_amplitude_of_oscillations(self, values):
        """Amplitude of Oscillations: Max range of metric values."""
        return np.max(values) - np.min(values)

    def calculate_clustering_coefficient_spectral(self, cc_values):
        """Fourier Analysis of CC: Dominant frequency and spectral power."""
        frequencies, power = periodogram(cc_values)
        dominant_frequency = frequencies[np.argmax(power)]
        spectral_power = np.sum(power)
        return {"Dominant Frequency": dominant_frequency, "Spectral Power": spectral_power}

    # Global Metrics
    def calculate_cliques(self, graph):
        """Clique Metrics: Maximum clique size and clique distribution."""
        cliques = list(nx.find_cliques(graph))
        max_clique_size = max(len(clique) for clique in cliques)
        clique_distribution = {}
        for clique in cliques:
            size = len(clique)
            clique_distribution[size] = clique_distribution.get(size, 0) + 1
        return {"Max Clique Size": max_clique_size, "Clique Distribution": clique_distribution}

    def calculate_shortest_path_spectral(self, adjacency_matrix):
        """Shortest Path Spectral Metrics: Fourier analysis of path length fluctuations."""
        graph = nx.from_numpy_array(adjacency_matrix)
        try:
            apl_values = nx.average_shortest_path_length(graph)
        except nx.NetworkXError:
            apl_values = None
        if apl_values is not None:
            apl_values = np.array([apl_values])
            frequencies, power = periodogram(apl_values)
            return {"Dominant Frequency": frequencies[np.argmax(power)], "Spectral Power": np.sum(power)}
        return {"Dominant Frequency": None, "Spectral Power": None}

    # Helper Methods
    def summarize_metric(self, values):
        """Summarizes a metric over the entire run."""
        return {
            "Mean": np.mean(values),
            "StdDev": np.std(values),
            "Max": np.max(values),
            "Min": np.min(values),
        }