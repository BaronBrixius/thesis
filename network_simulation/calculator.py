import networkx as nx
import numpy as np
from scipy.signal import periodogram
from sklearn.metrics.cluster import adjusted_rand_score
from network_simulation.utils import start_timing, stop_timing
from cdlib import algorithms

class Calculator:
    def __init__(self):
        pass

    ## Individual Metric Calculation Methods ##
    
    def calculate_clustering_coefficient(self, graph):
        """
        Clustering Coefficient (CC): Tendency of nodes to form tightly knit groups (triangles).
        """
        return nx.average_clustering(graph)

    def calculate_average_path_length(self, graph):
        """
        Average Path Length (APL): Average shortest path length between all pairs of nodes in the network.
        """
        try:
            return nx.average_shortest_path_length(graph)
        except nx.NetworkXError:
            return None  # Disconnected graph

    def calculate_rewiring_chance(self, adjacency_matrix, activities):
        """
        Rewiring Chance: Ratio of nodes that are not connected to their most similar node in the network.
        """
        num_nodes = activities.shape[0]
        activity_diff = np.abs(activities[:, np.newaxis] - activities[np.newaxis, :])               # Compute pairwise differences, ignoring self
        np.fill_diagonal(activity_diff, np.inf)
        most_similar_node = np.argmin(activity_diff, axis=1)                                        # Find the most similar node for each node
        not_connected = [adjacency_matrix[i, most_similar_node[i]] == 0 for i in range(num_nodes)]  # Check if each node is connected to its most similar node
        return np.mean(not_connected)                       # Calculate the rewiring chance as the fraction of nodes not connected to their most similar node

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
        Network Entropy: Degree distribution's unpredictability.
        """
        degrees = np.sum(adjacency_matrix, axis=1)
        prob = degrees / degrees.sum()
        return -np.sum(prob * np.log2(prob + 1e-10))  # Avoid log(0).

    def calculate_rich_club_coefficient(self, graph):
        """
        Rich-Club Coefficient: Tendency of high-degree nodes to form tightly interconnected subgraphs.
        """
        return nx.rich_club_coefficient(graph, normalized=False)    # normalization=True gives divide by zero errors in the code for generating the random graph (weird)

    def calculate_edge_persistence(self, current_adjacency, previous_adjacency, num_connections=None):
        """
        Edge Persistence: Ratio of existing edges that also existed in the previous snapshot.
        """
        if previous_adjacency is None:
            return None
        if num_connections is None:
            num_connections = np.sum(previous_adjacency > 0)
        overlap = np.sum((current_adjacency > 0) & (previous_adjacency > 0))
        return overlap / num_connections if num_connections > 0 else 0

    def detect_communities(self, adjacency_matrix, previous_assignments=None):
        if previous_assignments is not None:
            num_nodes = len(adjacency_matrix)
            initial_membership = [-1] * num_nodes
            for cluster_id, cluster in enumerate(previous_assignments):
                for node in cluster:
                    if node < num_nodes:  # Ensure node index is valid
                        initial_membership[node] = cluster_id
        else:
            initial_membership = None

        # Run Leiden algorithm
        clustering_result = algorithms.leiden(
            nx.from_numpy_array(adjacency_matrix),
            initial_membership=initial_membership
        )

        return clustering_result.communities


    def calculate_cluster_membership_stability(self, current_assignments, previous_assignments):
        """
        Cluster Membership Stability: Similarity between cluster assignments across time steps.
        """
        if previous_assignments is None:
            return None
        return adjusted_rand_score(previous_assignments, current_assignments)

    def calculate_cluster_size_variance(self, cluster_assignments):
        """Cluster Size Variance: Variability in cluster sizes."""
        _, counts = np.unique(cluster_assignments, return_counts=True)
        return np.var(counts)

    def calculate_intra_cluster_density(self, adjacency_matrix, cluster):
        cluster_nodes = list(cluster)
        subgraph = adjacency_matrix[np.ix_(cluster_nodes, cluster_nodes)]
        num_possible_edges = len(cluster_nodes) * (len(cluster_nodes) - 1) / 2
        return subgraph.sum() / (2 * num_possible_edges)

    # Fourier Analysis
    def calculate_amplitude_of_oscillations(self, values):
        """Amplitude of Oscillations: Max range of metric values."""
        return np.max(values) - np.min(values)

    def calculate_clustering_coefficient_spectral(self, cc_values):
        """Fourier Analysis of CC: Dominant frequency and spectral power."""
        frequencies, power = periodogram(cc_values)
        dominant_frequency = frequencies[np.argmax(power)]
        spectral_power = np.sum(power)
        return {"Dominant Frequency": dominant_frequency, "Spectral Power": spectral_power}

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