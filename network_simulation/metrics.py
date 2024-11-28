import networkx as nx
import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path

class Metrics:
    def __init__(self):
        # Initialize with default metrics
        self.metrics = {
            "CC": self.calculate_clustering_coefficient,
            "APL": self.calculate_average_path_length,
            "Modularity": self.calculate_modularity,
            "Assortativity": self.calculate_assortativity,
            "Betweenness Centrality": self.calculate_betweenness_centrality,
            "Efficiency": self.calculate_efficiency,
            "Degree Distribution": self.calculate_degree_distribution,
            "Rich-Club Coefficient": self.calculate_rich_club_coefficient,
            "Robustness": self.calculate_robustness,
            "Network Entropy": self.calculate_network_entropy,
            "Hierarchical Metrics": self.calculate_hierarchical_metrics,
            "Recurrence Quantification": self.calculate_recurrence_quantification,
        }

    def calculate_all(self, adjacency_matrix):
        results = {}
        for name, func in self.metrics.items():
            results[name] = func(adjacency_matrix)
        return results

    ## Metric Calculation Methods ##

    def calculate_clustering_coefficient(self, adjacency_matrix):
        graph = nx.from_numpy_array(adjacency_matrix)
        return nx.average_clustering(graph)

    def calculate_average_path_length(self, adjacency_matrix):
        graph = nx.from_numpy_array(adjacency_matrix)
        try:
            return nx.average_shortest_path_length(graph)
        except nx.NetworkXError:
            return None  # Disconnected graph/breakup

    def calculate_modularity(self, adjacency_matrix):
        graph = nx.from_numpy_array(adjacency_matrix)
        communities = {node: 0 for node in graph.nodes}
        return nx.algorithms.community.quality.modularity(graph, [list(communities.keys())])

    def calculate_assortativity(self, adjacency_matrix):
        graph = nx.from_numpy_array(adjacency_matrix)
        return nx.degree_assortativity_coefficient(graph)

    def calculate_betweenness_centrality(self, adjacency_matrix):
        graph = nx.from_numpy_array(adjacency_matrix)
        betweenness = nx.betweenness_centrality(graph)
        return np.mean(list(betweenness.values()))

    def calculate_efficiency(self, adjacency_matrix):
        graph = nx.from_numpy_array(adjacency_matrix)
        return nx.global_efficiency(graph)

    def calculate_degree_distribution(self, adjacency_matrix):
        degrees = np.sum(adjacency_matrix, axis=1)
        return degrees

    def calculate_rich_club_coefficient(self, adjacency_matrix):
        graph = nx.from_numpy_array(adjacency_matrix)
        return None #nx.rich_club_coefficient(graph, normalized=True)

    def calculate_robustness(self, adjacency_matrix):
        graph = nx.from_numpy_array(adjacency_matrix)
        largest_cc = max(nx.connected_components(graph), key=len)
        return len(largest_cc) / adjacency_matrix.shape[0]

    def calculate_network_entropy(self, adjacency_matrix):
        degrees = np.sum(adjacency_matrix, axis=1)
        prob = degrees / degrees.sum()
        return -np.sum(prob * np.log2(prob + 1e-10))  # Avoid log(0).

    def calculate_hierarchical_metrics(self, adjacency_matrix):
        # Placeholder for hierarchical metrics
        return None

    def calculate_recurrence_quantification(self, adjacency_matrix):
        # Placeholder for recurrence quantification
        return None
