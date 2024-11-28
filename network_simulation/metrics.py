import numpy as np
from scipy.sparse.csgraph import shortest_path
import networkx as nx  # For betweenness centrality
from networkx import Graph, degree_assortativity_coefficient
from networkx.algorithms.community import greedy_modularity_communities, modularity

class Metrics:
    def __init__(self):
        self.metrics = {
            "Clustering Coefficient": self.calculate_clustering_coefficient,
            "Characteristic Path Length": self.calculate_characteristic_path_length,
            "Modularity": self.calculate_modularity,
            "Assortativity": self.calculate_assortativity,
            "Betweenness Centrality": self.calculate_betweenness_centrality,
            "Efficiency": self.calculate_efficiency,
            "Degree Distribution": self.calculate_degree_distribution,
            "Rich-Club Coefficient": self.calculate_rich_club_coefficient,
        }

    def calculate_metrics(self, adjacency_matrix):
        """
        Calculate all registered metrics for the given adjacency matrix.
        """
        results = {}
        for name, func in self.metrics.items():
            results[name] = func(adjacency_matrix)
        return results

    # Metric calculation methods
    def calculate_clustering_coefficient(self, adjacency_matrix):
        clustering_coefficients = []
        for i in range(adjacency_matrix.shape[0]):
            neighbors = np.where(adjacency_matrix[i] > 0)[0]
            if len(neighbors) < 2:
                clustering_coefficients.append(0)
                continue
            subgraph = adjacency_matrix[neighbors][:, neighbors]
            total_links = np.sum(subgraph) / 2  # Undirected graph
            possible_links = len(neighbors) * (len(neighbors) - 1) / 2
            clustering_coefficients.append(total_links / possible_links)
        return np.mean(clustering_coefficients)

    def calculate_characteristic_path_length(self, adjacency_matrix):
        path_lengths = shortest_path(adjacency_matrix, directed=False, unweighted=True)
        if np.isinf(path_lengths).any():
            self.breakup_count += 1
        valid_lengths = path_lengths[(path_lengths < np.inf) & (path_lengths > 0)]   # FIXME right now, upon breakup, it removes "infinite" distances then computes the average as if that were okay
        return np.mean(valid_lengths) if valid_lengths.size > 0 else np.nan

    def calculate_modularity(self, adjacency_matrix):
        graph = Graph(adjacency_matrix)
        # Assume all nodes belong to a single community for simplicity
        communities = {node: 0 for node in graph.nodes}
        return modularity(graph, [list(communities.keys())])

    def calculate_assortativity(self, adjacency_matrix):
        graph = Graph(adjacency_matrix)
        return degree_assortativity_coefficient(graph)

    def calculate_betweenness_centrality(self, adjacency_matrix):
        graph = Graph(adjacency_matrix)
        centrality = np.array(list(nx.betweenness_centrality(graph).values()))
        return np.mean(centrality)

    def calculate_efficiency(self, adjacency_matrix):
        distances = shortest_path(adjacency_matrix, directed=False, unweighted=True)
        with np.errstate(divide='ignore'):
            efficiency_matrix = 1 / distances
        efficiency_matrix[~np.isfinite(efficiency_matrix)] = 0
        n = adjacency_matrix.shape[0]
        total_efficiency = np.sum(efficiency_matrix) - n  # Exclude diagonal
        return total_efficiency / (n * (n - 1))

    def calculate_degree_distribution(self, adjacency_matrix):
        degrees = np.sum(adjacency_matrix, axis=1)
        unique, counts = np.unique(degrees, return_counts=True)
        return dict(zip(unique, counts))

    def calculate_rich_club_coefficient(self, adjacency_matrix):
        degrees = np.sum(adjacency_matrix, axis=1)
        sorted_nodes = np.argsort(degrees)[::-1]  # Sort nodes by degree
        rich_club_coefficient = []
        for k in range(1, len(degrees) + 1):
            rich_nodes = sorted_nodes[:k]
            subgraph = adjacency_matrix[rich_nodes][:, rich_nodes]
            total_links = np.sum(subgraph) / 2
            max_links = k * (k - 1) / 2
            rich_club_coefficient.append(total_links / max_links if max_links > 0 else 0)
        return rich_club_coefficient
