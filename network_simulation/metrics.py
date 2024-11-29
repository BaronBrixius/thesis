import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics.cluster import adjusted_rand_score

class Metrics:
    def __init__(self):
        pass

    def calculate_all(self, adjacency_matrix):
        graph = nx.from_numpy_array(adjacency_matrix)
        metrics = {
            "Clustering Coefficient": self.calculate_clustering_coefficient(graph),
            "Average Path Length": self.calculate_average_path_length(graph),
            "Modularity": self.calculate_modularity(graph),
            "Assortativity": self.calculate_assortativity(graph),
            "Betweenness Centrality": self.calculate_betweenness_centrality(graph),
            "Efficiency": self.calculate_efficiency(graph),
            "Robustness": self.calculate_robustness(graph, adjacency_matrix),
            "Network Entropy": self.calculate_network_entropy(adjacency_matrix),
            "Rich-Club Coefficient": self.calculate_rich_club_coefficient(graph),
            "Hierarchical Metrics": self.calculate_hierarchical_metrics(graph),
            "Recurrence Quantification": self.calculate_recurrence_quantification(graph),
        }
        return metrics

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

    def calculate_efficiency(self, graph):
        """
        Efficiency:
        How efficiently information is exchanged in the network.
        """
        return nx.global_efficiency(graph)

    def calculate_robustness(self, graph, adjacency_matrix):
        """
        Robustness:
        The proportion of nodes in the largest connected component of the graph. Indicates the network's resilience to fragmentation.
        """
        largest_cc = max(nx.connected_components(graph), key=len)
        return len(largest_cc) / adjacency_matrix.shape[0]

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
        return 0# nx.rich_club_coefficient(graph, normalized=True)

    def calculate_hierarchical_metrics(self, graph):
        """
        Hierarchical Metrics:
        """
        #levels = nx.graph_clique_number(graph)
        return None # {"Max Clique Size": levels}

    def calculate_recurrence_quantification(self, graph):
        """
        Recurrence Quantification:
        """
        return {"Recurrence Rate": None}

    # Temporal Metrics

    def calculate_temporal_modularity(self, current_modularity, previous_modularity):
        """
        Temporal Modularity:
        Change in modularity between consecutive time steps.
        """
        if previous_modularity is None:
            return None
        return current_modularity - previous_modularity

    def calculate_edge_turnover(self, current_adjacency, previous_adjacency):
        """
        Edge Turnover:
        Ratio of persistent edges to the total number of edges over time.
        """
        overlap = np.sum((previous_adjacency > 0) & (current_adjacency > 0))
        total_edges = np.sum((previous_adjacency > 0) | (current_adjacency > 0))
        return overlap / total_edges if total_edges > 0 else 0

    def calculate_cluster_membership_stability(self, current_assignments, previous_assignments):
        """
        Cluster Membership Stability:
        Similarity between cluster assignments across time steps.
        """
        if previous_assignments is None:
            return None
        return adjusted_rand_score(previous_assignments, current_assignments)

    # Community Detection Helper

    def detect_communities(self, graph):
        """
        Assigns nodes to communities for modularity and cluster stability calculations.
        """
        clusters = nx.algorithms.community.louvain_communities(graph)

        cluster_assignments = {}
        for i, community in enumerate(clusters):
            for node in community:
                cluster_assignments[node] = i
        return np.array([cluster_assignments[node] for node in graph.nodes])
