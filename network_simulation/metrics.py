import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
from sklearn.metrics.cluster import adjusted_rand_score

class Metrics:
    def __init__(self):
        pass

    def calculate_all(self, adjacency_matrix, activities):
        graph = nx.from_numpy_array(adjacency_matrix)
        metrics = {
            "Clustering Coefficient": self.calculate_clustering_coefficient(graph),
            "Average Path Length": self.calculate_average_path_length(graph),
            "Modularity": self.calculate_modularity(graph),
            "Assortativity": self.calculate_assortativity(graph),
            "Betweenness Centrality": self.calculate_betweenness_centrality(graph),
            "Efficiency": self.calculate_efficiency(graph),
            "Network Entropy": self.calculate_network_entropy(adjacency_matrix),
            "Rich-Club Coefficient": self.calculate_rich_club_coefficient(graph),
            "Rewiring Chance": self.calculate_rewiring_chance(adjacency_matrix, activities), 
        }

        hierarchical_metrics = self.calculate_cliques(graph)
        if hierarchical_metrics:
            for key, value in hierarchical_metrics.items():
                metrics[str(key)] = value

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

    def calculate_efficiency(self, graph):
        """
        Efficiency:
        How efficiently information is exchanged in the network.
        Average of multiplicative inverse of the shortest path distance between each pair of nodes
        """
        return nx.global_efficiency(graph)

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

        # Maximum clique size
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

    # Temporal Metrics

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

    def calculate_cluster_membership_stability(self, current_assignments, previous_assignments):
        """
        Cluster Membership Stability:
        Similarity between cluster assignments across time steps.
        """
        if previous_assignments is None:
            return None
        return adjusted_rand_score(previous_assignments, current_assignments)

    # Community Detection Helper
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
