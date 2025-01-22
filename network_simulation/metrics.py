from graph_tool.all import Graph, local_clustering, shortest_distance
from graph_tool.inference import PPBlockState
import networkx as nx
import numpy as np
from typing import Optional, Dict
from functools import lru_cache

class Metrics:
    def __init__(self, graph):
        self.breakup_count = 0
        self.rewirings = {
            "intra_cluster": 0,
            "inter_cluster_change": 0,
            "inter_cluster_same": 0,
            "intra_to_inter": 0,
            "inter_to_intra": 0,
        }
        self.block_state = PPBlockState(graph)

    # Runtime Tracking
    def increment_breakup_count(self):
        self.breakup_count += 1

    def increment_rewiring_count(self, pivot, from_node, to_node, step: int):
        partitions = self.block_state.get_blocks().a
        pivot_cluster = partitions[int(pivot)]
        from_cluster = partitions[int(from_node)]
        to_cluster = partitions[int(to_node)]

        if pivot_cluster == from_cluster == to_cluster:
            self.rewirings["intra_cluster"] += 1
        elif pivot_cluster != from_cluster and pivot_cluster != to_cluster:
            if from_cluster == to_cluster:
                self.rewirings["inter_cluster_same"] += 1
            else:
                self.rewirings["inter_cluster_change"] += 1
        elif pivot_cluster == from_cluster and pivot_cluster != to_cluster:
            self.rewirings["intra_to_inter"] += 1
        elif pivot_cluster != from_cluster and pivot_cluster == to_cluster:
            self.rewirings["inter_to_intra"] += 1

    def reset_rewiring_counts(self):
        """Reset all rewiring counts."""
        self.rewirings = {key: 0 for key in self.rewirings}

    ## Individual Metric Calculation Methods ##

    @staticmethod
    def calculate_clustering_coefficient(graph: Graph) -> float:
        """
        Clustering Coefficient (CC): Tendency of nodes to form tightly knit groups (triangles).
        """
        return local_clustering(graph).get_array().mean()

    @staticmethod
    def calculate_average_path_length(graph: Graph) -> Optional[float]:
        """
        Average Path Length (APL): Average shortest path length between all pairs of nodes in the network.
        """
        distances = shortest_distance(graph, directed=False)
        n = graph.num_vertices()
        # Only sum the upper triangle of the distance matrix, and double it
        ave_path_length = 2 * sum([sum(row[j + 1:]) for j, row in enumerate(distances)])/(n**2-n)
        return ave_path_length

    @staticmethod
    def calculate_rich_club_coefficients(adjacency_matrix) -> Dict[int, float]:
        return nx.rich_club_coefficient(nx.from_numpy_array(adjacency_matrix), normalized=False)

    @lru_cache(maxsize=16)
    def get_cluster_metrics(self, graph: Graph, step: int) -> Dict[str, object]:
        cluster_assignments = self.get_cluster_assignments(graph, step)
        unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
        
        # Calculate cluster sizes and densities in one pass
        cluster_sizes = {}
        intra_cluster_densities = {}
        total_nodes = graph.num_vertices()
        total_density_weight = 0

        for cluster in unique_clusters:
            cluster_nodes = np.where(cluster_assignments == cluster)[0]
            size = len(cluster_nodes)
            cluster_sizes[cluster] = size
            density = self.get_intra_cluster_density(graph, tuple(cluster_nodes))
            intra_cluster_densities[cluster] = density
            total_density_weight += size * density

        cluster_metrics = {
            "Cluster Count": len(unique_clusters),
            "Cluster Sizes": cluster_sizes,
            "Cluster Densities": intra_cluster_densities,
            "Average Cluster Density": total_density_weight / total_nodes,
            "Cluster Size Variance": np.var(counts),
            "SBM Entropy Normalized": self.block_state.entropy() / graph.num_edges(),
        }

        return cluster_metrics

    @lru_cache(maxsize=16)
    def get_cluster_assignments(self, graph: Graph, step: int):
        self.block_state = self.block_state.copy(graph)     # update the block state with the latest graph, preserves the previous community assignments

        for _ in range(5):
            entropy_delta, _, _ = self.block_state.multilevel_mcmc_sweep()            #TODO multilevel isn't really needed for us, but the regular multiflip keeps hanging. change?
            if entropy_delta == 0:
                break
        
        return self.block_state.get_blocks().a

    @staticmethod
    @lru_cache(maxsize=8)
    def get_intra_cluster_density(graph: Graph, cluster_nodes: tuple) -> float:
        graph.set_vertex_filter(graph.new_vertex_property("bool", vals=[int(v) in cluster_nodes for v in graph.vertices()]), inverted=False)
        num_edges = graph.num_edges()
        num_possible_edges = len(cluster_nodes) * (len(cluster_nodes) - 1) / 2
        graph.set_vertex_filter(None)
        return num_edges / num_possible_edges if num_possible_edges > 0 else 0
