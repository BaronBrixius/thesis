from graph_tool.all import Graph, local_clustering, shortest_distance
from graph_tool.inference import PPBlockState
import networkx as nx
import numpy as np
from typing import Optional, Dict
from functools import lru_cache

class Metrics:
    def __init__(self, network):
        edge_list = list(zip(*network.adjacency_matrix.nonzero()))
        graph = Graph(g=edge_list, directed=False)
        self.block_state = PPBlockState(graph)

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
        unique_clusters, cluster_sizes = np.unique(cluster_assignments, return_counts=True)
        
        # Calculate cluster sizes and densities in one pass
        intra_cluster_densities = {}
        total_nodes = graph.num_vertices()
        total_density_weight = 0
        intra_cluster_edges = 0

        for cluster, size in zip(unique_clusters, cluster_sizes):
            cluster_nodes = tuple(np.where(cluster_assignments == cluster)[0])

            graph.set_vertex_filter(graph.new_vertex_property("bool", vals=[int(v) in cluster_nodes for v in graph.vertices()]), inverted=False)
            num_cluster_edges = graph.num_edges()
            intra_cluster_edges += num_cluster_edges
            num_possible_cluster_edges = len(cluster_nodes) * (len(cluster_nodes) - 1) / 2
            graph.set_vertex_filter(None)

            density = num_cluster_edges / num_possible_cluster_edges if num_possible_cluster_edges > 0 else 0
            intra_cluster_densities[cluster] = density
            total_density_weight += size * density

        cluster_metrics = {
            "Cluster Count": len(unique_clusters),
            "Cluster Sizes": dict(zip(unique_clusters, cluster_sizes)),
            "Cluster Densities": intra_cluster_densities,
            "Average Cluster Density Weighted": total_density_weight / total_nodes,
            "Cluster Size Variance": np.var(cluster_sizes),
            "SBM Entropy Normalized": self.block_state.entropy() / graph.num_edges() if graph.num_edges() > 0 else 0,
            "Intra-cluster Edges": intra_cluster_edges,
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
