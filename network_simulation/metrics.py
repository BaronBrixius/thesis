from graph_tool.all import Graph, local_clustering, shortest_distance
from graph_tool.inference import PPBlockState
import networkx as nx
import numpy as np
from typing import Optional, Dict
from functools import lru_cache
from network_simulation.utils import start_timing, stop_timing

class Metrics:
    def __init__(self, num_nodes):
        self.community_assignments = np.zeros(num_nodes, dtype=int)
        self.update_step = 0

    def create_graph(self, adjacency_matrix):
        edge_list = list(zip(*adjacency_matrix.nonzero()))
        return Graph(g=edge_list, directed=False)

    def compute_metrics(self, adjacency_matrix, activities, step):
        graph = self.create_graph(adjacency_matrix)
        nx_graph = nx.from_numpy_array(adjacency_matrix)

        # Compute row data
        row = {
            "Step": step,
            "Clustering Coefficient": self.calculate_clustering_coefficient(graph),
            "Average Path Length": self.calculate_average_path_length(graph),
            "Rich Club Coefficients": self.calculate_rich_club_coefficients(nx_graph),
        }

        # Update with cluster metrics
        row.update(self.get_community_metrics(graph, step))

        return row

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
    def calculate_rich_club_coefficients(nx_graph) -> Dict[int, float]:
        return nx.rich_club_coefficient(nx_graph, normalized=False)

    @lru_cache(maxsize=16)
    def get_community_metrics(self, graph: Graph, step: int) -> Dict[str, object]:
        block_state = self.get_block_state(graph)
        unique_communities, community_sizes = np.unique(block_state.get_blocks().a, return_counts=True)

        # Calculate cluster sizes and densities in one pass
        intra_community_densities = {}
        total_nodes = graph.num_vertices()
        total_density_weight = 0
        intra_community_edges = 0

        for community, size in zip(unique_communities, community_sizes):
            community_nodes = tuple(np.where(block_state.get_blocks().a == community)[0])

            graph.set_vertex_filter(graph.new_vertex_property("bool", vals=[int(v) in community_nodes for v in graph.vertices()]), inverted=False)
            num_community_edges = graph.num_edges()
            intra_community_edges += num_community_edges
            num_possible_community_edges = len(community_nodes) * (len(community_nodes) - 1) / 2
            graph.set_vertex_filter(None)

            density = num_community_edges / num_possible_community_edges if num_possible_community_edges > 0 else 0
            intra_community_densities[community] = density
            total_density_weight += size * density

        community_metrics = {
            "Community Count": len(unique_communities),
            "Community Sizes": dict(zip(unique_communities, community_sizes)),
            "Community Densities": intra_community_densities,
            "Average Community Density Weighted": total_density_weight / total_nodes,
            "Community Size Variance": np.var(community_sizes),
            "SBM Entropy Normalized": block_state.entropy() / graph.num_edges() if graph.num_edges() > 0 else 0,
            "Intra-community Edges": intra_community_edges,
        }

        return community_metrics

    def get_block_state(self, graph: Graph):
        block_state = PPBlockState(graph, b=self.community_assignments) # block state with the latest graph, preserves the previous community assignments

        for _ in range(5):
            entropy_delta, _, _ = block_state.multilevel_mcmc_sweep()            #TODO multilevel isn't really needed for us, but the regular multiflip keeps hanging. change?
            if entropy_delta == 0:
                break

        # Get the initial cluster assignments
        # TODO See if this does anything useful
        # community_assignments = block_state.get_blocks().a
        # unique_values = np.unique(community_assignments)  # Map community assignments to normalized ids (0, 1, 2, ...) instead of arbitrary values
        # value_map = {old: new for new, old in enumerate(unique_values)}
        # normalized_assignments = np.array([value_map[x] for x in community_assignments])
        # block_state.get_blocks().a = normalized_assignments 

        return block_state
