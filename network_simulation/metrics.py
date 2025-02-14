from graph_tool.all import Graph, local_clustering, shortest_distance
from graph_tool.inference import PPBlockState
import networkx as nx
import numpy as np
from typing import Optional, Dict
from network_simulation.utils import start_timing, stop_timing

class Metrics:
    def __init__(self):
        self.last_community_assignments = None
        self.last_entropy = None
        self.last_update_step = -1

    def compute_metrics(self, adjacency_matrix, activities, step):
        edge_list = list(zip(*adjacency_matrix.nonzero()))
        graph = Graph(g=edge_list, directed=False)
        nx_graph = nx.from_numpy_array(adjacency_matrix)

        # Compute row data
        row = {
            "Step": step,
            "Clustering Coefficient": self.calculate_clustering_coefficient(graph),
            "Average Path Length": self.calculate_average_path_length(graph),
            "Rich Club Coefficients": self.calculate_rich_club_coefficients(nx_graph),
        }

        # Update with community metrics
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
        num_vertex_pairs = (graph.num_vertices()**2 - graph.num_vertices())
        ave_path_length = 2 * sum([sum(row[j + 1:]) for j, row in enumerate(distances)]) / num_vertex_pairs  # Only sum the upper triangle of the distance matrix, and double it
        return ave_path_length

    @staticmethod
    def calculate_rich_club_coefficients(nx_graph) -> Dict[int, float]:
        return nx.rich_club_coefficient(nx_graph, normalized=False)

    def get_community_metrics(self, graph: Graph, step: int) -> Dict[str, object]:
        community_assignments, entropy = self.get_community_assignments(graph, step)
        unique_communities, community_sizes = np.unique(community_assignments, return_counts=True)
        intra_community_densities, intra_community_edges = self.calculate_community_densities(graph, community_assignments, unique_communities)

        return {
            "Community Count": len(unique_communities),
            "Community Sizes": dict(zip(unique_communities, community_sizes)),
            "Community Densities": intra_community_densities,
            "Community Size Variance": np.var(community_sizes),
            "SBM Entropy Normalized": entropy / graph.num_edges() if graph.num_edges() > 0 else 0,
            "Intra-Community Edges": intra_community_edges,
        }

    def calculate_community_densities(self, graph, community_assignments, unique_communities):
        intra_community_densities = {}
        intra_community_edges = 0

        for community in unique_communities:
            # Filter to only see the current community
            community_nodes = np.where(community_assignments == community)[0]
            graph.set_vertex_filter(graph.new_vertex_property("bool", vals=[int(v) in community_nodes for v in graph.vertices()]), inverted=False)

            # Calculate density
            num_community_edges = graph.num_edges()
            num_possible_community_edges = len(community_nodes) * (len(community_nodes) - 1) / 2
            intra_community_densities[community] = num_community_edges / num_possible_community_edges

            # Update total intra-community edges
            intra_community_edges += num_community_edges

            graph.set_vertex_filter(None)

        return intra_community_densities, intra_community_edges

    def get_community_assignments(self, graph: Graph, step: int):
        if step > self.last_update_step:
            self.last_community_assignments, self.last_entropy = self._calculate_community_assignments(graph)
            self.last_update_step = step

        return self.last_community_assignments, self.last_entropy

    def _calculate_community_assignments(self, graph: Graph):
        block_state = PPBlockState(graph, b=self.last_community_assignments)  # block state with the latest graph, b preserves the current community assignments as a starting point

        for _ in range(5):
            entropy_delta, _, _ = block_state.multilevel_mcmc_sweep()  # TODO multilevel isn't really needed for us, but the regular multiflip keeps hanging. change?
            if entropy_delta == 0:
                break

        return block_state.get_blocks().a, block_state.entropy()
