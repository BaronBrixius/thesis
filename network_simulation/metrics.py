from graph_tool.all import Graph, local_clustering, shortest_distance
from graph_tool.inference import PPBlockState
import networkx as nx
import numpy as np
from typing import Optional, Dict

class Metrics:
    def __init__(self, adjacency_matrix):
        self.block_state = PPBlockState(Graph(g=np.transpose(np.nonzero(adjacency_matrix)), directed=False))
        self.last_update_step = -1

    def compute_metrics(self, adjacency_matrix, step):
        # TODO move this
        community_metrics = self._calculate_community_metrics(adjacency_matrix, step)

        nx_graph = nx.from_numpy_array(adjacency_matrix)

        # Compute row data
        row = {
            "Step": step,
            "Clustering Coefficient": self._calculate_clustering_coefficient(self.block_state.g),
            "Average Path Length": self._calculate_average_path_length(self.block_state.g),
            "Rich Club Coefficients": self._calculate_rich_club_coefficients(nx_graph),
        }

        # Add community metrics
        row.update(community_metrics)

        return row

    @staticmethod
    def _calculate_clustering_coefficient(graph: Graph) -> float:
        """
        Clustering Coefficient (CC): Tendency of nodes to form tightly knit groups (triangles).
        """
        return local_clustering(graph).get_array().mean()

    @staticmethod
    def _calculate_average_path_length(graph: Graph) -> Optional[float]:
        """
        Average Path Length (APL): Average shortest path length between all pairs of nodes in the network.
        """
        distances = shortest_distance(graph, directed=False)
        num_vertex_pairs = (graph.num_vertices()**2 - graph.num_vertices())
        ave_path_length = 2 * sum([sum(row[j + 1:]) for j, row in enumerate(distances)]) / num_vertex_pairs  # Only sum the upper triangle of the distance matrix, and double it
        return ave_path_length

    @staticmethod
    def _calculate_rich_club_coefficients(nx_graph) -> Dict[int, float]:
        return nx.rich_club_coefficient(nx_graph, normalized=False)

    def _calculate_community_metrics(self, adjacency_matrix, step: int) -> Dict[str, object]:
        self._update_block_model(adjacency_matrix, step)
        community_assignments = self.block_state.get_blocks().a
        unique_communities, community_sizes = np.unique(community_assignments, return_counts=True)
        intra_community_densities, intra_community_edges = self._calculate_community_densities(self.block_state.g, community_assignments, unique_communities)

        return {
            "Community Count": len(unique_communities),
            "Community Sizes": dict(zip(unique_communities, community_sizes)),
            "Community Densities": intra_community_densities,
            "Community Size Variance": np.var(community_sizes),
            "SBM Entropy Normalized": (self.block_state.entropy() / self.block_state.g.num_edges()) if self.block_state.g.num_edges() > 0 else 0,
            "Intra-Community Edges": intra_community_edges,
        }

    @staticmethod
    def _calculate_community_densities(graph, community_assignments, unique_communities):
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

    def _update_block_model(self, adjacency_matrix, step: int, max_sweeps=5):
        if step > self.last_update_step:    # Only update if the adjacency matrix has changed
            # Update the graph with the latest adjacency matrix
            graph = self.block_state.g
            graph.clear_edges()
            graph.add_edge_list(np.transpose(np.nonzero(adjacency_matrix)))  

            # Recreate the block state to reflect the new graph
            self.block_state = PPBlockState(g=graph, b=self.block_state.b) 

            # MCMC sweeps to update the community assignments
            for _ in range(max_sweeps):
                entropy_delta, _, _ = self.block_state.multilevel_mcmc_sweep()  # TODO multilevel isn't really needed for us, but the regular multiflip keeps hanging. change?
                if entropy_delta == 0:
                    break

            self.last_update_step = step
