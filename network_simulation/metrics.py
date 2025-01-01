from graph_tool.all import Graph, local_clustering, shortest_distance
from graph_tool.inference import BlockState, minimize_blockmodel_dl
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from typing import Optional, Dict, List
from functools import lru_cache

class Metrics:
    def __init__(self):
        self.breakup_count = 0
        self.rewirings = {
            "intra_cluster": 0,
            "inter_cluster_change": 0,
            "inter_cluster_same": 0,
            "intra_to_inter": 0,
            "inter_to_intra": 0,
        }
        self.current_cluster_assignments = None
        self.block_state = None

    # Runtime Tracking
    def increment_breakup_count(self):
        self.breakup_count += 1

    def increment_rewiring_count(self, pivot, from_node, to_node, graph: Graph, step: int):
        if self.current_cluster_assignments is None:
            self.current_cluster_assignments = self.get_cluster_assignments(graph, step)

        partitions = self.current_cluster_assignments
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
        return sum([sum(i) for i in distances])/(graph.num_vertices()**2-graph.num_vertices())

    @staticmethod
    def calculate_rewiring_chance(graph: Graph, activities: np.ndarray) -> float:
        """
        Rewiring Chance: Ratio of nodes that are not connected to their most similar node in the network.
        """
        num_nodes = activities.shape[0]
        activity_diff = np.abs(activities[:, np.newaxis] - activities[np.newaxis, :])
        np.fill_diagonal(activity_diff, np.inf)
        most_similar_node = np.argmin(activity_diff, axis=1)
        not_connected = [not graph.edge(i, most_similar_node[i]) for i in range(num_nodes)]
        return np.mean(not_connected)

    @lru_cache(maxsize=128)
    def get_cluster_metrics(self, graph: Graph, step: int) -> Dict[str, object]:
        # Tuples to be hashable for lru_cache
        old_cluster_assignments = tuple(self.current_cluster_assignments) if self.current_cluster_assignments is not None else None
        cluster_assignments = self.get_cluster_assignments(graph, step)
        cluster_assignments_tuple = tuple(cluster_assignments)
        unique_clusters = set(cluster_assignments)

        cluster_sizes = self.get_cluster_sizes(cluster_assignments_tuple)
        intra_cluster_densities = self.get_cluster_densities(graph, cluster_assignments_tuple)

        cluster_metrics = {
            "Cluster Membership": {i: np.where(self.current_cluster_assignments == i)[0].tolist() for i in unique_clusters},
            "Cluster Count": len(unique_clusters),
            "Cluster Membership Stability": self.get_cluster_membership_stability(tuple(self.current_cluster_assignments), old_cluster_assignments),
            "Cluster Sizes": cluster_sizes,
            "Average Cluster Size": np.mean(list(cluster_sizes.values())),
            "Cluster Densities": intra_cluster_densities,
            "Average Cluster Density": np.mean(list(intra_cluster_densities.values())),
            "Cluster Size Variance": self.calculate_cluster_size_variance(self.current_cluster_assignments),
            "SBM Entropy": self.block_state.entropy(),
        }

        # Add SBM-based metrics
        # sbm_posterior = self.get_sbm_posterior_probabilities(graph)

        # cluster_metrics.update({
            # "SBM Mean Posterior": sbm_posterior["Mean Entropy"],
            # "SBM StdDev Posterior": sbm_posterior["StdDev Entropy"],
            # "SBM Best Posterior": sbm_posterior["Best Entropy"],
        # })

        return cluster_metrics

    @lru_cache(maxsize=128)
    def get_cluster_assignments(self, graph: Graph, step: int):
        self.block_state = BlockState(graph, b=self.current_cluster_assignments)
        self.block_state.multiflip_mcmc_sweep(niter=10, beta=np.inf)
        cluster_assignments = self.block_state.get_blocks().a
        self.current_cluster_assignments = cluster_assignments
        return cluster_assignments

    @staticmethod
    @lru_cache(maxsize=128)
    def get_cluster_sizes(cluster_assignments: tuple) -> Dict[int, int]:
        counts = np.unique(cluster_assignments, return_counts=True)[1]
        return {i: count for i, count in enumerate(counts)}

    @staticmethod
    @lru_cache(maxsize=128)
    def get_cluster_densities(graph: Graph, cluster_assignments: tuple) -> Dict[int, float]:
        unique_clusters = np.unique(cluster_assignments)
        return {
            cluster: Metrics.get_intra_cluster_density(
                graph, tuple(np.where(cluster_assignments == cluster)[0])
            )
            for cluster in unique_clusters
        }

    @staticmethod
    @lru_cache(maxsize=128)
    def get_intra_cluster_density(graph: Graph, cluster_nodes: tuple) -> float:
        graph.set_vertex_filter(graph.new_vertex_property("bool", vals=[int(v) in cluster_nodes for v in graph.vertices()]), inverted=False)
        num_edges = graph.num_edges()
        num_possible_edges = len(cluster_nodes) * (len(cluster_nodes) - 1) / 2
        graph.set_vertex_filter(None)
        return num_edges / num_possible_edges if num_possible_edges > 0 else 0

    @staticmethod
    @lru_cache(maxsize=128)
    def get_cluster_membership_stability(current_assignments: tuple, previous_assignments: tuple) -> float:
        """Cluster Membership Stability: Similarity between cluster assignments across time steps."""
        if previous_assignments is None:
            return 0.0

        return adjusted_rand_score(previous_assignments, current_assignments)

    @staticmethod
    def calculate_cluster_size_variance(cluster_assignments: np.ndarray) -> float:
        """Cluster Size Variance: Variability in cluster sizes."""
        _, counts = np.unique(cluster_assignments, return_counts=True)
        return np.var(counts)

    @staticmethod
    def get_sbm_posterior_probabilities(graph, num_runs=10):
        """Run SBM multiple times and calculate posterior probabilities."""
        entropies = []
        for _ in range(num_runs):
            state = minimize_blockmodel_dl(graph)
            entropies.append(state.entropy())
        return {
            "Mean Entropy": np.mean(entropies),
            "StdDev Entropy": np.std(entropies),
            "Best Entropy": min(entropies),
        }