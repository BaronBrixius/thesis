from graph_tool.all import Graph, local_clustering, shortest_distance
from graph_tool.inference import BlockState
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
        for key in self.rewirings.keys():
            self.rewirings[key] = 0

    ## Individual Metric Calculation Methods ##

    @staticmethod
    def get_clustering_coefficient(graph: Graph) -> float:
        """
        Clustering Coefficient (CC): Tendency of nodes to form tightly knit groups (triangles).
        """
        return local_clustering(graph).get_array().mean()

    @staticmethod
    def calculate_average_path_length(graph: Graph) -> Optional[float]:
        """
        Average Path Length (APL): Average shortest path length between all pairs of nodes in the network.
        """
        distances = shortest_distance(graph, directed=False).get_2d_array(range(graph.num_vertices()))
        finite_distances = distances[np.isfinite(distances)]
        return np.mean(finite_distances) if len(finite_distances) > 0 else None

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

    # @lru_cache(maxsize=128)
    def get_cluster_assignments(self, graph: Graph, step: int):
        state = BlockState(graph, b=self.current_cluster_assignments)
        state.multiflip_mcmc_sweep(niter=10, beta=np.inf)
        cluster_assignments = state.get_blocks().a
        self.current_cluster_assignments = cluster_assignments
        return cluster_assignments

    def get_cluster_metrics(self, graph: Graph, step: int) -> Dict[str, object]:
        old_cluster_assignment = self.current_cluster_assignments
        cluster_assignments = self.get_cluster_assignments(graph, step)
        unique_clusters = np.unique(cluster_assignments)

        cluster_sizes = self.get_cluster_sizes(tuple(cluster_assignments))
        intra_cluster_densities = self.get_cluster_densities(graph, tuple(cluster_assignments))

        return {
            "Cluster Membership": {i: np.where(cluster_assignments == i)[0].tolist() for i in unique_clusters},
            "Cluster Count": len(unique_clusters),
            "Cluster Membership Stability": self.calculate_cluster_membership_stability(cluster_assignments, old_cluster_assignment),
            "Cluster Sizes": cluster_sizes,
            "Average Cluster Size": np.mean(list(cluster_sizes.values())),
            "Cluster Densities": intra_cluster_densities,
            "Average Cluster Density": np.mean(list(intra_cluster_densities.values())),
            "Cluster Size Variance": self.calculate_cluster_size_variance(cluster_assignments),
        }

    @staticmethod
    # @lru_cache(maxsize=128)
    def get_cluster_sizes(cluster_assignments: tuple) -> Dict[int, int]:
        counts = np.unique(cluster_assignments, return_counts=True)[1]
        return {i: count for i, count in enumerate(counts)}

    @staticmethod
    # @lru_cache(maxsize=128)
    def get_cluster_densities(graph: Graph, cluster_assignments: tuple) -> Dict[int, float]:
        unique_clusters = np.unique(cluster_assignments)
        return {
            cluster: Metrics.get_intra_cluster_density(
                graph, tuple(np.where(cluster_assignments == cluster)[0])
            )
            for cluster in unique_clusters
        }

    @staticmethod
    # @lru_cache(maxsize=128)
    def get_intra_cluster_density(graph: Graph, cluster_nodes: tuple) -> float:
        graph.set_vertex_filter(graph.new_vertex_property("bool", vals=[int(v) in cluster_nodes for v in graph.vertices()]), inverted=False)
        num_edges = graph.num_edges()
        num_possible_edges = len(cluster_nodes) * (len(cluster_nodes) - 1) / 2
        graph.set_vertex_filter(None)
        return num_edges / num_possible_edges if num_possible_edges > 0 else 0

    @staticmethod
    def calculate_cluster_membership_stability(current_assignments: np.ndarray, previous_assignments: np.ndarray) -> float:
        """Cluster Membership Stability: Similarity between cluster assignments across time steps."""
        if previous_assignments is None:
            return 0.0

        return adjusted_rand_score(previous_assignments, current_assignments)

    @staticmethod
    def calculate_cluster_size_variance(cluster_assignments: np.ndarray) -> float:
        """Cluster Size Variance: Variability in cluster sizes."""
        _, counts = np.unique(cluster_assignments, return_counts=True)
        return np.var(counts)

    # Fourier Analysis
    @staticmethod
    def calculate_amplitude_of_oscillations(values: np.ndarray) -> float:
        """Amplitude of Oscillations: Max range of metric values."""
        return np.max(values) - np.min(values)

    @staticmethod
    def summarize_metric(values: np.ndarray) -> Dict[str, float]:
        """Summarizes a metric over the entire run."""
        return {
            "Mean": np.mean(values),
            "StdDev": np.std(values),
            "Max": np.max(values),
            "Min": np.min(values),
        }

    @staticmethod
    def calculate_fourier_transform(values: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.fft(values))
