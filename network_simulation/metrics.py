from graph_tool.all import Graph, local_clustering, shortest_distance
from graph_tool import topology
import networkx as nx
import numpy as np
from scipy.signal import periodogram
from sklearn.metrics.cluster import adjusted_rand_score
from network_simulation.utils import start_timing, stop_timing
from cdlib.algorithms import leiden

class Metrics:
    def __init__(self):
        self.breakup_count = 0
        self.rewirings = {
            "intra_cluster": 0,
            "inter_cluster_change": 0,
            "inter_cluster_same": 0,
            "intra_to_inter": 0,
            "inter_to_intra": 0
        }
        self.current_cluster_assignments = None
        self.assignment_step = None                     # Step at which the cluster assignments were last calculated

    # Runtime Tracking
    def increment_breakup_count(self):
        self.breakup_count += 1

    def increment_rewiring_count(self, pivot, from_node, to_node, adjacency_matrix, step):
        """Categorize and count rewiring events."""
        if self.current_cluster_assignments is None:    # TODO I don't love using cached cluster assignments, but recalculating every step is insanely slow. Find a way, if this is valuable (leiden is iterative, could just apply 1 iteration each step?) 
            self.current_cluster_assignments = self.get_cluster_assignments(adjacency_matrix)
        partitions = self._convert_communities_to_partition(self.current_cluster_assignments, len(adjacency_matrix))

        pivot_cluster = partitions[pivot]
        from_cluster = partitions[from_node]
        to_cluster = partitions[to_node]

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

    def reset_rewiring_count(self):
        """Reset all rewiring counts."""
        for key in self.rewirings.keys():
            self.rewirings[key] = 0

    ## Individual Metric Calculation Methods ##

    @staticmethod
    def calculate_clustering_coefficient(graph):
        """
        Clustering Coefficient (CC): Tendency of nodes to form tightly knit groups (triangles).
        """
        return local_clustering(graph).get_array().mean()

    @staticmethod
    def calculate_average_path_length(graph):
        """
        Average Path Length (APL): Average shortest path length between all pairs of nodes in the network.
        """
        distances = shortest_distance(graph, directed=False).get_2d_array(range(graph.num_vertices()))
        finite_distances = distances[np.isfinite(distances)]
        return np.mean(finite_distances) if len(finite_distances) > 0 else None

    @staticmethod
    def calculate_rewiring_chance(graph, activities):
        """
        Rewiring Chance: Ratio of nodes that are not connected to their most similar node in the network.
        """
        num_nodes = activities.shape[0]
        activity_diff = np.abs(activities[:, np.newaxis] - activities[np.newaxis, :])
        np.fill_diagonal(activity_diff, np.inf)
        most_similar_node = np.argmin(activity_diff, axis=1)
        not_connected = [not graph.edge(i, most_similar_node[i]) for i in range(num_nodes)]
        return np.mean(not_connected)

    def get_cluster_assignments(self, graph):
        """Get cluster assignments using connected components."""
        components, _ = topology.label_components(graph)
        return components.a

    @staticmethod
    def calculate_cluster_membership_stability(current_assignments, previous_assignments):
        """
        Cluster Membership Stability: Similarity between cluster assignments across time steps.
        """
        if previous_assignments is None:
            return None

        return adjusted_rand_score(previous_assignments, current_assignments)

    @staticmethod
    def calculate_cluster_size_variance(cluster_assignments):
        """Cluster Size Variance: Variability in cluster sizes."""
        _, counts = np.unique(cluster_assignments, return_counts=True)
        return np.var(counts)

    @staticmethod
    def calculate_intra_cluster_density(graph, cluster):
        """Density of intra-cluster connections."""
        cluster_nodes = [int(v) for v in cluster]
        subgraph = graph.copy()
        subgraph.set_vertex_filter(lambda v: int(v) in cluster_nodes)
        num_edges = subgraph.num_edges()
        num_possible_edges = len(cluster_nodes) * (len(cluster_nodes) - 1) / 2
        return num_edges / num_possible_edges if num_possible_edges > 0 else 0

    # Fourier Analysis
    @staticmethod
    def calculate_amplitude_of_oscillations(values):
        """Amplitude of Oscillations: Max range of metric values."""
        return np.max(values) - np.min(values)

    @staticmethod
    def summarize_metric(values):
        """Summarizes a metric over the entire run."""
        return {
            "Mean": np.mean(values),
            "StdDev": np.std(values),
            "Max": np.max(values),
            "Min": np.min(values),
        }
