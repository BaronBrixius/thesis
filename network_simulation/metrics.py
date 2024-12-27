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
    
    def calculate_clustering_coefficient(self, graph):
        """
        Clustering Coefficient (CC): Tendency of nodes to form tightly knit groups (triangles).
        """
        return nx.average_clustering(graph)

    def calculate_average_path_length(self, graph):
        """
        Average Path Length (APL): Average shortest path length between all pairs of nodes in the network.
        """
        try:
            return nx.average_shortest_path_length(graph)
        except nx.NetworkXError:
            return None  # Disconnected graph

    def calculate_rewiring_chance(self, adjacency_matrix, activities):
        """
        Rewiring Chance: Ratio of nodes that are not connected to their most similar node in the network.
        """
        num_nodes = activities.shape[0]
        activity_diff = np.abs(activities[:, np.newaxis] - activities[np.newaxis, :])               # Compute pairwise differences, ignoring self
        np.fill_diagonal(activity_diff, np.inf)
        most_similar_node = np.argmin(activity_diff, axis=1)                                        # Find the most similar node for each node
        not_connected = [adjacency_matrix[i, most_similar_node[i]] == 0 for i in range(num_nodes)]  # Check if each node is connected to its most similar node
        return np.mean(not_connected)                       # Calculate the rewiring chance as the fraction of nodes not connected to their most similar node

    def calculate_rich_club_coefficient(self, graph):
        """
        Rich-Club Coefficient: Tendency of high-degree nodes to form tightly interconnected subgraphs.
        """
        return nx.rich_club_coefficient(graph, normalized=False)    # normalization=True gives divide by zero errors in the code for generating the random graph (weird)

    def calculate_edge_persistence(self, current_adjacency, previous_adjacency, num_connections=None):
        """
        Edge Persistence: Ratio of existing edges that also existed in the previous snapshot.
        """
        if previous_adjacency is None:
            return None
        if num_connections is None:
            num_connections = np.sum(previous_adjacency > 0)
        overlap = np.sum((current_adjacency > 0) & (previous_adjacency > 0))
        return overlap / num_connections if num_connections > 0 else 0

    def get_cluster_assignments(self, adjacency_matrix, step=None):
        if self.assignment_step == step:
            return self.current_cluster_assignments

        # Reformat current assignments so they can be used for the calculation, if available
        initial_membership = self._convert_communities_to_partition(self.current_cluster_assignments, len(adjacency_matrix)) if self.current_cluster_assignments is not None else None

        new_cluster_assignments = leiden(nx.from_numpy_array(adjacency_matrix), initial_membership=initial_membership)

        self.current_cluster_assignments = new_cluster_assignments.communities
        self.assignment_step = step
        return self.current_cluster_assignments

    def _convert_communities_to_partition(self, communities, num_nodes):
        """
        Convert a list of communities to a partition format.
        """
        partition = np.full(num_nodes, -1, dtype=int)  # Default to unassigned
        for cluster_id, cluster in enumerate(communities):
            for node in cluster:
                if node < num_nodes:
                    partition[node] = cluster_id
        return partition

    def calculate_cluster_membership_stability(self, current_assignments, previous_assignments):
        """
        Cluster Membership Stability: Similarity between cluster assignments across time steps.
        """
        if previous_assignments is None:
            return None

        # Flatten cluster assignments for comparison
        def flatten_assignments(assignments):
            flat = {}
            for cluster_id, cluster in enumerate(assignments):
                for node in cluster:
                    flat[node] = cluster_id
            return [flat[node] for node in sorted(flat.keys())]

        current_flat = flatten_assignments(current_assignments)
        previous_flat = flatten_assignments(previous_assignments)

        return adjusted_rand_score(previous_flat, current_flat)

    def calculate_cluster_size_variance(self, cluster_assignments):
        """Cluster Size Variance: Variability in cluster sizes."""
        _, counts = np.unique(cluster_assignments, return_counts=True)
        return np.var(counts)

    def calculate_intra_cluster_density(self, adjacency_matrix, cluster):
        cluster_nodes = list(cluster)
        subgraph = adjacency_matrix[np.ix_(cluster_nodes, cluster_nodes)]
        num_possible_edges = len(cluster_nodes) * (len(cluster_nodes) - 1) / 2
        return subgraph.sum() / (2 * num_possible_edges)

    # Fourier Analysis
    def calculate_amplitude_of_oscillations(self, values):
        """Amplitude of Oscillations: Max range of metric values."""
        return np.max(values) - np.min(values)

    def calculate_clustering_coefficient_spectral(self, cc_values):
        """Fourier Analysis of CC: Dominant frequency and spectral power."""
        frequencies, power = periodogram(cc_values)
        dominant_frequency = frequencies[np.argmax(power)]
        spectral_power = np.sum(power)
        return {"Dominant Frequency": dominant_frequency, "Spectral Power": spectral_power}

    def calculate_shortest_path_spectral(self, adjacency_matrix):
        """Shortest Path Spectral Metrics: Fourier analysis of path length fluctuations."""
        graph = nx.from_numpy_array(adjacency_matrix)
        try:
            apl_values = nx.average_shortest_path_length(graph)
        except nx.NetworkXError:
            apl_values = None
        if apl_values is not None:
            apl_values = np.array([apl_values])
            frequencies, power = periodogram(apl_values)
            return {"Dominant Frequency": frequencies[np.argmax(power)], "Spectral Power": np.sum(power)}
        return {"Dominant Frequency": None, "Spectral Power": None}

    # Helper Methods
    def summarize_metric(self, values):
        """Summarizes a metric over the entire run."""
        return {
            "Mean": np.mean(values),
            "StdDev": np.std(values),
            "Max": np.max(values),
            "Min": np.min(values),
        }