from graph_tool.all import Graph, local_clustering, shortest_distance
from graph_tool.inference import minimize_nested_blockmodel_dl
import numpy as np
from scipy.signal import periodogram
from sklearn.metrics.cluster import adjusted_rand_score
from graph_tool.inference import BlockState, NestedBlockState

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

    def increment_rewiring_count(self, pivot, from_node, to_node, graph, step):
        """Categorize and count rewiring events."""
        if self.current_cluster_assignments is None:
            self.current_cluster_assignments = self.get_cluster_assignments(graph, step)

        partitions = self.current_cluster_assignments
        # print(partitions)
        pivot_index = int(pivot)
        from_index = int(from_node)
        to_index = int(to_node)

        pivot_cluster = partitions[pivot_index]
        from_cluster = partitions[from_index]
        to_cluster = partitions[to_index]

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

    def get_cluster_assignments(self, graph, step=None):
        """
        Get cluster assignments using the Stochastic Block Model (SBM) without hierarchy.
        - Uses cached results if available.
        """

        print(f"Step {step}: Calculating cluster assignments using Stochastic Block Model...")

        # Initialize SBM with prior assignments if available
        if self.current_cluster_assignments is not None:
            print(f"Step {step}: Using prior assignments as initialization.")
            state = BlockState(
                graph,
                b=self.current_cluster_assignments,
                state_args=dict(deg_corr=True)
            )
        else:
            # Fresh initialization without prior
            state = BlockState(
                graph,
                state_args=dict(deg_corr=True)
            )

        # Optimize the state
        state.multiflip_mcmc_sweep(niter=10, beta=np.inf)  # Perform refinement to improve clustering

        # Get clustering
        cluster_assignments = state.get_blocks().a  # Extract assignments as a NumPy array

        # Validate and log cluster details
        unique_clusters = np.unique(cluster_assignments)
        cluster_sizes = {cluster: np.sum(cluster_assignments == cluster) for cluster in unique_clusters}
        print(f"Step {step}: Found {len(unique_clusters)} clusters with sizes {cluster_sizes}")

        # Cache the results for future use
        self.current_cluster_assignments = cluster_assignments
        self.assignment_step = step

        return cluster_assignments

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
