from graph_tool.all import Graph
import numpy as np
from network_simulation.metrics import Metrics
from network_simulation.utils import start_timing, stop_timing

class NodeNetwork:
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, random_seed=None):
        # Seed for reproducibility
        np.random.seed(random_seed)

        self.num_nodes = num_nodes
        self.num_connections = num_connections
        self.alpha = alpha
        self.epsilon = epsilon

        # Initialize graph
        self.graph = Graph(directed=False, fast_edge_removal=True)
        self.graph.add_vertex(num_nodes)  # Add nodes to the graph
        self.add_random_connections(num_connections)

        # Initialize node activities
        self.activities = self.graph.new_vertex_property("float")
        self.activities.a = np.random.uniform(-0.7, 1.0, num_nodes)

        self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
        edges = self.graph.get_edges()
        lognormal = np.random.lognormal(mean=0.0, sigma=0.5, size=len(edges))
        # normaliz the lognormal distribution to (0,1]
        lognormal = (lognormal - np.min(lognormal)) / (np.max(lognormal) - np.min(lognormal))
        self.adjacency_matrix[edges[:, 0], edges[:, 1]] = lognormal
        self.adjacency_matrix[edges[:, 1], edges[:, 0]] = lognormal

        # Preallocate reused arrays
        self.vertices = self.graph.get_vertices()
        self.weighted_degrees = self.graph.new_vertex_property("float")
        self.weighted_degrees.a = np.sum(self.adjacency_matrix, axis=1) # self.degrees.a = self.graph.get_total_degrees(self.vertices)
        self.shuffled_indices = np.arange(num_nodes)

        self.metrics = Metrics(self.graph)

    def add_random_connections(self, num_connections_to_add):
        """Add random connections to the graph."""
        num_nodes = self.graph.num_vertices()
        edges = set()
        while len(edges) < num_connections_to_add:
            v1 = np.random.randint(0, num_nodes)
            v2 = np.random.randint(0, num_nodes)
            if v1 != v2 and (v1, v2) not in edges and (v2, v1) not in edges:
                edges.add((v1, v2))

        self.graph.add_edge_list(edges)

    def update_activity(self):
        # Sum up neighbor activities
        neighbor_sums = np.einsum("ij,j->i", self.adjacency_matrix, self.activities.a)
        # Split activity between neighbors (determined by epsilon)
        connected_nodes = self.weighted_degrees.a > 0
        self.activities.a[connected_nodes] = (
            (1 - self.epsilon)  * self.activities.a[connected_nodes] + 
            self.epsilon        * neighbor_sums[connected_nodes] / self.weighted_degrees.a[connected_nodes]
        )
        # Apply logistic map
        self.activities.a = 1 - self.alpha * (self.activities.a)**2

    def rewire(self, step):
        # Select a pivot node
        pivot = np.random.randint(self.num_nodes)
        pivot_neighbors = self.graph.get_out_neighbors(pivot)
        while len(pivot_neighbors) == 0:            # Select another pivot if no neighbors, very rarely happens in practice
            nodes_with_neighbors = np.where(self.weighted_degrees.a > 0)[0]
            if len(nodes_with_neighbors) == 0:
                return  # No rewiring possible if no nodes have neighbors
            pivot = np.random.choice(nodes_with_neighbors)
            pivot_neighbors = self.graph.get_out_neighbors(pivot)

        # 2. From all other units, select the one that is most synchronized (henceforth: candidate) and least synchronized neighbor
        activity_diff = np.abs(self.activities.a - self.activities.a[pivot])
        activity_diff[pivot] = np.inf                                   # stop the pivot from connecting to itself
        np.random.shuffle(self.shuffled_indices)
        shuffled_candidate = np.argmin(activity_diff[self.shuffled_indices])        # Find the index of the minimum in the shuffled array
        candidate = self.shuffled_indices[shuffled_candidate]                       # Randomly chosen among most similar nodes
        least_similar_neighbor = pivot_neighbors[np.argmax(activity_diff[pivot_neighbors])]     # least similar neighbor

        # 3a. If there is a connection between the pivot and the candidate already, do nothing
        if self.adjacency_matrix[pivot, candidate] > 0:
            return
        self.swap_edge(pivot, least_similar_neighbor, candidate)

        # Update metrics
        self.metrics.increment_rewiring_count(pivot, least_similar_neighbor, candidate, step)

    def swap_edge(self, pivot, old_target, new_target):
        """Swap one endpoint of an edge, replacing old_target with new_target."""
        edge = self.graph.edge(pivot, old_target)
        if not edge:
            return  # Edge doesn't exist, no need to proceed

        # Update adjacency matrix
        weight = self.adjacency_matrix[pivot, old_target]
        self.adjacency_matrix[pivot, new_target] = self.adjacency_matrix[new_target, pivot] = weight
        self.adjacency_matrix[pivot, old_target] = self.adjacency_matrix[old_target, pivot] = 0.0

        # Update degrees
        self.weighted_degrees[old_target] -= weight
        self.weighted_degrees[new_target] += weight

        # Replace the edge
        self.graph.remove_edge(edge)
        self.graph.add_edge(pivot, new_target)

    def update_network(self, step):
        for _ in range(20):
            self.update_activity()
        try:
            self.rewire(step)
        except Exception as e:
            print(e)
            pass
