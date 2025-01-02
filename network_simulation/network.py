from graph_tool.all import Graph, adjacency
import numpy as np
from network_simulation.metrics import Metrics

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

        self.vertices = self.graph.get_vertices()
        # Preallocate reusable arrays
        self.degrees = self.graph.new_vertex_property("int")
        self.degrees.a = self.graph.get_total_degrees(self.vertices)

        self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)
        for edge in self.graph.get_edges():
            self.adjacency_matrix[edge[0], edge[1]] = self.adjacency_matrix[edge[1], edge[0]] = True

        self.metrics = Metrics()

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

    def swap_edge(self, pivot, old_target, new_target):
        """Swap one endpoint of an edge, replacing old_target with new_target."""
        edge = self.graph.edge(pivot, old_target)
        if not edge:
            return  # Edge doesn't exist, no need to proceed

        # Update adjacency matrix
        self.adjacency_matrix[pivot, old_target] = self.adjacency_matrix[old_target, pivot] = False
        self.adjacency_matrix[pivot, new_target] = self.adjacency_matrix[new_target, pivot] = True

        # Update degrees
        self.degrees[old_target] -= 1
        self.degrees[new_target] += 1

        # Efficiently replace the edge
        self.graph.remove_edge(edge)
        self.graph.add_edge(pivot, new_target)

    def update_activity(self):
        # Sum up neighbor activities
        neighbor_sums = np.einsum("ij,j->i", self.adjacency_matrix, self.activities.a)
        # Split activity between neighbors (determined by epsilon)
        try:
            self.activities.a = (1 - self.epsilon) * self.activities.a + self.epsilon * neighbor_sums / self.degrees.a   # TODO add a catch for division by zero
        except ZeroDivisionError:   # disconnected graph, it's fairly rare, so we can afford to just recalculate
            print("ZeroDivisionError")  # TODO this doesn't seem to trigger even when we get errors?
            neighbor_counts = self.graph.get_total_degrees(self.graph.get_vertices())
            connected_nodes = neighbor_counts > 0
            self.activities.a[connected_nodes] = ((1 - self.epsilon) * self.activities.a[connected_nodes] + self.epsilon * neighbor_sums[connected_nodes] / neighbor_counts[connected_nodes])

        # Apply logistic map
        self.activities.a = 1 - self.alpha * (self.activities.a)**2

    def rewire(self, step):
        # Select a pivot node
        pivot = np.random.randint(self.num_nodes)
        pivot_neighbors = self.graph.get_out_neighbors(pivot)
        while len(pivot_neighbors) == 0:            # Select another pivot if pivot has no neighbors, very rarely happens in practice
            pivot = np.random.randint(self.num_nodes)
            pivot_neighbors = self.graph.get_out_neighbors(pivot)

        # 2. From all other units, select the one that is most synchronized (henceforth: candidate) and least synchronized neighbor
        activity_diff = np.abs(self.activities.a - self.activities.a[pivot])
        activity_diff[pivot] = np.inf                                   # stop the pivot from connecting to itself
        candidate = np.argmin(activity_diff)                            # most similar activity
        least_similar_neighbor = pivot_neighbors[np.argmax(activity_diff[pivot_neighbors])]     # least similar neighbor

        # 3a. If there is a connection between the pivot and the candidate already, do nothing
        if self.graph.edge(pivot, candidate):
            return
        self.swap_edge(pivot, least_similar_neighbor, candidate)

        # Update metrics
        self.metrics.increment_rewiring_count(pivot, least_similar_neighbor, candidate, self.graph, step)

    def update_network(self, step):
        self.update_activity()
        self.rewire(step)
