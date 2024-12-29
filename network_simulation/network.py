from graph_tool.all import Graph, adjacency
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
        self.graph = Graph(directed=False)
        self.graph.add_vertex(num_nodes)  # Add nodes to the graph
        self.add_random_connections(num_connections)

        # Initialize node activities
        self.activities = self.graph.new_vertex_property("float")
        self.activities.a = np.random.uniform(-1, 1, num_nodes)

        self.vertices = self.graph.get_vertices()
        # Preallocate reusable arrays
        self.neighbor_sums = self.graph.new_vertex_property("float")
        self.degrees = self.graph.new_vertex_property("int")
        self.degrees.a = self.graph.get_total_degrees(self.vertices)

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

    def add_edge(self, source, target):
        """Add an edge and update the degrees."""
        if not self.graph.edge(source, target):
            self.graph.add_edge(source, target)
            self.degrees[source] += 1
            self.degrees[target] += 1

    def remove_edge(self, source, target):
        """Remove an edge and update the degrees."""
        edge = self.graph.edge(source, target)
        if edge:
            self.graph.remove_edge(edge)
            self.degrees[source] -= 1
            self.degrees[target] -= 1

    def update_activity(self):
        """Perform one step of activity update and rewiring in a single pass."""
        start_timing("activity1")
        edges = self.graph.get_edges()
        stop_timing("activity1")

        start_timing("activity2")
        # Vectorized activity update
        self.neighbor_sums.a.fill(0)
        np.add.at(self.neighbor_sums.a, edges[:, 0], self.activities.a[edges[:, 1]])
        np.add.at(self.neighbor_sums.a, edges[:, 1], self.activities.a[edges[:, 0]])
        stop_timing("activity2")

        start_timing("activity3")
        self.activities.a = (
            (1 - self.epsilon) * self.activities.a
            + self.epsilon * self.neighbor_sums.a / self.degrees.a
        )

        # connected_nodes = self.degrees > 0
        # stop_timing("activity2")

        # start_timing("activity3")
        # self.activities.a[connected_nodes] = (
        #     (1 - self.epsilon) * self.activities.a[connected_nodes]
        #     + self.epsilon * self.neighbor_sums[connected_nodes] / self.degrees[connected_nodes]
        # )


        # Logistic map
        self.activities.a = 1 - self.alpha * self.activities.a**2
        stop_timing("activity3")

    def rewire(self, step):
        start_timing("rewire1")
        # Select a pivot node
        pivot_idx = np.random.randint(0, self.num_nodes)
        pivot_neighbors = self.graph.get_out_neighbors(pivot_idx)
        while len(pivot_neighbors) == 0:            # Select another pivot if pivot has no neighbors, very rarely happens in practice
            pivot_idx = np.random.randint(0, self.num_nodes)
            pivot_neighbors = self.graph.get_out_neighbors(pivot_idx)
        stop_timing("rewire1")

        start_timing("rewire2")
        # Calculate activity differences
        is_neighbor = np.zeros(self.num_nodes, dtype=bool)
        is_neighbor[pivot_neighbors] = True
        is_neighbor[pivot_idx] = True
        non_neighbors = np.where(~is_neighbor)[0]

        # Calculate activity differences
        activity_diffs = np.abs(self.activities.a - self.activities.a[pivot_idx])

        # Select most similar non-neighbor and least similar neighbor
        candidate = non_neighbors[np.argmin(activity_diffs[non_neighbors])]
        least_similar_neighbor = pivot_neighbors[np.argmax(activity_diffs[pivot_neighbors])]
        stop_timing("rewire2")

        start_timing("rewire3")
        # Add connection to candidate and remove from least similar neighbor
        if not self.graph.edge(pivot_idx, candidate):
            self.graph.add_edge(pivot_idx, candidate)
            edge_to_remove = self.graph.edge(pivot_idx, least_similar_neighbor)
            if edge_to_remove:
                self.graph.remove_edge(edge_to_remove)

        # Update metrics
        self.metrics.increment_rewiring_count(pivot_idx, least_similar_neighbor, candidate, self.graph, step)
        stop_timing("rewire3")

    def update_network(self, step):
        start_timing("update_network")
        self.update_activity()
        self.rewire(step)
        stop_timing("update_network")
