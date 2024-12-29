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
        self.adjacency_matrix = adjacency(self.graph).todense()

    def update_activity(self):
        """Update node activities based on neighbors' activities."""
        start_timing("activity1")
        edges = self.graph.get_edges()
        stop_timing("activity1")

        start_timing("activity2")
        # Vectorized activity update
        neighbor_sum = np.zeros(self.num_nodes)
        np.add.at(neighbor_sum, edges[:, 0], self.activities.a[edges[:, 1]])
        np.add.at(neighbor_sum, edges[:, 1], self.activities.a[edges[:, 0]])
        neighbor_counts = self.graph.get_total_degrees(self.graph.get_vertices())
        connected_nodes = neighbor_counts > 0
        stop_timing("activity2")

        start_timing("activity3")
        self.activities.a[connected_nodes] = (
            (1 - self.epsilon) * self.activities.a[connected_nodes]
            + self.epsilon * neighbor_sum[connected_nodes] / neighbor_counts[connected_nodes]
        )

        # Logistic map
        self.activities.a = 1 - self.alpha * self.activities.a**2
        stop_timing("activity3")

    def rewire(self, step):
        """Rewire the graph based on activity similarity."""
        start_timing("rewire1")
        # Select a pivot node
        pivot = self.graph.vertex(np.random.randint(0, self.num_nodes))
        neighbors = list(pivot.out_neighbors())
        while not neighbors:
            pivot = self.graph.vertex(np.random.randint(0, self.num_nodes))
            neighbors = list(pivot.out_neighbors())
        stop_timing("rewire1")

        start_timing("rewire2")
        # Calculate activity differences
        activity_diffs = np.abs(self.activities.a - self.activities[pivot])
        all_vertices = np.arange(self.graph.num_vertices())
        non_neighbors = np.setdiff1d(all_vertices, [int(v) for v in neighbors] + [int(pivot)])

        # Select most similar non-neighbor and least similar neighbor
        candidate = non_neighbors[np.argmin(activity_diffs[non_neighbors])]
        least_similar_neighbor = neighbors[np.argmax(activity_diffs[[int(n) for n in neighbors]])]
        stop_timing("rewire2")

        start_timing("rewire3")
        # Add connection to candidate and remove from least similar neighbor
        if not self.graph.edge(pivot, candidate):
            self.graph.add_edge(pivot, candidate)
            edge_to_remove = self.graph.edge(pivot, least_similar_neighbor)
            if edge_to_remove:
                self.graph.remove_edge(edge_to_remove)

        # Update metrics
        self.metrics.increment_rewiring_count(
            pivot, least_similar_neighbor, candidate, self.graph, step
        )
        stop_timing("rewire3")

    def update_network(self, step, iterations=1):
        """Update network activities and rewire connections over multiple iterations."""
        start_timing("update_network")
        for _ in range(iterations):
            self.update_activity()
            self.rewire(step)
        stop_timing("update_network")
