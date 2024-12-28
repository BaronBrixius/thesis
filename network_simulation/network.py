from graph_tool.all import Graph, adjacency
import numpy as np
from network_simulation.metrics import Metrics

class NodeNetwork:
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, random_seed=None):
        # Seed for reproducibility
        np.random.seed(random_seed)
        self.num_nodes = num_nodes
        self.num_connections = num_connections

        # Initialize graph
        self.graph = Graph(directed=False)
        self.graph.add_vertex(num_nodes)  # Add nodes to the graph
        self.alpha = alpha
        self.epsilon = epsilon

        # Initialize node activities and positions
        self.activities = self.graph.new_vertex_property("float")
        for v in self.graph.vertices():
            self.activities[v] = np.random.uniform(-1, 1)

        self.positions = self.graph.new_vertex_property("vector<float>")
        for v in self.graph.vertices():
            self.positions[v] = np.random.uniform(0.1, 0.9, 2)

        # Add random connections
        self.add_random_connections(num_connections)
        
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

    def remove_random_connections(self, num_connections_to_remove):
        """Remove random connections from the graph."""
        edges = list(self.graph.edges())
        if len(edges) < num_connections_to_remove:
            raise ValueError("Not enough connections to remove.")

        selected_edges = np.random.choice(len(edges), size=num_connections_to_remove, replace=False)
        for edge_idx in selected_edges:
            self.graph.remove_edge(edges[edge_idx])

    def update_activity(self):
        """Update node activities based on neighbors' activities."""
        if self.adjacency_matrix is None:
            self.adjacency_matrix = adjacency(self.graph).todense()

        # Vectorized activity update
        neighbor_sums = np.array(self.adjacency_matrix @ self.activities.a).flatten()
        neighbor_counts = np.array(self.adjacency_matrix.sum(axis=1)).flatten()

        # Avoid division by zero
        nonzero_neighbors = neighbor_counts > 0
        self.activities.a[nonzero_neighbors] = (
            (1 - self.epsilon) * self.activities.a[nonzero_neighbors] +
            self.epsilon * (neighbor_sums[nonzero_neighbors] / neighbor_counts[nonzero_neighbors])
        )

        # Logistic map
        self.activities.a = 1 - self.alpha * self.activities.a**2

    def rewire(self, step):
        """Rewire the graph based on activity similarity."""
        pivot = self.graph.vertex(np.random.randint(0, self.num_nodes))
        neighbors = list(pivot.out_neighbors())
        while not neighbors:    # Ensure pivot node has neighbors, in practice this almost never triggers
            pivot = self.graph.vertex(np.random.randint(0, self.num_nodes))
            neighbors = list(pivot.out_neighbors())

        activity_diffs = np.abs(self.activities.a - self.activities[pivot])

        # Most similar node (candidate) and least similar neighbor
        all_vertices = np.arange(self.graph.num_vertices())
        non_neighbors = np.setdiff1d(all_vertices, [int(v) for v in neighbors] + [int(pivot)])
        candidate = non_neighbors[np.argmin(activity_diffs[non_neighbors])]

        neighbor_indices = [int(n) for n in neighbors]
        least_similar_neighbor = neighbor_indices[np.argmax(activity_diffs[neighbor_indices])]

        # Add connection to candidate, remove connection to least similar neighbor
        if not self.graph.edge(pivot, candidate):
            self.graph.add_edge(pivot, candidate)
            edge = self.graph.edge(pivot, least_similar_neighbor)
            if edge:
                self.graph.remove_edge(edge)

    def update_network(self, step):
        """Update network activities and rewire connections."""
        self.update_activity()
        self.rewire(step)

    def apply_forces(self, effective_iterations=1):
        """Placeholder for force-directed layout (if needed)."""
        pass  # Graph-tool provides visualization, but forces are external.

    def get_adjacency_matrix(self):
        """Get the adjacency matrix of the graph."""
        return self.graph.get_adjacency().data
