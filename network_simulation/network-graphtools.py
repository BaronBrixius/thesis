from graph_tool.all import Graph
import numpy as np
from network_simulation.metrics import Metrics
from network_simulation.physics import Physics

class NodeNetwork:
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, random_seed=None):
        # Store params
        np.random.seed(random_seed)
        self.num_nodes = num_nodes
        self.num_connections = num_connections

        self.graph = Graph(directed=False)
        self.graph.add_vertex(num_nodes)  # Add nodes to the graph

        self.alpha = alpha
        self.epsilon = epsilon

        # Construct network
        self.activities = self.graph.new_vertex_property("float")
        for v in self.graph.vertices():
            self.activities[v] = np.random.uniform(-1, 1)

        self.positions = self.graph.new_vertex_property("vector<float>")
        for v in self.graph.vertices():
            self.positions[v] = np.random.uniform(0.1, 0.9, 2)
        self.add_random_edges(num_connections)

        # Initialize subclasses
        self.physics = Physics(normal_distance=(0.5 * np.sqrt(self.num_connections + self.num_nodes) / self.num_nodes))
        self.metrics = Metrics()

    def update_network_structure(self, new_node_count, new_connection_count):
        pass
        # """
        # Update the structure of the network by changing the number of nodes and connections.
        # """
        # # Handle node updates
        # if new_node_count > self.num_nodes:     # Add new nodes
        #     diff = new_node_count - self.num_nodes
        #     self.activities = np.append(self.activities, np.random.rand(diff))
        #     self.adjacency_matrix = np.pad(self.adjacency_matrix, ((0, diff), (0, diff)), mode='constant')

        #     # Add positions for new nodes along the sides of the space
        #     new_positions = np.random.rand(diff * 3, 2)  # Generate more candidates than needed
        #     distances = np.linalg.norm(new_positions - 0.5, axis=1)  # Calculate distances from center
        #     valid_positions = new_positions[distances >= 0.35]  # Filter invalid positions
        #     self.positions = np.vstack([self.positions, valid_positions[:diff]])  # Add required positions

        # elif new_node_count < self.num_nodes:       # Remove nodes
        #     diff = self.num_nodes - new_node_count
        #     self.activities = self.activities[:new_node_count]
        #     self.adjacency_matrix = self.adjacency_matrix[:new_node_count, :new_node_count]
        #     self.positions = self.positions[:new_node_count]

        # self.num_nodes = new_node_count
        # self.num_connections = np.sum(self.adjacency_matrix) // 2

        # # Recalculate maximum possible edges based on updated nodes
        # max_possible_edges = new_node_count * (new_node_count - 1) // 2
        # new_connection_count = min(new_connection_count, max_possible_edges)

        # # Adjust connections
        # if new_connection_count > self.num_connections:
        #     self.add_random_connections(new_connection_count - self.num_connections)
        # elif new_connection_count < self.num_connections:
        #     self.remove_random_connections(self.num_connections - new_connection_count)

        # self.num_connections = new_connection_count

        # self.metrics.reset_rewiring_count()
        # self.metrics.current_cluster_assignments = None

    def add_random_edges(self, num_connections_to_add):
        """Add random connections to the graph."""
        edges_to_add = np.random.choice(
            [e for e in self.graph.get_edge_ids() if not self.graph.edge(*e)],
            size=num_connections_to_add,
            replace=False,
        )
        for source, target in edges_to_add:
            self.add_connection(source, target)

    def remove_random_connections(self, num_connections_to_remove):
        """Remove random connections from the graph."""
        edges = list(self.graph.edges())
        if len(edges) < num_connections_to_remove:
            raise ValueError("Not enough connections to remove.")

        selected_edges = np.random.choice(len(edges), size=num_connections_to_remove, replace=False)
        for edge_idx in selected_edges:
            self.graph.remove_edge(edges[edge_idx])

    def add_connection(self, a, b):
        self.graph.add_edge(a, b)

    def remove_connection(self, a, b):
        self.graph.remove_edge(a, b)

    def update_activity(self):
        """Update node activities based on neighbors' activities."""
        new_activities = self.graph.new_vertex_property("float")
        for v in self.graph.vertices():
            neighbors = v.out_neighbors()
            neighbor_count = len(list(neighbors))
            if neighbor_count > 0:
                neighbor_sum = sum(self.activities[n] for n in neighbors)
                avg_neighbor_activity = neighbor_sum / neighbor_count
                new_activities[v] = (1 - self.epsilon) * self.activities[v] + self.epsilon * avg_neighbor_activity
            else:
                new_activities[v] = self.activities[v]

            # Logistic map: x(n+1) = 1 - a*x(n)^2
            new_activities[v] = 1 - self.alpha * new_activities[v]**2

        self.activities = new_activities

    def rewire(self, step):
        # 1. Pick a unit at random (henceforth: pivot)
        v_pivot = np.random.choice(list(self.graph.vertices()))
        while v_pivot.out_degree() == 0: # zero-connection nodes cannot be pivots
            v_pivot = np.random.choice(list(self.graph.vertices()))

        activity_diffs = {v: abs(self.activities[v_pivot] - self.activities[v]) for v in self.graph.vertices() if v != v_pivot}

        # Most similar node (candidate) and least similar neighbor
        candidate = min(activity_diffs, key=activity_diffs.get)
        least_similar_neighbor = max(v_pivot.out_neighbors(), key=lambda v: abs(self.activities[v_pivot] - self.activities[v]))

        # Add connection to candidate, remove connection to least similar neighbor
        if not self.graph.edge(v_pivot, candidate):
            self.graph.add_edge(v_pivot, candidate)
            self.graph.remove_edge(v_pivot, least_similar_neighbor)
        self.metrics.increment_rewiring_count(v_pivot, from_node=least_similar_neighbor, to_node=candidate, adjacency_matrix=self.get_adjacency_matrix(), step=step)

    # Update the activity of all nodes
    def update_network(self, step):
        self.update_activity()
        self.rewire(step)

    def apply_forces(self, effective_iterations=1):
        self.positions = self.physics.apply_forces(self.get_adjacency_matrix(), self.positions, effective_iterations)

    def get_adjacency_matrix(self):
        """Get the adjacency matrix of the graph."""
        return self.graph.get_adjacency()