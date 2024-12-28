import numpy as np
from network_simulation.metrics import Metrics
from network_simulation.physics import Physics

class NodeNetwork:
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, random_seed=None):
        # Store params
        np.random.seed(random_seed)
        self.num_nodes = num_nodes
        self.num_connections = num_connections
        self.alpha = alpha
        self.epsilon = epsilon

        # Construct network
        self.activities = np.random.uniform(-1, 1, num_nodes)   # Random initial activity
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)
        self.add_random_connections(num_connections)
        self.positions = np.random.uniform([0.1, 0.1], [0.9, 0.9], (num_nodes, 2))

        # Initialize subclasses
        self.physics = Physics(normal_distance=(0.5 * np.sqrt(self.num_connections + self.num_nodes) / self.num_nodes))
        self.metrics = Metrics()

    def update_network_structure(self, new_node_count, new_connection_count):
        """
        Update the structure of the network by changing the number of nodes and connections.
        """
        # Handle node updates
        if new_node_count > self.num_nodes:     # Add new nodes
            diff = new_node_count - self.num_nodes
            self.activities = np.append(self.activities, np.random.rand(diff))
            self.adjacency_matrix = np.pad(self.adjacency_matrix, ((0, diff), (0, diff)), mode='constant')

            # Add positions for new nodes along the sides of the space
            new_positions = np.random.rand(diff * 3, 2)  # Generate more candidates than needed
            distances = np.linalg.norm(new_positions - 0.5, axis=1)  # Calculate distances from center
            valid_positions = new_positions[distances >= 0.35]  # Filter invalid positions
            self.positions = np.vstack([self.positions, valid_positions[:diff]])  # Add required positions

        elif new_node_count < self.num_nodes:       # Remove nodes
            diff = self.num_nodes - new_node_count
            self.activities = self.activities[:new_node_count]
            self.adjacency_matrix = self.adjacency_matrix[:new_node_count, :new_node_count]
            self.positions = self.positions[:new_node_count]

        self.num_nodes = new_node_count
        self.num_connections = np.sum(self.adjacency_matrix) // 2

        # Recalculate maximum possible edges based on updated nodes
        max_possible_edges = new_node_count * (new_node_count - 1) // 2
        new_connection_count = min(new_connection_count, max_possible_edges)

        # Adjust connections
        if new_connection_count > self.num_connections:
            self.add_random_connections(new_connection_count - self.num_connections)
        elif new_connection_count < self.num_connections:
            self.remove_random_connections(self.num_connections - new_connection_count)

        self.num_connections = new_connection_count

        self.metrics.reset_rewiring_count()
        self.metrics.current_cluster_assignments = None

    def add_random_connections(self, num_connections_to_add):
        """Add random connections to the network."""
        possible_edges = np.array(np.triu_indices(self.num_nodes, k=1)).T
        available_edges = possible_edges[~self.adjacency_matrix[possible_edges[:, 0], possible_edges[:, 1]]]
        selected_edges = available_edges[np.random.choice(len(available_edges), size=num_connections_to_add, replace=False)]

        self.adjacency_matrix[selected_edges[:, 0], selected_edges[:, 1]] = True
        self.adjacency_matrix[selected_edges[:, 1], selected_edges[:, 0]] = True

    def remove_random_connections(self, num_connections_to_remove):
        """Remove random connections from the network."""
        existing_edges = np.array(np.triu_indices(self.num_nodes, k=1)).T
        connected_edges = existing_edges[self.adjacency_matrix[existing_edges[:, 0], existing_edges[:, 1]]]

        if len(connected_edges) < num_connections_to_remove:
            raise ValueError("Not enough connections to remove.")

        selected_edges = connected_edges[np.random.choice(len(connected_edges), size=num_connections_to_remove, replace=False)]

        self.adjacency_matrix[selected_edges[:, 0], selected_edges[:, 1]] = False
        self.adjacency_matrix[selected_edges[:, 1], selected_edges[:, 0]] = False

    def add_connection(self, a, b):
        self.adjacency_matrix[a, b] = self.adjacency_matrix[b, a] = True

    def remove_connection(self, a, b):
        self.adjacency_matrix[a, b] = self.adjacency_matrix[b, a] = False

    def update_activity(self):
        # Calculate neighbor activities as a matrix multiplication of adjacency and activities, then row-wise summing
        neighbor_sum = np.einsum('ij,j->i', self.adjacency_matrix, self.activities) # maybe faster than self.adjacency_matrix @ self.activities
        neighbor_counts = self.adjacency_matrix.sum(axis=1)
        connected_nodes = neighbor_counts > 0  # Boolean array indicating connected nodes

        # xᵢ(n+1) = (1 − ε) * f(xᵢ(n)) + (ε / Mᵢ) * ∑(f(xⱼ(n) for j in B(i))
        self.activities[connected_nodes] = (
            (1 - self.epsilon) * self.activities[connected_nodes]
            + self.epsilon * neighbor_sum[connected_nodes] / neighbor_counts[connected_nodes]
        )
        # logistic map: x(n+1) = f(x(n)) = 1 - ax(n)²
        self.activities = 1 - self.alpha * self.activities**2

    def rewire(self, step):
        # 1. Pick a unit at random (henceforth: pivot)
        pivot = np.random.randint(self.num_nodes)
        while not np.any(self.adjacency_matrix[pivot]): # zero-connection nodes cannot be pivots
            pivot = np.random.randint(self.num_nodes)

        # 2. From all other units, select the one that is most synchronized (henceforth: candidate) and least synchronized neighbor
        # TODO optimize with a loop to look for both at once?
        activity_diff = np.abs(self.activities - self.activities[pivot])
        activity_diff_neighbors = activity_diff * self.adjacency_matrix[pivot]

        activity_diff[pivot] = np.inf                       # stop the pivot from connecting to itself
        candidate = np.argmin(activity_diff)                # most similar activity
        least_synchronized_neighbor = np.argmax(activity_diff_neighbors)    # least similar neighbor

        # 3a. If there is a connection between the pivot and the candidate already, do nothing
        if self.adjacency_matrix[pivot, candidate]:
            return

        # 3b. If there is no connection between the pivot and the candidate, establish it, and break the connection between the pivot and its least synchronized neighbor.
        self.add_connection(pivot, candidate)
        self.remove_connection(pivot, least_synchronized_neighbor)
        self.metrics.increment_rewiring_count(pivot, from_node=least_synchronized_neighbor, to_node=candidate, adjacency_matrix=self.adjacency_matrix, step=step)

    # Update the activity of all nodes
    def update_network(self, step):
        self.update_activity()
        self.rewire(step)

    def apply_forces(self, effective_iterations=1):
        self.positions = self.physics.apply_forces(self.adjacency_matrix, self.positions, effective_iterations)
    
    def get_adjacency_matrix(self):
        return self.adjacency_matrix