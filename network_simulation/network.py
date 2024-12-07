from collections import deque
import numpy as np
from network_simulation.metrics import Metrics
from network_simulation.physics import Physics
from scipy.sparse.csgraph import shortest_path

class NodeNetwork:
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, stabilization_threshold=None, random_seed=None):
        np.random.seed(random_seed)
        
        self.num_nodes = num_nodes
        self.num_connections = num_connections
        self.alpha = alpha
        self.epsilon = epsilon

        self.activities = np.random.uniform(-1, 1, num_nodes)   # Random initial activity
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)

        self.positions = np.random.uniform([0.1, 0.1], [0.9, 0.9], (num_nodes, 2))
        normal_distance = 0.5 * np.sqrt(self.num_connections + self.num_nodes) / self.num_nodes
        self.physics = Physics(normal_distance)

        # Add initial connections
        self.add_random_connections(num_connections)

        self.metrics = Metrics()
        self.breakup_count = 0
        self.successful_rewirings = 0

        self.cpl_history = deque(maxlen=100)
        self.cc_history = deque(maxlen=100)
        self.stabilization_threshold = stabilization_threshold
        self.stabilized = False # Relatable

    def add_random_connections(self, num_connections):
        """Add random connections to the network."""
        possible_edges = np.triu_indices(self.num_nodes, k=1)
        edge_indices = np.random.choice(len(possible_edges[0]), size=num_connections, replace=False)
        for index in edge_indices:
            i, j = possible_edges[0][index], possible_edges[1][index]
            self.adjacency_matrix[i, j] = self.adjacency_matrix[j, i] = 1

    def remove_random_connections(self, num_connections):
        """Remove random connections from the network."""
        existing_edges = np.array(np.where(np.triu(self.adjacency_matrix, k=1))).T
        if len(existing_edges) < num_connections:
            raise ValueError("Not enough connections to remove.")
        edge_indices = np.random.choice(len(existing_edges), size=num_connections, replace=False)
        for index in edge_indices:
            i, j = existing_edges[index]
            self.adjacency_matrix[i, j] = self.adjacency_matrix[j, i] = 0

    def update_node_count(self, new_node_count):
        """Update the number of nodes in the network."""
        if new_node_count > self.num_nodes:
            # Add new nodes
            diff = new_node_count - self.num_nodes
            self.activities = np.append(self.activities, np.random.rand(diff))
            self.adjacency_matrix = np.pad(self.adjacency_matrix, ((0, diff), (0, diff)), mode='constant')

            # Add positions for new nodes along the edges
            new_positions = np.random.rand(diff * 3, 2)  # Generate more candidates than needed to minimize rejection
            distances = np.linalg.norm(new_positions - 0.5, axis=1)  # Calculate distances from center
            valid_positions = new_positions[distances >= 0.35]  # Filter out invalid positions
            self.positions = np.vstack([self.positions, valid_positions[:diff]])  # Add only required positions
        elif new_node_count < self.num_nodes:
            # Remove nodes
            diff = self.num_nodes - new_node_count
            self.activities = self.activities[:new_node_count]
            self.adjacency_matrix = self.adjacency_matrix[:new_node_count, :new_node_count]
            self.positions = self.positions[:new_node_count]

        self.num_nodes = new_node_count

    def update_connection_count(self, new_connection_count):
        """Update the number of connections in the network."""
        current_connections = np.sum(self.adjacency_matrix) // 2
        if new_connection_count > current_connections:
            # Add connections
            self.add_random_connections(new_connection_count - current_connections)
        elif new_connection_count < current_connections:
            # Remove connections
            self.remove_random_connections(current_connections - new_connection_count)

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

    def rewire(self):
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
        self.successful_rewirings += 1

    # Update the activity of all nodes
    def update_network(self):
        self.update_activity()
        self.rewire()

    def characteristic_path_length(self):
        path_lengths = shortest_path(self.adjacency_matrix, directed=False, unweighted=True)
        if np.isinf(path_lengths).any():
            self.breakup_count += 1
        valid_lengths = path_lengths[(path_lengths < np.inf) & (path_lengths > 0)]  # FIXME right now, upon breakup, it removes "infinite" distances then computes the average as if that were okay
        return np.mean(valid_lengths)
    
    def clustering_coefficient(self):
        clustering_coefficients = []
        for i in range(self.num_nodes):
            neighbors = np.where(self.adjacency_matrix[i])[0]
            if len(neighbors) < 2:
                clustering_coefficients.append(0)
                continue
            neighbor_pairs = self.adjacency_matrix[neighbors][:, neighbors]
            connections = np.sum(neighbor_pairs)
            possible_connections = len(neighbors) * (len(neighbors) - 1)
            clustering_coefficients.append(connections / possible_connections)
        return np.mean(clustering_coefficients)

    def check_stabilization(self):
        if len(self.cpl_history) < self.cpl_history.maxlen or len(self.cc_history) < self.cc_history.maxlen:
            return False

        cpl_range = max(self.cpl_history) - min(self.cpl_history)
        cc_range = max(self.cc_history) - min(self.cc_history)

        cpl_stable = (cpl_range / max(self.cpl_history)) <= self.stabilization_threshold
        cc_stable = (cc_range / max(self.cc_history)) <= self.stabilization_threshold

        return cpl_stable and cc_stable

    def calculate_stats(self):
        metrics = self.metrics.calculate_all(self.adjacency_matrix)
        cpl = metrics.get("Characteristic Path Length", float('nan'))
        cc = metrics.get("Clustering Coefficient", float('nan'))

        if np.isnan(cpl):   # Network breakup
            self.stabilized = False
            return metrics

        # Add metrics to history
        self.cpl_history.append(cpl)
        self.cc_history.append(cc)

        # Update stabilization state
        self.stabilized = self.check_stabilization()
        metrics["Stabilized"] = self.stabilized

        return metrics

    def apply_forces(self, effective_iterations=1):
        self.positions = self.physics.apply_forces(self.adjacency_matrix, self.positions, effective_iterations)