import numpy as np
from network_simulation.physics import NetworkPhysics
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

        positions = np.random.uniform([0.1, 0.1], [0.9, 0.9], (num_nodes, 2))
        normal_distance = 0.5 * np.sqrt(self.num_connections + self.num_nodes) / self.num_nodes
        self.physics = NetworkPhysics(self.adjacency_matrix, positions, normal_distance)

        self.initialize_connections(num_connections)    # Add initial connections

        self.cpl_history = []
        self.breakup_count = 0
        self.cc_history = []
        self.stabilization_threshold = stabilization_threshold
        self.stabilized = False # Relatable

    # Initialize random connections between the nodes
    def initialize_connections(self, num_connections):
        possible_pairs = [(i, j) for i in range(self.num_nodes) for j in range(i+1, self.num_nodes)]
        np.random.shuffle(possible_pairs)

        for i, j in possible_pairs[:num_connections]:
            self.add_connection(i, j)

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

    def calculate_stats(self):
        char_path_length = self.characteristic_path_length()
        avg_clustering = self.clustering_coefficient()

        if not self.stabilization_threshold:
            return char_path_length, avg_clustering

        # Add the new CPL and CC values to history
        self.cpl_history.append(char_path_length)
        self.cc_history.append(avg_clustering)

        if len(self.cpl_history) > 100:
            self.cpl_history.pop(0)
        if len(self.cc_history) > 100:
            self.cc_history.pop(0)

        # Check stabilization for both CPL and CC
        if len(self.cpl_history) == 100 and len(self.cc_history) == 100:
            cpl_min, cpl_max = min(self.cpl_history), max(self.cpl_history)
            cpl_stable = (cpl_max - cpl_min) / cpl_max <= self.stabilization_threshold

            cc_min, cc_max = min(self.cc_history), max(self.cc_history)
            cc_stable = (cc_max - cc_min) / cc_max <= self.stabilization_threshold

            # If both CPL and CC are stable, mark the network as stabilized
            self.stabilized = cpl_stable and cc_stable

        return char_path_length, avg_clustering

    def apply_forces(self, effective_iterations=1):
        self.physics.apply_forces(self.adjacency_matrix, effective_iterations)