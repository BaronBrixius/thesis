import numpy as np

class NodeNetwork:
    def __init__(self, num_nodes, num_edges, alpha=1.7, epsilon=0.4, random_seed=None, process_num=None):
        # Parameters
        np.random.seed(random_seed)
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.alpha = alpha
        self.epsilon = epsilon

        # Initialize network
        self.activities = np.random.uniform((1.0 - self.alpha), 1.0, self.num_nodes)    # Random initial activity
        self.adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=bool)
        self._add_random_edges(self.num_edges)
        self.degrees = np.sum(self.adjacency_matrix, axis=1)            # We track this instead of recalculating it every step

    def _add_random_edges(self, num_edges_to_add):
        """Add random edges to the network."""
        all_possible_edges = np.array(np.triu_indices(self.num_nodes, k=1)).T
        available_edges = all_possible_edges[~self.adjacency_matrix[all_possible_edges[:, 0], all_possible_edges[:, 1]]]
        edges_to_add = available_edges[np.random.choice(len(available_edges), size=num_edges_to_add, replace=False)]

        self.adjacency_matrix[edges_to_add[:, 0], edges_to_add[:, 1]] = True
        self.adjacency_matrix[edges_to_add[:, 1], edges_to_add[:, 0]] = True

    def _update_activity(self):
        # Sum up neighbor activities
        neighbor_sums = np.einsum("ij,j->i", self.adjacency_matrix, self.activities)

        # Split activity between neighbors (determined by epsilon)
        connected_nodes = self.degrees > 0
        self.activities[connected_nodes] = (
            (1 - self.epsilon)  * self.activities[connected_nodes] + 
            self.epsilon        * neighbor_sums[connected_nodes] / self.degrees[connected_nodes]
        )
        # Apply logistic map
        self.activities = 1 - self.alpha * (self.activities)**2

    def _rewire(self):
        # 1. Pick a unit at random (henceforth: pivot)
        pivot = np.random.randint(self.num_nodes)
        if self.degrees[pivot] == 0:                                    # Rare edge case: picked node has no neighbors
            pivot = np.random.choice(np.flatnonzero(self.degrees))      # Fallback: explicitly choose from nodes with neighbors

        # 2. From all other units, select the one that is most synchronized (henceforth: candidate) and least synchronized neighbor
        activity_diff = np.abs(self.activities - self.activities[pivot])
        activity_diff_neighbors = activity_diff * self.adjacency_matrix[pivot]

        activity_diff[pivot] = np.inf                       # stop the pivot from connecting to itself
        candidate = np.argmin(activity_diff)                # most similar activity
        least_synchronized_neighbor = np.argmax(activity_diff_neighbors)    # least similar neighbor

        # 3a. If there is an edge between the pivot and the candidate already, do nothing
        if self.adjacency_matrix[pivot, candidate] or activity_diff_neighbors[least_synchronized_neighbor] == 0.0:  # Rare edge case: pivot and least_synchronized_neighbor were identical
            return
        self._swap_edge(pivot, least_synchronized_neighbor, candidate)

    def _swap_edge(self, pivot, old_target, new_target):
        # Update adjacency matrix
        self.adjacency_matrix[pivot, old_target] = self.adjacency_matrix[old_target, pivot] = False
        self.adjacency_matrix[pivot, new_target] = self.adjacency_matrix[new_target, pivot] = True

        # Update degrees
        self.degrees[old_target] -= 1
        self.degrees[new_target] += 1

    def update_network(self, iterations):
        for _ in range(iterations):
            self._update_activity()
            for _ in range(250):
                self._rewire()

        return self.get_adjacency_matrix(), self.get_activities()

    def get_adjacency_matrix(self):
        return self.adjacency_matrix
    
    def get_activities(self):
        return self.activities
