import numpy as np

class NodeNetwork:
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, random_seed=None):
        # Seed for reproducibility
        np.random.seed(random_seed)

        # Params to store
        self.num_nodes = num_nodes
        self.num_connections = num_connections
        self.alpha = alpha
        self.epsilon = epsilon

        # Preallocate reused arrays
        self.degrees = np.zeros(num_nodes, dtype=int)
        self.shuffled_indices = np.arange(num_nodes)

        # Construct network
        self.activities = np.random.uniform(-0.7, 1.0, num_nodes)
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)
        self.add_random_connections(num_connections)

    def add_random_connections(self, num_connections_to_add):
        """Add random connections to the graph."""
        edges = set()
        while len(edges) < num_connections_to_add:
            v1 = np.random.randint(0, self.num_nodes)
            v2 = np.random.randint(0, self.num_nodes)
            if v1 != v2 and (v1, v2) not in edges and (v2, v1) not in edges:
                edges.add((v1, v2))

        for edge in edges:
            self.adjacency_matrix[edge[0], edge[1]] = self.adjacency_matrix[edge[1], edge[0]] = True
            self.degrees[edge[0]] += 1
            self.degrees[edge[1]] += 1

    def update_activity(self):
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

    def rewire(self):
        # Select a pivot node
        pivot = np.random.randint(self.num_nodes)
        pivot_neighbors = np.where(self.adjacency_matrix[pivot])[0]
        if len(pivot_neighbors) == 0:            # Select another pivot if no neighbors, very rarely happens in practice
            nodes_with_neighbors = np.where(self.degrees > 0)[0]
            if len(nodes_with_neighbors) == 0:
                return  # No rewiring possible if no nodes have neighbors
            pivot = np.random.choice(nodes_with_neighbors, 1)
            pivot_neighbors = np.where(self.adjacency_matrix[pivot])[0]

        # 2. From all other units, select the one that is most synchronized (henceforth: candidate) and least synchronized neighbor
        activity_diff = np.abs(self.activities - self.activities[pivot])
        activity_diff[pivot] = np.inf                                   # stop the pivot from connecting to itself
        candidate = np.argmin(activity_diff)        # Find the index of the minimum in the shuffled array
        least_similar_neighbor = pivot_neighbors[np.argmax(activity_diff[pivot_neighbors])]     # least similar neighbor

        # 3a. If there is a connection between the pivot and the candidate already, do nothing
        if self.adjacency_matrix[pivot, candidate]:
            return
        self.swap_edge(pivot, least_similar_neighbor, candidate)

    def swap_edge(self, pivot, old_target, new_target):
        # Update adjacency matrix
        self.adjacency_matrix[pivot, old_target] = self.adjacency_matrix[old_target, pivot] = False
        self.adjacency_matrix[pivot, new_target] = self.adjacency_matrix[new_target, pivot] = True

        # Update degrees
        self.degrees[old_target] -= 1
        self.degrees[new_target] += 1

    def update_network(self):
        self.update_activity()
        self.rewire()
