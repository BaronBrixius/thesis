import cupy as cp
from network_simulation.metrics import Metrics
from network_simulation.utils import start_timing, stop_timing

class NodeNetwork:
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, random_seed=None):
        # Seed for reproducibility
        cp.random.seed(random_seed)

        # Params to store
        self.num_nodes = num_nodes
        self.num_connections = num_connections
        self.alpha = alpha
        self.epsilon = epsilon

        # Construct network
        self.activities = cp.random.uniform(-0.7, 1.0, num_nodes)
        self.adjacency_matrix = cp.zeros((num_nodes, num_nodes), dtype=cp.int8)
        self._initialize_network(num_connections)

    def _initialize_network(self, num_connections):
        edges = cp.random.choice(self.num_nodes * self.num_nodes, num_connections)
        row = edges // self.num_nodes
        col = edges % self.num_nodes
        mask = row != col
        row, col = row[mask], col[mask]
        self.adjacency_matrix[row, col] = 1
        self.adjacency_matrix[col, row] = 1

    def update_activity(self):
        start_timing("update_activity1a")
        # Average activities of neighbors
        neighbor_sums = cp.dot(self.adjacency_matrix, self.activities)
        degrees = self.adjacency_matrix.sum(axis=1)
        avgs1 = neighbor_sums / degrees
        # degrees_safe = cp.where(degrees == 0, 1, degrees)  # Avoid division by zero #TODO
        stop_timing("update_activity1a")
        adjacency_nan = cp.where(self.adjacency_matrix == 1, 1.0, cp.nan)
        start_timing("update_activity1b")
        average_neighbor_activity = cp.nanmean(adjacency_nan * self.activities, axis=1)
        stop_timing("update_activity1b")
        # print()
        # print('adjacency_matrix', self.adjacency_matrix)
        # print('activities', self.activities)
        # print('1', avgs1, '2', average_neighbor_activity)
        start_timing("update_activity2")
        self.activities = (
                (1 - self.epsilon) * self.activities +
                self.epsilon * average_neighbor_activity
        )

        stop_timing("update_activity2")

        start_timing("update_activity3")
        # Apply logistic map
        self.activities = 1 - self.alpha * (self.activities)**2
        stop_timing("update_activity3")

    def rewire(self, pivot):
        start_timing("rewire1")
        pivot_neighbors = cp.flatnonzero(self.adjacency_matrix[pivot])
        while pivot_neighbors.size == 0:            # Select another pivot if no neighbors, very rarely happens in practice
            nodes_with_neighbors = cp.where(cp.sum(self.adjacency_matrix, axis=1) > 0)[0]
            if len(nodes_with_neighbors) == 0:
                return  # No rewiring possible if no nodes have neighbors
            pivot = cp.random.choice(nodes_with_neighbors, 1)
            pivot_neighbors = cp.where(self.adjacency_matrix[pivot])[0]
        stop_timing("rewire1")
        # 2. From all other units, select the one that is most synchronized (henceforth: candidate) and least synchronized neighbor
        start_timing("rewire2")
        activity_diff = cp.abs(self.activities - self.activities[pivot])
        activity_diff[pivot] = cp.inf                                   # stop the pivot from connecting to itself
        stop_timing("rewire2")
        start_timing("rewire3")
        candidate = cp.argmin(activity_diff)    # TODO ties? (currently first one is selected)
        stop_timing("rewire3")
        start_timing("rewire4")
        least_similar_neighbor = pivot_neighbors[cp.argmax(activity_diff[pivot_neighbors])]     # least similar neighbor
        stop_timing("rewire4")
        start_timing("rewire5")
        # Update adjacency matrix
        if not self.adjacency_matrix[pivot, candidate]:
            self.adjacency_matrix[pivot, least_similar_neighbor] = self.adjacency_matrix[least_similar_neighbor, pivot] = False
            self.adjacency_matrix[candidate, pivot] = self.adjacency_matrix[pivot, candidate] = True
        stop_timing("rewire5")

    def update_network(self, iterations=1000):
        start_timing("random_indices")
        random_indices = cp.random.randint(0, self.num_nodes, size=iterations)  # Pre-generate random indices
        stop_timing("random_indices")
        for i in range(iterations):
            self.update_activity() 
            self.rewire(random_indices[i])

    def get_adjacency_matrix(self):
        return cp.asnumpy(self.adjacency_matrix)

    def get_activities(self):
        return cp.asnumpy(self.activities)
