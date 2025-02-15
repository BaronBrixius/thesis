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

        # Preallocate reused arrays
        self.vertices = cp.arange(num_nodes)
        self.shuffled_indices = cp.arange(num_nodes)

        # Construct network
        self.activities = cp.random.uniform(-0.7, 1.0, num_nodes)
        self.adjacency_matrix = cp.zeros((num_nodes, num_nodes), dtype=bool)
        self.add_random_connections(num_connections)

    def add_random_connections(self, num_connections_to_add):
        """Add random connections to the graph."""
        edges = set()
        while len(edges) < num_connections_to_add:
            v1 = cp.random.randint(0, self.num_nodes).item()
            v2 = cp.random.randint(0, self.num_nodes).item()
            if v1 != v2 and (v1, v2) not in edges and (v2, v1) not in edges:
                edges.add((v1, v2))

        for edge in edges:
            self.adjacency_matrix[edge[0], edge[1]] = self.adjacency_matrix[edge[1], edge[0]] = True

    def update_activity(self):
        start_timing("update_activity1")
        # Sum up neighbor activities
        neighbor_sums = cp.dot(self.adjacency_matrix, self.activities)
        stop_timing("update_activity1")

        start_timing("update_activity2")
        # Calculate connected nodes and normalized neighbor sums
        degrees = cp.sum(self.adjacency_matrix, axis=1)
        connected_nodes = degrees > 0
        self.activities[connected_nodes] = (
            (1 - self.epsilon) * self.activities[connected_nodes] + 
            self.epsilon * (neighbor_sums[connected_nodes] / degrees[connected_nodes])
        )
        stop_timing("update_activity2")

        start_timing("update_activity3")
        # Apply logistic map
        self.activities = 1 - self.alpha * (self.activities)**2
        stop_timing("update_activity3")

    def rewire(self, pivot):
        start_timing("rewire1b")
        pivot_neighbors = cp.where(self.adjacency_matrix[pivot])[0]
        stop_timing("rewire1b")
        start_timing("rewire1c")
        while len(pivot_neighbors) == 0:            # Select another pivot if no neighbors, very rarely happens in practice
            nodes_with_neighbors = cp.where(cp.sum(self.adjacency_matrix, axis=1) > 0)[0]
            if len(nodes_with_neighbors) == 0:
                return  # No rewiring possible if no nodes have neighbors
            pivot = cp.random.choice(nodes_with_neighbors, 1)
            pivot_neighbors = cp.where(self.adjacency_matrix[pivot])[0]
        stop_timing("rewire1c")
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
        # 3a. If there is a connection between the pivot and the candidate already, do nothing
        if self.adjacency_matrix[pivot, candidate]:
            return
        start_timing("rewire5")
        # Update adjacency matrix
        self.adjacency_matrix[pivot, least_similar_neighbor] = self.adjacency_matrix[least_similar_neighbor, pivot] = False
        self.adjacency_matrix[pivot, candidate] = self.adjacency_matrix[candidate, pivot] = True
        stop_timing("rewire5")

    def update_network(self, iterations):
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
