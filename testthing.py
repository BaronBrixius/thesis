import cupy as cp
from cupyx.profiler import benchmark
import time
from cupy_backends.cuda.api.runtime import profilerStart as start
from cupy_backends.cuda.api.runtime import profilerStop as stop

class NodeNetwork:
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, random_seed=None):
        # Seed for reproducibility
        cp.random.seed(random_seed)

        self.num_nodes = num_nodes
        self.num_connections = num_connections
        self.alpha = alpha
        self.epsilon = epsilon

        # Initialize node activities
        self.activities = cp.random.uniform(-0.7, 1.0, num_nodes)

        # Preallocate reused arrays
        self.vertices = cp.arange(num_nodes)
        self.degrees = cp.zeros(num_nodes, dtype=int)
        self.shuffled_indices = cp.arange(num_nodes)

        self.adjacency_matrix = cp.zeros((num_nodes, num_nodes), dtype=bool)
        self.add_random_connections(num_connections)

    def add_random_connections(self, num_connections_to_add):
        """Add random connections to the graph."""
        num_nodes = self.num_nodes
        edges = set()
        while len(edges) < num_connections_to_add:
            v1 = cp.random.randint(0, num_nodes).item()
            v2 = cp.random.randint(0, num_nodes).item()
            if v1 != v2 and (v1, v2) not in edges and (v2, v1) not in edges:
                edges.add((v1, v2))

        for edge in edges:
            self.adjacency_matrix[edge[0], edge[1]] = self.adjacency_matrix[edge[1], edge[0]] = True
            self.degrees[edge[0]] += 1
            self.degrees[edge[1]] += 1

    def update_activity(self):
        # # Sum up neighbor activities
        # neighbor_sums = cp.einsum("ij,j->i", self.adjacency_matrix, self.activities)
        # # Split activity between neighbors (determined by epsilon)
        # connected_nodes = self.degrees > 0
        # self.activities[connected_nodes] = (
        #     (1 - self.epsilon)  * self.activities[connected_nodes] + 
        #     self.epsilon        * neighbor_sums[connected_nodes] / self.degrees[connected_nodes]
        # )
        # Apply logistic map
        self.activities = 1 - self.alpha * (self.activities)**2

    def rewire(self):
        """Simple rewire function for testing."""
        pivot = cp.random.randint(self.num_nodes).item()
        # pivot_neighbors = cp.where(self.adjacency_matrix[pivot])[0]
        # if len(pivot_neighbors) == 0:
        #     return

        # least_similar_neighbor = pivot_neighbors[0]
        # candidate = (pivot + 1) % self.num_nodes

        # if not self.adjacency_matrix[pivot, candidate]:
        #     self.adjacency_matrix[pivot, least_similar_neighbor] = self.adjacency_matrix[least_similar_neighbor, pivot] = False
        #     self.adjacency_matrix[pivot, candidate] = self.adjacency_matrix[candidate, pivot] = True

        #     self.degrees[least_similar_neighbor] -= 1
        #     self.degrees[candidate] += 1

    def update_network(self):
        for _ in range(1000):
            self.update_activity()
            self.rewire()

# Example usage
network = NodeNetwork(num_nodes=300, num_connections=5000)
# print(benchmark(network.update_network, n_repeat=1))

# # Profiling and memory usage
start_time = time.time()
start()  # Start the profiler
for step in range(1000):
    if step % 100 == 0:
        print(f"Step {step} completed")
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        print(f"Memory: {mempool.used_bytes()} used, {mempool.total_bytes()} total")
        print(f"Pinned blocks free: {pinned_mempool.n_free_blocks()} blocks")
    network.update_network()
stop()  # Stop the profiler
end_time = time.time()

print(f"Total time: {end_time - start_time} seconds")