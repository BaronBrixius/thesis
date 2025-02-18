import cupy as cp

class NodeNetwork:
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, random_seed=None):
        cp.random.seed(random_seed)
        self.num_nodes = num_nodes
        self.num_connections = num_connections
        self.alpha = alpha
        self.epsilon = epsilon
        self.activities = cp.random.uniform(-0.7, 1.0, num_nodes, dtype=cp.float32)
        self.adjacency_matrix = cp.zeros((num_nodes, num_nodes), dtype=cp.int8)
        self._initialize_network(num_connections)

        self.module = cp.RawModule(code=f"""
        extern "C" __global__ void network_update(
            char* adj, float* act, int* rand_idx, int iterations, int* degrees) {{

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= {num_nodes}) return;

            for (int iter = 0; iter < iterations; ++iter) {{
                float sum_neighbors = 0.0;
                for (int j = 0; j < {num_nodes}; ++j) {{
                    if (adj[idx * {num_nodes} + j] == 1) {{
                        sum_neighbors += act[j];
                    }}
                }}
                float avg_activity = (degrees[idx] > 0) ? sum_neighbors / degrees[idx] : act[idx];
                act[idx] = (1 - {epsilon}) * act[idx] + {epsilon} * avg_activity;
                act[idx] = 1.0 - {alpha} * (act[idx] * act[idx]);
                __syncthreads();

                int pivot = rand_idx[iter];
                if (idx == pivot) {{
                    int max_diff_idx = 0, min_diff_idx = 0;
                    float max_diff = -1.0, min_diff = 1e9;
                    for (int j = 0; j < {num_nodes}; ++j) {{
                        float diff = fabsf(act[pivot] - act[j]);
                        if (adj[pivot * {num_nodes} + j] && diff > max_diff) {{
                            max_diff = diff;
                            max_diff_idx = j;
                        }}
                        if (j != pivot && !adj[pivot * {num_nodes} + j] && diff < min_diff) {{
                            min_diff = diff;
                            min_diff_idx = j;
                        }}
                    }}
                    if (max_diff_idx != min_diff_idx) {{
                        adj[pivot * {num_nodes} + max_diff_idx] = 0;
                        adj[max_diff_idx * {num_nodes} + pivot] = 0;
                        adj[pivot * {num_nodes} + min_diff_idx] = 1;
                        adj[min_diff_idx * {num_nodes} + pivot] = 1;
                        degrees[max_diff_idx] -= 1;
                        degrees[min_diff_idx] += 1;
                    }}
                }}
                __syncthreads();
            }}
        }}
        """)
        self.network_update = self.module.get_function('network_update')

    def _initialize_network(self, num_connections):
        edges = cp.random.choice(self.num_nodes * self.num_nodes, num_connections)
        row = edges // self.num_nodes
        col = edges % self.num_nodes
        mask = row != col
        row, col = row[mask], col[mask]
        self.adjacency_matrix[row, col] = 1
        self.adjacency_matrix[col, row] = 1
        self.degrees = cp.sum(self.adjacency_matrix, axis=1, dtype=cp.int32)

    def update_network(self, iterations=1000):
        random_indices = cp.random.randint(0, self.num_nodes, size=iterations)
        block_size = min(1024, self.num_nodes)
        grid_size = (self.num_nodes + block_size - 1) // block_size

        self.network_update(
            (grid_size,), (block_size,),
            (self.adjacency_matrix.data, self.activities.data, random_indices.data, iterations, self.degrees.data)
        )
        return self.get_adjacency_matrix(), self.get_activities()

    def get_adjacency_matrix(self):
        return cp.asnumpy(self.adjacency_matrix)

    def get_activities(self):
        return cp.asnumpy(self.activities)
