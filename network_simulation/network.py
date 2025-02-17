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
        print(cp.asnumpy(self.activities))

    def _initialize_network(self, num_connections):
        edges = cp.random.choice(self.num_nodes * self.num_nodes, num_connections)
        row = edges // self.num_nodes
        col = edges % self.num_nodes
        mask = row != col
        row, col = row[mask], col[mask]
        self.adjacency_matrix[row, col] = 1
        self.adjacency_matrix[col, row] = 1

    def _kernel_code(self):
        return """
        extern "C" __global__ void network_update(
            int* adj, float* act, int* rand_idx, int iterations, int n) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= 200) return;
            for (int iter = 0; iter < iterations; ++iter) {
                // Update activity
                float sum_neighbors = 0.0;
                int degree = 0;
                for (int j = 0; j < n; ++j) {
                    if (adj[idx * n + j]) {
                        sum_neighbors += act[j];
                        degree++;
                    }
                }
                float avg_activity = (degree > 0) ? sum_neighbors / degree : act[idx];
                act[idx] = 0.6 * act[idx] + 0.4 * avg_activity;
                act[idx] = 1.0 - 1.7 * (act[idx] * act[idx]);

                __syncthreads();

                // Rewire step
                int pivot = rand_idx[iter];
                if (idx == pivot) {
                    int max_diff_idx = -1;
                    float max_diff = -1.0;
                    for (int j = 0; j < n; ++j) {
                        if (adj[pivot * n + j]) {
                            float diff = fabsf(act[pivot] - act[j]);
                            if (diff > max_diff) {
                                max_diff = diff;
                                max_diff_idx = j;
                            }
                        }
                    }

                    int min_diff_idx = -1;
                    float min_diff = 1e9;
                    for (int j = 0; j < n; ++j) {
                        if (!adj[pivot * n + j]) {
                            float diff = fabsf(act[pivot] - act[j]);
                            if (diff < min_diff) {
                                min_diff = diff;
                                min_diff_idx = j;
                            }
                        }
                    }

                    if (max_diff_idx != -1 && min_diff_idx != -1) {
                        adj[pivot * n + max_diff_idx] = 0;
                        adj[max_diff_idx * n + pivot] = 0;
                        adj[pivot * n + min_diff_idx] = 1;
                        adj[min_diff_idx * n + pivot] = 1;
                    }
                }
                __syncthreads();
            }

        }
        """

    def update_network(self, iterations=1000):
        kernel_code = self._kernel_code()
        module = cp.RawModule(code=kernel_code)
        network_update = module.get_function('network_update')

        random_indices = cp.random.randint(0, self.num_nodes, size=iterations)

        block_size = 200
        grid_size = 1

        network_update(
            (grid_size,), (block_size,),
            (self.adjacency_matrix, self.activities, random_indices, 1, 200)
        )
        cp.cuda.Device(0).synchronize()
        print(cp.asnumpy(self.activities))

    def get_adjacency_matrix(self):
        return cp.asnumpy(self.adjacency_matrix)

    def get_activities(self):
        return cp.asnumpy(self.activities)
