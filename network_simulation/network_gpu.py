import cupy as cp
import os
os.environ["CUPY_CUDA_PER_THREAD_DEFAULT_STREAM"] = "1"
os.environ["CUPY_GPU_MEMORY_LIMIT"] = "95%"

class NodeNetwork:
    def __init__(self, num_nodes, num_edges, alpha=1.7, epsilon=0.4, random_seed=None, process_num = 0):
        # Cuda setup
        cp.cuda.Device(process_num % cp.cuda.runtime.getDeviceCount()).use()
        cp.random.seed(random_seed)

        # Store params
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        # Initialize network
        self.activities = cp.random.uniform(1.0 - alpha, 1.0, num_nodes, dtype=cp.float32)
        self.adjacency_matrix = cp.zeros((self.num_nodes, self.num_nodes), dtype=cp.int8)
        self._add_random_edges(num_edges)
        self.degrees = cp.sum(self.adjacency_matrix, axis=1, dtype=cp.int32)    # Store degrees for faster computation

        self.network_update = cp.RawKernel(f"""
        extern "C" __global__ void network_update(
            char* adj, float* act, int* rand_idx, int iterations, int* degrees) {{

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= {num_nodes}) return;

            for (int i = 0; i < iterations; ++i) {{
                float sum_neighbors = 0.0;
                for (int j = 0; j < {num_nodes}; ++j) {{
                    if (adj[idx * {num_nodes} + j] == 1) {{
                        sum_neighbors += act[j];
                    }}
                }}
                float avg_activity = (degrees[idx] > 0) ? sum_neighbors / degrees[idx] : act[idx];
                act[idx] = (1.0 - {epsilon}) * act[idx] + {epsilon} * avg_activity;
                act[idx] = 1.0 - {alpha} * (act[idx] * act[idx]);
                __syncthreads();

                int pivot = rand_idx[i];
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
        """, 'network_update')

    def _add_random_edges(self, num_edges):
        possible_edges = cp.array(cp.triu_indices(self.num_nodes, k=1)).T
        selected_edges = possible_edges[cp.random.choice(len(possible_edges), size=num_edges, replace=False)]

        # Create adjacency matrix and set selected edges to 1
        self.adjacency_matrix[selected_edges[:, 0], selected_edges[:, 1]] = 1
        self.adjacency_matrix[selected_edges[:, 1], selected_edges[:, 0]] = 1  # Symmetric

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
