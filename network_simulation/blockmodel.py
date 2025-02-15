from graph_tool.all import Graph
from graph_tool.inference import PPBlockState
import numpy as np

class BlockModel:
    def __init__(self, adjacency_matrix):
        self.block_state = PPBlockState(Graph(g=np.transpose(np.nonzero(adjacency_matrix)), directed=False))
        self.last_update_step = -1

    def update_block_model(self, adjacency_matrix, step: int, max_sweeps=5):
        if step > self.last_update_step:    # Only update if the adjacency matrix has changed
            # Update the graph with the latest adjacency matrix
            graph = self.block_state.g
            graph.clear_edges()
            graph.add_edge_list(np.transpose(np.nonzero(adjacency_matrix)))  

            # Recreate the block state to reflect the new graph
            self.block_state = PPBlockState(g=graph, b=self.block_state.b) 

            # MCMC sweeps to update the community assignments
            for _ in range(max_sweeps):
                entropy_delta, _, _ = self.block_state.multilevel_mcmc_sweep()
                if entropy_delta == 0:
                    break

            self.last_update_step = step

    def get_community_assignments(self):
        return self.block_state.get_blocks().a

    def get_entropy(self):
        return self.block_state.entropy()

    def get_graph(self):
        return self.block_state.g