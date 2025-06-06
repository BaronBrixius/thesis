from network.network_cpu import NodeNetwork
from runtime_output.csvwriter import CSVWriter
from runtime_output.visualization import ColorBy, Visualization
from runtime_output.blockmodel import BlockModel
import runtime_output.metrics as Metrics

class Simulation:
    def __init__(self, num_nodes, num_edges, color_by=ColorBy.ACTIVITY, simulation_dir=None, alpha=1.7, epsilon=0.4, display_interval=1000, metrics_interval=1000, random_seed=None, process_num=0):
        # Simulation Parameters
        self.display_interval = display_interval
        self.metrics_interval = metrics_interval

        # Initialize Network
        self.network = NodeNetwork(num_nodes, num_edges, alpha, epsilon, random_seed, process_num)

        # Initialize Simulation Components
        adjacency_matrix, activities = self.network.get_adjacency_matrix(), self.network.get_activities()
        self.output = CSVWriter(simulation_dir)
        self.block_model = BlockModel(adjacency_matrix)
        self.visualization = Visualization(adjacency_matrix, activities, self.block_model.get_graph(), self.block_model.get_community_assignments(), simulation_dir, color_by)

        # Initial Output
        self._handle_output(adjacency_matrix, activities, self.block_model, 0)

    def run(self, num_steps):
        """Main loop for the simulation."""
        step = 0
        while step < num_steps:
            # Update Network
            iterations_to_next_interval = min(num_steps - step, self.display_interval - step % self.display_interval if self.display_interval else float('inf'), self.metrics_interval - step % self.metrics_interval if self.metrics_interval else float('inf'))
            adjacency_matrix, activities = self.network.update_network(iterations_to_next_interval)
            step += iterations_to_next_interval

            # Update Block Model
            self.block_model.update_block_model(adjacency_matrix, step)

            # Output Metrics and Visualization
            self._handle_output(adjacency_matrix, activities, self.block_model, step)

        # Finalize
        self.output.close()
        self.visualization.close()

    def _handle_output(self, adjacency_matrix, activities, block_model, step):
        """Checks and handles display and metrics intervals."""
        if self.metrics_interval and step % self.metrics_interval == 0:
            self.output.write_metrics_line(Metrics.compute_all_metrics(adjacency_matrix, block_model.get_graph(), block_model.get_entropy(), block_model.get_community_assignments(), step))

        if self.display_interval and step % self.display_interval == 0:
            self.visualization.draw_visual(adjacency_matrix, activities, block_model.get_graph(), block_model.get_community_assignments(), step, max_iter=self.display_interval)
