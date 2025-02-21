from network_simulation.network import NodeNetwork
from network_simulation.csvwriter import CSVWriter
from network_simulation.visualization import ColorBy, Visualization
from network_simulation.blockmodel import BlockModel
import network_simulation.metrics as Metrics

class Simulation:
    def __init__(self, num_nodes, num_connections, color_by=ColorBy.ACTIVITY, simulation_dir=None, alpha=1.7, epsilon=0.4, random_seed=None):
        self.network = NodeNetwork(num_nodes, num_connections, alpha, epsilon, random_seed)
        adjacency_matrix, activities = self.network.get_adjacency_matrix(), self.network.get_activities()
        self.output = CSVWriter(simulation_dir)
        self.block_model = BlockModel(adjacency_matrix)
        self.visualization = Visualization(adjacency_matrix, activities, self.block_model.get_graph(), self.block_model.get_community_assignments(), simulation_dir, color_by)

    def run(self, num_steps, display_interval=1000, metrics_interval=1000):
        adjacency_matrix, activities = self.network.get_adjacency_matrix(), self.network.get_activities()

        # Main Loop
        step = 0
        while step < num_steps:
            # Output Metrics and Visualization
            self._handle_output(adjacency_matrix, activities, self.block_model, step, display_interval, metrics_interval)

            # Update Network
            iterations_to_next_interval = min(num_steps - step, display_interval - step % display_interval if display_interval else float('inf'), metrics_interval - step % metrics_interval if metrics_interval else float('inf'))
            adjacency_matrix, activities = self.network.update_network(iterations_to_next_interval)
            step += iterations_to_next_interval

            # Update Block Model
            self.block_model.update_block_model(adjacency_matrix, step)

        # Final Output
        self._handle_output(adjacency_matrix, activities, self.block_model, step, display_interval, metrics_interval)
        self.output.close()

    def _handle_output(self, adjacency_matrix, activities, block_model, step, display_interval, metrics_interval):
        """Checks and handles display and metrics intervals."""
        if metrics_interval and step % metrics_interval == 0:
            self.output.write_metrics_line(Metrics.compute_metrics(adjacency_matrix, block_model.get_graph(), block_model.get_entropy(), block_model.get_community_assignments(), step))

        if display_interval and step % display_interval == 0:
            self.visualization.draw_visual(adjacency_matrix, activities, block_model.get_graph(), block_model.get_community_assignments(), step, max_iter=display_interval)
