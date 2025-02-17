from network_simulation.network import NodeNetwork
from network_simulation.csvwriter import CSVWriter
from network_simulation.visualization import ColorBy, Visualization
from network_simulation.blockmodel import BlockModel
from network_simulation.metrics import Metrics

class Simulation:
    def __init__(self, num_nodes, num_connections, color_by=ColorBy.ACTIVITY, output_dir=None, alpha=1.7, epsilon=0.4, random_seed=None):
        self.network = NodeNetwork(num_nodes=num_nodes, num_connections=num_connections, alpha=alpha, epsilon=epsilon, random_seed=random_seed)
        adjacency_matrix, activities = self.network.get_adjacency_matrix(), self.network.get_activities()
        self.output = CSVWriter(output_dir)
        self.block_model = BlockModel(adjacency_matrix)
        self.visualization = Visualization(adjacency_matrix, activities, output_dir=output_dir, color_by=color_by, graph=self.block_model.get_graph(), community_assignments=self.block_model.get_community_assignments())

    def run(self, num_steps, display_interval=1000, metrics_interval=1000):
        # Main Loop
        step = 0
        while step < num_steps:
            iterations = min(num_steps - step, display_interval - step % display_interval, metrics_interval - step % metrics_interval)
            adjacency_matrix, activities = self.network.update_network(iterations)
            self._handle_output(adjacency_matrix, activities, step, display_interval, metrics_interval)
            step += iterations

        # Final Metrics
        self._handle_output(adjacency_matrix, activities, num_steps, display_interval, metrics_interval)        

    def _handle_output(self, adjacency_matrix, activities, step, display_interval, metrics_interval):
        """Checks and handles display and metrics intervals."""
        if metrics_interval and step % metrics_interval == 0:
            self.block_model.update_block_model(adjacency_matrix, step)
            self.output.write_metrics_line(Metrics.compute_metrics(adjacency_matrix, self.block_model.get_graph(), self.block_model.get_entropy(), step, self.block_model.get_community_assignments()))

        if display_interval and step % display_interval == 0:
            self.block_model.update_block_model(adjacency_matrix, step)
            self.visualization.draw_visual(adjacency_matrix, activities, self.block_model.get_community_assignments(), self.block_model.get_graph(), step)
