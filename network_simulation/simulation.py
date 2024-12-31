import logging
from network_simulation.network import NodeNetwork
from network_simulation.output import Output
from network_simulation.visualization import ColorBy
from network_simulation.visualization import Visualization

class Simulation:
    def __init__(self, num_nodes, num_connections, color_by=ColorBy.ACTIVITY, output_dir=None, alpha=1.7, epsilon=0.4, random_seed=None):
        self.logger = logging.getLogger(__name__)
        self.network = NodeNetwork(num_nodes=num_nodes, num_connections=num_connections, alpha=alpha, epsilon=epsilon, random_seed=random_seed)
        self.output = Output(output_dir, num_nodes=num_nodes, num_connections=num_connections)
        cluster_assignments = self.network.metrics.get_cluster_assignments(self.network.graph, step=0),   # FIXME this shouldn't be needed but it is, something about the  assignment caching probably
        self.visualization = Visualization(network=self.network, output_dir=output_dir, color_by=color_by)

    def run(self, num_steps, display_interval=1000, metrics_interval=1000):
        self.logger.info(f"Starting with Nodes: {self.network.num_nodes}, Connections: {self.network.num_connections}, Steps: {num_steps}")

        # Main Loop
        for step in range(num_steps):
            self._step(step, display_interval, metrics_interval)

        self._finalize_simulation(num_steps, display_interval, metrics_interval)

    def _step(self, step, display_interval, metrics_interval):
        """Processes a single simulation step."""
        if metrics_interval and step % metrics_interval == 0:
            self.output.write_metrics_line(step, self.network)
            self.network.metrics.reset_rewiring_counts()

        if display_interval and step % display_interval == 0:
            self._update_visualization(step, display_interval)

        self.network.update_network(step)

    def _update_visualization(self, step, display_interval):
        """Update and save the visualization."""
        self.visualization.refresh_visual(self.network, step, max_iter=min(25, display_interval))
        self.visualization.save_visual(self.network, step)

    def _finalize_simulation(self, num_steps, display_interval, metrics_interval):
        """Handles final outputs after the simulation loop ends."""
        if metrics_interval:
            self.output.write_metrics_line(num_steps, self.network)
            self.network.metrics.reset_rewiring_counts()

        if display_interval:
            self._update_visualization(num_steps, display_interval)
