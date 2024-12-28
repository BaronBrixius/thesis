import logging
from network_simulation.network import NodeNetwork
from network_simulation.output import Output
from graph_tool.draw import graph_draw
import numpy as np
import os
from network_simulation.visualization import Visualization

class Simulation:
    def __init__(self, num_nodes, num_connections, output_dir=None, alpha=1.7, epsilon=0.4, random_seed=None):
        self.logger = logging.getLogger(__name__)
        self.network = NodeNetwork(num_nodes=num_nodes, num_connections=num_connections, alpha=alpha, epsilon=epsilon, random_seed=random_seed)
        self.output = Output(output_dir, num_nodes=num_nodes, num_connections=num_connections)

        # Initialize Visualization
        self.visualization = Visualization(
            graph=self.network.graph,
            activities=self.network.activities,
            cluster_assignments=self.network.metrics.get_cluster_assignments(self.network.graph),
            output_dir=output_dir
        )

    def run(self, num_steps, display_interval=1000, metrics_interval=1000, show=False, color_by="cluster"):
        self.logger.info(f"Starting with Nodes: {self.network.num_nodes}, Connections: {self.network.num_connections}, Steps: {num_steps}")

        # Main Loop
        for step in range(num_steps):
            self._step(step, display_interval, metrics_interval)

        self._finalize_simulation(num_steps, display_interval, metrics_interval)

    def _step(self, step, display_interval, metrics_interval):
        """Processes a single simulation step."""
        if metrics_interval and step % metrics_interval == 0:
            self.output.write_metrics_line(step, self.network)
            self.network.metrics.reset_rewiring_count()

        if display_interval and step % display_interval == 0:
            self._update_visualization(step)

        self.network.update_network(step)

    def _update_visualization(self, step):
        """Update and save the visualization."""
        self.visualization.cluster_assignments = self.network.metrics.get_cluster_assignments(self.network.graph, step)
        self.visualization.activities = self.network.activities
        self.visualization.refresh_visual()
        self.visualization.save_visual(step)

    def _finalize_simulation(self, num_steps, display_interval, metrics_interval):
        """Handles final outputs after the simulation loop ends."""
        if metrics_interval:
            self.output.write_metrics_line(num_steps, self.network)
            self.network.metrics.reset_rewiring_count()
