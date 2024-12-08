from network_simulation.network import NodeNetwork
from network_simulation.output import Output
from network_simulation.visualization import Visualization, ColorBy
import numpy as np
import time

class Simulation:
    def __init__(self, num_nodes, num_connections, output_dir=None, alpha=1.7, epsilon=0.4, random_seed=None):
        self.network = NodeNetwork(num_nodes=num_nodes, num_connections=num_connections, alpha=alpha, epsilon=epsilon, random_seed=random_seed)
        self.output = Output(output_dir, num_nodes=num_nodes, num_connections=num_connections)

    def run(self, num_steps, display_interval=1000, metrics_interval=1000, show=True, color_by:ColorBy=ColorBy.ACTIVITY):
        if display_interval:
            self.plot = Visualization(self.network.positions, self.network.activities, self.network.adjacency_matrix, show=show, color_by=color_by)

        start = time.time()
        self.output.logger.info(f"Starting with Nodes: {self.network.num_nodes}, Connections: {self.network.num_connections}, Steps: {num_steps}")

        # Main Loop
        for step in range(num_steps):
            if self.network.stabilized == True:
                self.output.logger(f"Stabilized after {step} iterations.")
                break

            if metrics_interval and step % metrics_interval == 0:
                self.output.output_state_snapshot(step, self.network.activities, self.network.adjacency_matrix, self.network.successful_rewirings)
                self.network.successful_rewirings = 0   #FIXME this is a bad, hacky way to do this!

            if display_interval and step % display_interval == 0:
                self.network.apply_forces(min(100, display_interval))
                self.plot.update_plot(self.network.positions, self.network.activities, self.network.adjacency_matrix, title=f"{self.network.num_nodes} Nodes, {self.network.num_connections} Connections, Generation {step}")
                self.output.output_network_image(self.plot, step)

            self.network.update_network()

        # Final metrics and outputs after the main loop ends
        if display_interval:
            self.network.apply_forces(min(150, display_interval))
            self.plot.update_plot(self.network.positions, self.network.activities, self.network.adjacency_matrix, title=f"{self.network.num_nodes} Nodes, {self.network.num_connections} Connections, Generation {num_steps}")
            self.output.output_network_image(self.plot, num_steps)

        if metrics_interval:
            self.output.output_state_snapshot(num_steps, self.network.activities, self.network.adjacency_matrix, self.network.successful_rewirings)

        self.output.logger.info(f"Run over: {time.time() - start}")
        
        if metrics_interval:
            self.output.post_run_output()

        self.output.logger.info(f"End time: {time.time() - start}")