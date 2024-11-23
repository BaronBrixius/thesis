from network_simulation.network import NodeNetwork
from network_simulation.output_manager import OutputManager
from network_simulation.visualization import NetworkPlot
import numpy as np
import time

class SimulationManager:
    def __init__(self, num_nodes, num_connections, output_dir=None, alpha=1.7, epsilon=0.4, random_seed=None):
        self.network = NodeNetwork(num_nodes=num_nodes, num_connections=num_connections, alpha=alpha, epsilon=epsilon, random_seed=random_seed)
        self.output_manager = OutputManager(output_dir)

    def metrics(self, step, start):
        characteristic_path_length, clustering_coefficient = self.network.calculate_stats()
        time_since_start = time.time() - start
        print(f"Iteration {step}: CPL={characteristic_path_length}, CC={clustering_coefficient}, Breakups={self.network.breakup_count}, Time={time_since_start:.2f}")

        self.output_manager.save_stats(step, characteristic_path_length, clustering_coefficient, self.network.breakup_count, time_since_start)
        self.output_manager.save_matrix(self.network.adjacency_matrix, step)
        self.output_manager.save_histogram(np.sum(self.network.adjacency_matrix, axis=1), step)
        self.output_manager.save_activities(self.network.activities, step)

    def run(self, num_steps, display_interval=1000, metrics_interval=1000, show=True):
        start = time.time()

        if display_interval:
            self.plot = NetworkPlot(self.network.physics.positions, self.network.activities, self.network.adjacency_matrix, show)

        # Main Loop
        for step in range(num_steps):
            if self.network.stabilized == True:
                print(f"Stabilized after {step} iterations.")
                break

            self.network.update_network()            

            if step % metrics_interval == 0:
                self.metrics(step, start)

            if display_interval and step % display_interval == 0:
                self.network.apply_forces(min(25, display_interval))
                self.plot.update_plot(self.network.physics.positions, self.network.activities, self.network.adjacency_matrix, title=f"{self.network.num_nodes} Nodes, {self.network.num_connections} Connections, Generation {step}")
                self.output_manager.save_network_image(self.plot, step)

        # Final metrics and outputs after the main loop ends
        self.metrics(step, start)

        if display_interval:
            self.network.apply_forces(min(150, display_interval))
            self.plot.update_plot(self.network.physics.positions, self.network.activities, self.network.adjacency_matrix, title=f"{self.network.num_nodes} Nodes, {self.network.num_connections} Connections, Generation {step}")
            self.output_manager.save_network_image(self.plot, step)