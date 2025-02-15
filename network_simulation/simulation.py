from network_simulation.network import NodeNetwork
from network_simulation.csvwriter import Output
from network_simulation.visualization import ColorBy, Visualization

class Simulation:
    def __init__(self, num_nodes, num_connections, color_by=ColorBy.ACTIVITY, output_dir=None, alpha=1.7, epsilon=0.4, random_seed=None):
        self.network = NodeNetwork(num_nodes=num_nodes, num_connections=num_connections, alpha=alpha, epsilon=epsilon, random_seed=random_seed)
        self.output = Output(output_dir)
        self.visualization = Visualization(network=self.network, output_dir=output_dir, color_by=color_by)

    def run(self, num_steps, display_interval=1000, metrics_interval=1000):
        # Main Loop
        for step in range(num_steps):
            self._step(step, display_interval, metrics_interval)

        # Final Metrics
        self._handle_output(num_steps, display_interval, metrics_interval)

    def _step(self, step, display_interval, metrics_interval):
        """Processes a single simulation step."""
        self._handle_output(step, display_interval, metrics_interval)
        self.network.update_network()

    def _handle_output(self, step, display_interval, metrics_interval):
        """Checks and handles display and metrics intervals."""
        if metrics_interval and step % metrics_interval == 0:
            self.output.write_metrics_line(self.network.metrics.compute_metrics(self.network.adjacency_matrix, step))
            # mempool = cp.get_default_memory_pool()
            # pinned_mempool = cp.get_default_pinned_memory_pool()
            # print(f"Memory: {mempool.used_bytes()} used, {mempool.total_bytes()} total")
            # print(f"Pinned blocks free: {pinned_mempool.n_free_blocks()} blocks")

        if display_interval and step % display_interval == 0:
            self.visualization.draw_visual(self.network, step)
