import csv
import os
from network_simulation.network import NodeNetwork

class Output:
    def __init__(self, project_dir, num_nodes=None, num_connections=None):
        self.num_nodes = num_nodes
        self.num_connections = num_connections

        self.project_dir = project_dir
        os.makedirs(project_dir, exist_ok=True)

        metrics_file_path = os.path.join(self.project_dir, f"summary_metrics_nodes_{self.num_nodes}_edges_{self.num_connections}.csv")
        self.metrics_file = open(metrics_file_path, mode="w", newline="")
        self.csv_writer = None

    # Runtime Metrics Writing
    def write_metrics_line(self, step, network: NodeNetwork):
        row = self._compute_row(step, network)

        if self.csv_writer is None: # Lazy loading the writer also allows us to wait and see what headers will be needed
            self.csv_writer = csv.DictWriter(self.metrics_file, fieldnames=row.keys())
            self.csv_writer.writeheader()

        self.csv_writer.writerow(row)

    def _compute_row(self, step, network: NodeNetwork):
        # Compute row data
        row = {
            "Step": step,
            "Clustering Coefficient": network.metrics.calculate_clustering_coefficient(network.graph),
            "Average Path Length": network.metrics.calculate_average_path_length(network.graph),
            "Rich Club Coefficients": network.metrics.calculate_rich_club_coefficients(network.adjacency_matrix),
        }

        # Update with cluster metrics
        row.update(network.metrics.get_cluster_metrics(network.graph, step))

        # Add rewiring metrics
        row.update(
            {f"Rewirings ({key})": value for key, value in network.metrics.rewirings.items()}
        )
        return row
