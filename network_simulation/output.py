import csv
import os
from network_simulation.network import NodeNetwork

class Output:
    def __init__(self, project_dir, num_nodes=None, num_connections=None):
        os.makedirs(project_dir, exist_ok=True)
        metrics_file_path = os.path.join(project_dir, f"summary_metrics_nodes_{num_nodes}_edges_{num_connections}.csv")
        self.metrics_file = open(metrics_file_path, mode="w", newline="")

        self.csv_writer = None

    # Runtime Metrics Writing
    def write_metrics_line(self, step, network: NodeNetwork):
        row = network.metrics.compute_metrics(step, network)

        if self.csv_writer is None: # Lazy loading the writer also allows us to wait and see what headers will be needed
            self.csv_writer = csv.DictWriter(self.metrics_file, fieldnames=row.keys())
            self.csv_writer.writeheader()

        self.csv_writer.writerow(row)
