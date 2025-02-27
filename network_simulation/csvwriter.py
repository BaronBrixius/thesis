import csv
import os

class CSVWriter:
    def __init__(self, project_dir, file_name="metrics.csv"):
        os.makedirs(project_dir, exist_ok=True)
        metrics_file_path = os.path.join(project_dir, file_name)
        self.metrics_file = open(metrics_file_path, mode="w", newline="")
        self.csv_writer = None  # Initialize lazily on first write

    # Runtime Metrics Writing
    def write_metrics_line(self, row):
        if self.csv_writer is None:  # Set fieldnames on first write
            self.csv_writer = csv.DictWriter(self.metrics_file, fieldnames=row.keys())
            if self.metrics_file.tell() == 0:  # Only write headers if file is empty
                self.csv_writer.writeheader()
        self.csv_writer.writerow(row)

    def close(self):
        self.metrics_file.flush()
        self.metrics_file.close()
