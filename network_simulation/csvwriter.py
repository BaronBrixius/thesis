import csv
import os

class CSVWriter:
    def __init__(self, project_dir, file_name="metrics.csv"):
        os.makedirs(project_dir, exist_ok=True)
        metrics_file_path = os.path.join(project_dir, file_name)
        metrics_file = open(metrics_file_path, mode="w", newline="")
        self.csv_writer = csv.DictWriter(metrics_file, fieldnames=[])  # Fieldnames will be set on first write

    # Runtime Metrics Writing
    def write_metrics_line(self, row):
        if not self.csv_writer.fieldnames:  # Set fieldnames on first write
            self.csv_writer.fieldnames = row.keys()
            self.csv_writer.writeheader()

        self.csv_writer.writerow(row)
