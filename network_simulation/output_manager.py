import matplotlib.pyplot as plt
import numpy as np
import os

class OutputManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        if not self.base_dir:
            return
        self.folders = {
            "histograms": os.path.join(base_dir, "histograms"),
            "images": os.path.join(base_dir, "images"),
            "matrices": os.path.join(base_dir, "matrices"),
            "activities": os.path.join(base_dir, "activities"),
            "metrics_graphs": os.path.join(base_dir, "metrics_graphs"),
        }
        self.prepare_directories()

    def prepare_directories(self):
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)

    def save_metrics(self, step, characteristic_path_length, clustering_coefficient, breakup_count, time_since_start):
        if not self.base_dir:
            return
        metrics_file = os.path.join(self.base_dir, "metrics.csv")
        with open(metrics_file, "a") as f:
            if step == 0:  # Write headers if it's the first step
                f.write("Step_Num,CPL,CC,Breakups,Time\n")
            f.write(f"{step},{characteristic_path_length},{clustering_coefficient},{breakup_count},{time_since_start:.2f}\n")

    def save_histogram(self, connections, step):
        if not self.base_dir:
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(connections, bins=np.arange(connections.min(), connections.max() + 2), 
                color='black', edgecolor='white')
        ax.set_title(f"Connection Distribution (Step {step})")
        ax.set_xlabel("#connections per unit")
        ax.set_ylabel("#units")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot as an image
        plot_path = os.path.join(self.folders["histograms"], f"histogram_{step}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved histogram plot at step {step}: {plot_path}")

    def save_matrix(self, matrix, step):
        if not self.base_dir:
            return
        file_path = os.path.join(self.folders["matrices"], f"matrix_{step}.csv")
        np.savetxt(file_path, matrix, fmt="%i", delimiter=",")
        print(f"Saved matrix at step {step}: {file_path}")

    def save_network_image(self, plot, step):
        if not self.base_dir:
            return
        file_path = os.path.join(self.folders["images"], f"image_{step}.png")
        plot.fig.savefig(file_path, dpi=300)
        print(f"Saved network image at step {step}: {file_path}")

    def save_activities(self, activities, step):
        if not self.base_dir:
            return
        file_path = os.path.join(self.folders["activities"], f"activities_{step}.csv")
        np.savetxt(file_path, activities, fmt="%.6f", delimiter=",")
        print(f"Saved activities at step {step}: {file_path}")

    # Reads the metrics.csv file and outputs as a graph
    def convert_metrics_to_graph(self):
        if not self.base_dir:
            return

        metrics_file = os.path.join(self.base_dir, "metrics.csv")
        if os.path.exists(metrics_file):
            # Read metrics.csv into a DataFrame
            data = pd.read_csv(metrics_file)

            # Check if required columns exist
            if {'Step_Num', 'CPL', 'CC'}.issubset(data.columns):
                # Plot CC and CPL on the same graph
                plt.figure(figsize=(10, 6))
                plt.plot(data['Step_Num'], data['CPL'], label='CPL (Characteristic Path Length)', color='blue')
                plt.plot(data['Step_Num'], data['CC'], label='CC (Clustering Coefficient)', color='green')

                # Add titles and labels
                plt.title(f"Metrics Graph")
                plt.xlabel("Step Number")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)

                # Save the graph to the metrics_graphs folder
                output_file = os.path.join(self.folders["metrics_graphs"], "metrics_graph.png")
                plt.savefig(output_file, dpi=300)
                plt.close()
                print(f"Saved metrics graph: {output_file}")
            else:
                print(f"Missing required columns in {metrics_file}")
        else:
            print(f"No metrics.csv found at {metrics_file}")