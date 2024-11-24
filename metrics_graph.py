import os
import pandas as pd
import matplotlib.pyplot as plt

def create_metrics_graphs(base_dir):
    # Iterate through subdirectories in the base directory
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):  # Ensure it's a directory
            metrics_file = os.path.join(folder_path, "metrics.csv")
            if os.path.exists(metrics_file):
                # Read metrics.csv into a DataFrame
                data = pd.read_csv(metrics_file)

                # Check if required columns exist
                if {'Step_Num', 'CPL', 'CC'}.issubset(data.columns):
                    # Plot CC and CPL on the same graph
                    plt.figure(figsize=(8, 6))
                    plt.plot(data['Step_Num'], data['CPL'], label='CPL (Characteristic Path Length)', color='blue')
                    plt.plot(data['Step_Num'], data['CC'], label='CC (Clustering Coefficient)', color='green')

                    # Add titles and labels
                    plt.title(f"Metrics for {folder}")
                    plt.xlabel("Step Number")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)

                    # Save the graph to the folder
                    output_file = os.path.join(folder_path, f"{folder}_metrics_graph.png")
                    plt.savefig(output_file, dpi=300)
                    plt.close()
                    print(f"Saved graph: {output_file}")
                else:
                    print(f"Missing required columns in {metrics_file}")
            else:
                print(f"No metrics.csv found in {folder_path}")

# Example usage:
base_directory = "density_test_data"  # Replace with the actual base directory containing subfolders
create_metrics_graphs(base_directory)
