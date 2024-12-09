from network_simulation.output import Output
from network_simulation.simulation import Simulation
from network_simulation.utils import print_times
from network_simulation.visualization import ColorBy
import cProfile
import pstats
import os
import tkinter as tk
from gui.app import NetworkControlApp

# Network parameters
NUM_NODES = 200
NUM_CONNECTIONS = int(0.1 * (NUM_NODES * (NUM_NODES - 1) / 2)) # 10% density * total possible connections n*(n-1)/2

# Simulation parameters
NUM_STEPS = 2_000_000
METRICS_INTERVAL = 10_000
DISPLAY_INTERVAL = 100_000
STABILIZATION_THRESHOLD = 0
OUTPUT_DIR = "foo"

# Misc
RANDOM_SEED = 42

if __name__ == "__main__":
    profiler = None #cProfile.Profile()
    if profiler: profiler.enable()

    root = tk.Tk()
    app = NetworkControlApp(root)
    root.mainloop()

    # Run the simulation
    # sim = Simulation(num_nodes=NUM_NODES, num_connections=NUM_CONNECTIONS, output_dir=OUTPUT_DIR, random_seed=RANDOM_SEED)
    # sim.run(num_steps=NUM_STEPS, display_interval=DISPLAY_INTERVAL, metrics_interval=METRICS_INTERVAL, show=False)

    # folder_name = f"littleone_seed_7"
    # for num_connections in range(1660, 1675):
    #     scenario_dir = os.path.join(folder_name, f"edges_{num_connections}")
    #     sim = Simulation(num_nodes=200, num_connections=num_connections, output_dir=scenario_dir, random_seed=7)
    #     # sim.run(num_steps=1_000_000, display_interval=200_000, metrics_interval=10_000, show=False)
    #     sim.output.post_run_output()
    # Output.aggregate_metrics(os.path.join("output", folder_name), starting_step=500_000)

    # folder_name = f"density_data_2_seed_7"
    # for num_connections in range(4300, 4601, 300):  # [1200, 1700, 2400, 3000, 3500, 3950, 4150, 4450, 4600, 4850]:  # Adjust connection density
    #     scenario_dir = os.path.join(folder_name, f"edges_{num_connections}")
    #     sim = Simulation(num_nodes=200, num_connections=num_connections, output_dir=scenario_dir, random_seed=7)
    #     sim.run(num_steps=2_000_000, display_interval=100_000, metrics_interval=1_000, show=False)
    # Output.aggregate_metrics(os.path.join("output", folder_name), starting_step=1500_000)

    # 1: Networks with varying connection densities
    # folder_name = f"density_data_2_long_small_runs"
    # for off in range(0, 100, 25):
    #     for num_connections in range(1600 + off, 1901, 100):  # [1200, 1700, 2400, 3000, 3500, 3950, 4150, 4450, 4600, 4850]:  # Adjust connection density
    #         if num_connections == 1600:
    #             continue
    #         scenario_dir = os.path.join(folder_name, f"edges_{num_connections}")
    #         sim = Simulation(num_nodes=200, num_connections=num_connections, output_dir=scenario_dir, random_seed=7)
    #         sim.run(num_steps=20_000_000, display_interval=100_000, metrics_interval=1_000, show=False)
    # Output.aggregate_metrics(os.path.join("output", folder_name))

    # 2: 600-unit networks with connection matrix and histogram
    # num_connections_600 = int(0.1 * (600 * (600 - 1) / 2))
    # for i in range(3):
    #     scenario_dir = os.path.join("600_nodes_test_data", f"seed_{i}")
    #     sim = Simulation(num_nodes=600, num_connections=num_connections_600, output_dir=scenario_dir, random_seed=i)
    #     sim.run(num_steps=10_000_000, display_interval=1000000, metrics_interval=1000000, show=False, color_by=ColorBy.CONNECTIONS)

    # 10_000_000 steps 1000 interval
    # scenario_dir = os.path.join("600_nodes_test_data", f"1000_interval")
    # sim = SimulationManager(num_nodes=600, num_connections=num_connections_600, output_dir=scenario_dir, random_seed=42)
    # sim.run(num_steps=10_000_000, display_interval=1000000, metrics_interval=1000, show=False, color_by=ColorBy.CONNECTIONS)

    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(40)

    print_times()
    