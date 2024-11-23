from network_simulation.simulation_manager import SimulationManager
from network_simulation.utils import print_times
import cProfile
import pstats
import os

# Network parameters
NUM_NODES = 200
NUM_CONNECTIONS = int(0.1 * (NUM_NODES * (NUM_NODES - 1) / 2)) # 10% density * total possible connections n*(n-1)/2

# Simulation parameters
NUM_STEPS = 100_000
METRICS_INTERVAL = 1000
DISPLAY_INTERVAL = 1000
STABILIZATION_THRESHOLD = 0
OUTPUT_DIR = None

# Misc
RANDOM_SEED = 42

if __name__ == "__main__":
    profiler = None #cProfile.Profile()
    if profiler: profiler.enable()

    # Run the simulation
    print("Nodes:", NUM_NODES, "Connections:", NUM_CONNECTIONS, "Steps:", NUM_STEPS)
    sim = SimulationManager(num_nodes=NUM_NODES, num_connections=NUM_CONNECTIONS, output_dir=OUTPUT_DIR)
    sim.run(num_steps=NUM_STEPS, show=True)

    # 1: Networks with varying connection densities
    # for num_connections in range(50, 5050, 50):  # Adjust connection density
    #     scenario_dir = os.path.join("density_test_data", f"density_{num_connections}")
    #     sim = SimulationManager(num_nodes=200, num_connections=num_connections, output_dir=scenario_dir, random_seed=42)
    #     sim.run(num_steps=1_000_000, display_interval=1000, metrics_interval=1000, show=False)

    # 2: 600-unit networks with connection matrix and histogram
    # num_connections_600 = int(0.1 * (NUM_NODES * (NUM_NODES - 1) / 2))
    # for i in range(5):
    #     scenario_dir = os.path.join("600_nodes_test_data", f"trial_{i}")
    #     sim = SimulationManager(num_nodes=600, num_connections=num_connections_600, output_dir=f"metrics_600_nodes_{i}", random_seed=i)
    #     sim.run(num_steps=10_000_000, display_interval=1000, metrics_interval=1000, show=False)

    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)

    print_times()
    