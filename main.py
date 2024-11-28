from network_simulation.simulation_manager import SimulationManager
from network_simulation.utils import print_times
from network_simulation.visualization import ColorBy
import cProfile
import pstats
import os

# Network parameters
NUM_NODES = 100
NUM_CONNECTIONS = int(0.1 * (NUM_NODES * (NUM_NODES - 1) / 2)) # 10% density * total possible connections n*(n-1)/2

# Simulation parameters
NUM_STEPS = 10000
METRICS_INTERVAL = 1000
DISPLAY_INTERVAL = 1000
STABILIZATION_THRESHOLD = 0
OUTPUT_DIR = "foo"
COLOR_BY = ColorBy.CONNECTIONS

# Misc
RANDOM_SEED = 42

if __name__ == "__main__":
    profiler = None #cProfile.Profile()
    if profiler: profiler.enable()

    # Run the simulation
    print("Nodes:", NUM_NODES, "Connections:", NUM_CONNECTIONS, "Steps:", NUM_STEPS)
    sim = SimulationManager(num_nodes=NUM_NODES, num_connections=NUM_CONNECTIONS, output_dir=OUTPUT_DIR)
    sim.run(num_steps=NUM_STEPS, display_interval=DISPLAY_INTERVAL, metrics_interval=METRICS_INTERVAL, show=False, color_by=COLOR_BY)

    # 1: Networks with varying connection densities
    # for num_connections in range(50, 5050, 50):  # Adjust connection density
    #     scenario_dir = os.path.join("density_test_data", f"density_{num_connections}")
    #     sim = SimulationManager(num_nodes=200, num_connections=num_connections, output_dir=scenario_dir, random_seed=42)
    #     sim.run(num_steps=1_000_000, display_interval=1000, metrics_interval=1000, show=False)

    # 2: 600-unit networks with connection matrix and histogram
    # num_connections_600 = int(0.1 * (600 * (600 - 1) / 2))
    # for i in range(1,3):
    #     scenario_dir = os.path.join("600_nodes_test_data", f"seed_{i}")
    #     sim = SimulationManager(num_nodes=600, num_connections=num_connections_600, output_dir=scenario_dir, random_seed=i)
    #     sim.run(num_steps=10_000_000, display_interval=1000000, metrics_interval=1000000, show=False, color_by=ColorBy.CONNECTIONS)

    # 10_000_000 steps 1000 interval
    # scenario_dir = os.path.join("600_nodes_test_data", f"1000_interval")
    # sim = SimulationManager(num_nodes=600, num_connections=num_connections_600, output_dir=scenario_dir, random_seed=42)
    # sim.run(num_steps=10_000_000, display_interval=1000000, metrics_interval=1000, show=False, color_by=ColorBy.CONNECTIONS)

    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)

    print_times()
    