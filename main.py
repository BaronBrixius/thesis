from network_simulation.output import Output
from network_simulation.simulation import Simulation
from network_simulation.utils import print_times
from network_simulation.visualization import ColorBy
import cProfile
import pstats
import os
from gui.app import NetworkControlApp
from concurrent.futures import ThreadPoolExecutor 

def run_simulation_in_thread(simulation, num_steps, display_interval=1000, metrics_interval=1000):
    simulation.run(
        num_steps=num_steps,
        display_interval=display_interval,
        metrics_interval=metrics_interval,
        show=False
    )

if __name__ == "__main__":
    profiler = cProfile.Profile()
    if profiler: profiler.enable()

    ## Run in GUI
    # app = NetworkControlApp() 


    ## Quick Run
    # Simulation(num_nodes=200, num_connections=1990, output_dir="foo", random_seed=42).run(num_steps=50_000, display_interval=1_000, metrics_interval=1_000, show=False)


    ## Experiment Run
    import matplotlib
    matplotlib.use('Agg')

    num_nodes = 200
    seed = 8
    folder_name = f"littleone_seed_{seed}"
    os.makedirs(os.path.join("output", folder_name), exist_ok=True)

    # Create Simulation instances
    simulations = [
        Simulation(
            num_nodes=num_nodes,
            num_connections=num_connections,
            output_dir=os.path.join(folder_name, f"edges_{num_connections}"),
            random_seed=seed
        )
        for num_connections in range(2000, 2500, 100)
    ]

    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
        futures = [executor.submit(run_simulation_in_thread, sim, num_steps=50_000, display_interval=100000, metrics_interval=1000) for sim in simulations]

        # Monitor progress
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Simulation failed: {e}")
    Output.aggregate_metrics(os.path.join("output", folder_name), starting_step=4_000)

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

    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(40)

    print_times()
    