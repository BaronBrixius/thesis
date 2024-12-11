from network_simulation.output import Output
from network_simulation.utils import print_times
from network_simulation.visualization import ColorBy
import cProfile
import pstats
import os
from gui.app import NetworkControlApp
from concurrent.futures import ProcessPoolExecutor
import time

def run_simulation_in_process(num_nodes, num_connections, output_dir, num_steps, display_interval, metrics_interval, random_seed):
    from network_simulation.simulation import Simulation  # Import inside to ensure clean process
    sim = Simulation(num_nodes=num_nodes, num_connections=num_connections, output_dir=output_dir, random_seed=random_seed)
    sim.run(num_steps=num_steps, display_interval=display_interval, metrics_interval=metrics_interval, show=False)
    return f"Simulation completed for {num_connections} connections with seed {random_seed}"

if __name__ == "__main__":
    profiler = None #cProfile.Profile()
    if profiler: profiler.enable()
    start = time.time()

    ## Run in GUI
    # app = NetworkControlApp() 


    ## Quick Run
    # Simulation(num_nodes=200, num_connections=1990, output_dir="foo", random_seed=42).run(num_steps=50_000, display_interval=1_000, metrics_interval=1_000, show=False)


    ## Experiment Run
    import matplotlib
    matplotlib.use('Agg')

    experiment_folder = "density3"
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for seed in range(1):
            folder_name = os.path.join(experiment_folder, f"seed_{seed}")
            for num_connections in range(1000, 19901, 250):
                scenario_dir = os.path.join(folder_name, f"edges_{num_connections}")
                futures.append(
                    executor.submit(
                        run_simulation_in_process,
                        num_nodes=200,
                        num_connections=num_connections,
                        output_dir=scenario_dir,
                        num_steps=2_000_000,
                        display_interval=100_000,
                        metrics_interval=1_000,
                        random_seed=seed
                    )
                )

        # Monitor progress and handle exceptions
        for future in futures:
            try:
                result = future.result()
                print(result, "at time", {time.time() - start})
            except Exception as e:
                print(f"Simulation failed: {e}")

    Output.aggregate_metrics(os.path.join("output", experiment_folder), starting_step=4_000)

    # folder_name = f"density_data_2_seed_7"
    # for num_connections in range(4300, 4601, 300):  # [1200, 1700, 2400, 3000, 3500, 3950, 4150, 4450, 4600, 4850]:  # Adjust connection density
    #     scenario_dir = os.path.join(folder_name, f"edges_{num_connections}")
    #     sim = Simulation(num_nodes=200, num_connections=num_connections, output_dir=scenario_dir, random_seed=7)
    #     sim.run(num_steps=2_000_000, display_interval=100_000, metrics_interval=1_000, show=False)
    # Output.aggregate_metrics(os.path.join("output", folder_name), starting_step=1500_000)

    # Networks with varying connection densities
    # folder_name = f"density_data_2_long_small_runs"
    # for off in range(0, 100, 25):
    #     for num_connections in range(1600 + off, 1901, 100):  # [1200, 1700, 2400, 3000, 3500, 3950, 4150, 4450, 4600, 4850]:  # Adjust connection density
    #         if num_connections == 1600:
    #             continue
    #         scenario_dir = os.path.join(folder_name, f"edges_{num_connections}")
    #         sim = Simulation(num_nodes=200, num_connections=num_connections, output_dir=scenario_dir, random_seed=7)
    #         sim.run(num_steps=20_000_000, display_interval=100_000, metrics_interval=1_000, show=False)
    # Output.aggregate_metrics(os.path.join("output", folder_name))

    print(f"End time: {time.time() - start}")
    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(30)

    print_times()
    