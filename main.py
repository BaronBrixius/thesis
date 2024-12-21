from network_simulation.output import Output
from network_simulation.analyzer import PostRunAnalyzer
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
    print(f"Simulation starting for {num_connections} connections with seed {random_seed}")
    sim = Simulation(num_nodes=num_nodes, num_connections=num_connections, output_dir=output_dir, random_seed=random_seed)
    sim.run(num_steps=num_steps, display_interval=display_interval, metrics_interval=metrics_interval, show=False)
    return f"Simulation completed for {num_connections} connections with seed {random_seed}"

def run_experiment(experiment_folder, seed_range, connections_range, num_steps, display_interval, metrics_interval, num_nodes):
    start = time.time()

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for num_connections in connections_range:
            for seed in seed_range:
                scenario_dir = os.path.join(experiment_folder, f"seed_{seed}", f"edges_{num_connections}")
                if os.path.exists(scenario_dir):
                    continue
                futures.append(
                    executor.submit(
                        run_simulation_in_process,
                        num_nodes=num_nodes,
                        num_connections=num_connections,
                        output_dir=scenario_dir,
                        num_steps=num_steps,
                        display_interval=display_interval,
                        metrics_interval=metrics_interval,
                        random_seed=seed
                    )
                )

        # Monitor progress and handle exceptions
        for future in futures:
            try:
                result = future.result()
                print(result, "at time", time.time() - start)
            except Exception as e:
                print(f"Simulation failed: {e}")

    total_time = time.time() - start
    print(f"End time: {total_time} s = {total_time / 3600} h")

if __name__ == "__main__":
    profiler = None
    # profile = cProfile.Profile()
    if profiler: profiler.enable()

    ## Run in GUI
    app = NetworkControlApp()

    ## Quick Run
    # run_simulation_in_process(num_nodes=200, num_connections=2_000, output_dir="foo",  num_steps=500_000, display_interval=100_000, metrics_interval=1_000, random_seed=42)

    ## Experiment Run
    # experiment_folder = "D:\OneDrive - Vrije Universiteit Amsterdam\Y3-Thesis\code\output\\foo"
    # run_experiment(experiment_folder,
    #                seed_range=range(2),
    #                connections_range=range(1250, 1571, 250),
    #                num_steps=200_000,
    #                display_interval=100_000,
    #                metrics_interval=10_000,
    #                num_nodes=200)
    # PostRunAnalyzer(experiment_folder).aggregate_metrics(os.path.join("output", experiment_folder), starting_step=150_000)

    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(40)

    print_times()
    