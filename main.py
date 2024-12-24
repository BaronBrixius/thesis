from itertools import product
from network_simulation.output import Output
from network_simulation.analyzer import PostRunAnalyzer
from network_simulation.utils import print_times
from network_simulation.visualization import ColorBy
import cProfile
import pstats
import os
from gui.app import NetworkControlApp
from concurrent.futures import ProcessPoolExecutor
import threading
import time

class Experiment:
    def __init__(self, experiment_folder):
        self.experiment_folder = experiment_folder
        self.termination_flag = False
        threading.Thread(target=self.monitor_input, daemon=True).start()

    def monitor_input(self):
        """Listen for 'quit' or 'exit' to terminate all runs."""
        while True:
            user_input = input("Enter 'quit' or 'exit' to stop the experiment: ").strip().lower()
            if user_input in {"quit", "exit"}:
                self.termination_flag = True
                print("Termination signal received. Stopping experiment...")
                break

    def run_simulation(self, num_nodes, num_connections, output_dir, num_steps, display_interval, metrics_interval, random_seed):
        if self.termination_flag:
             return f"Simulation {random_seed, num_nodes, num_connections} terminated by user."
        from network_simulation.simulation import Simulation  # Import inside to ensure clean process
        sim = Simulation(num_nodes=num_nodes, num_connections=num_connections, output_dir=output_dir, random_seed=random_seed)
        print(f"Simulation starting for {num_connections} connections with seed {random_seed}")
        sim.run(num_steps=num_steps, display_interval=display_interval, metrics_interval=metrics_interval, show=False)
        return f"Simulation completed for {num_connections} connections with seed {random_seed}"

    def run_experiment(self, experiment_folder, seed_range, nodes_range, connections_range, num_steps, display_interval, metrics_interval):
        start = time.time()

        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = []
            for num_nodes, num_connections, seed in product(nodes_range, connections_range, seed_range):
                scenario_dir = os.path.join(experiment_folder, f"seed_{seed}", f"nodes_{num_nodes}", f"edges_{num_connections}")
                if os.path.exists(scenario_dir):
                    continue

                futures.append(
                    executor.submit(
                        self.run_simulation,
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
                if self.termination_flag:
                    print("Stopping pending simulations.")
                    break
                result = future.result()
                print(result, "at time", time.time() - start)

        total_time = time.time() - start
        print(f"End time: {total_time} s = {total_time / 3600} h")

if __name__ == "__main__":
    profiler = None
    # profile = cProfile.Profile()
    if profiler: profiler.enable()

    ## Run in GUI
    # app = NetworkControlApp()

    ## Quick Run
    # Experiment("output").run_simulation(num_nodes=200, num_connections=2_000, output_dir="foo",  num_steps=500_000, display_interval=100_000, metrics_interval=1_000, random_seed=42)

    ## Experiment Run
    experiment_folder = "D:/OneDrive - Vrije Universiteit Amsterdam/Y3-Thesis/code/output/foo"
    experiment = Experiment(experiment_folder)
    experiment.run_experiment(experiment_folder,
                              seed_range=range(2),
                              nodes_range=[300],
                              connections_range=range(500, 44851, 500),
                              num_steps=5_000_000,
                              display_interval=250_000,
                              metrics_interval=1_000)
    PostRunAnalyzer(experiment_folder).aggregate_metrics(os.path.join("output", experiment_folder), starting_step=4_000_000)

    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(40)

    print_times()   #TODO Remove for final version
