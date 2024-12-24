from concurrent.futures import ProcessPoolExecutor
from itertools import product
import os
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

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
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