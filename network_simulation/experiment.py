from concurrent.futures import ProcessPoolExecutor
import itertools
from network_simulation.visualization import  ColorBy
import logging
import os
import pandas as pd
import shutil
import threading
import multiprocessing
import time

class Experiment:
    def __init__(self, experiment_folder):
        self.experiment_folder = experiment_folder
        manager = multiprocessing.Manager()
        self.termination_flag = manager.Event()
        self.logger = logging.getLogger(__name__)

    def monitor_input_early_termination(self):
        """Listen for 'quit' or 'exit' to terminate all runs."""
        while True:
            user_input = input("Enter 'quit' or 'exit' to stop the experiment: ").strip().lower()
            if user_input in {"quit", "exit"}:
                self.termination_flag.set()
                self.logger.info("Termination signal received. Slow-stopping experiment...")
                break

    def is_scenario_completed(self, scenario_dir, expected_num_rows):
        """Check if the scenario is completed based on existing metrics."""
        if not os.path.exists(scenario_dir):
            return False
        for file_name in os.listdir(scenario_dir):
            if file_name.startswith("summary_metrics") and file_name.endswith(".csv"):
                metrics_path = os.path.join(scenario_dir, file_name)
                if os.path.exists(metrics_path):
                    try:
                        metrics_df = pd.read_csv(metrics_path)
                        if len(metrics_df) >= expected_num_rows:    # if the metrics file is long enough
                            self.logger.info(f"Skipping completed scenario: {scenario_dir}")
                            return True
                        else:
                            shutil.rmtree(scenario_dir)             # remove incomplete scenario
                    except Exception as e:
                        self.logger.error(f"Error reading metrics file for {scenario_dir}: {e}")
        return False

    def run_simulation(self, num_nodes, num_connections, output_dir, num_steps, display_interval, metrics_interval, random_seed, color_by=ColorBy.ACTIVITY):
        if self.termination_flag.is_set():
            return f"Simulation {random_seed, num_nodes, num_connections} terminated by user."
        from network_simulation.simulation import Simulation  # Import inside to ensure clean process
        sim = Simulation(num_nodes=num_nodes, num_connections=num_connections, output_dir=os.path.join(self.experiment_folder, output_dir), color_by=color_by, random_seed=random_seed)
        sim.run(num_steps=num_steps, display_interval=display_interval, metrics_interval=metrics_interval)
        return f"Simulation completed for {num_nodes} nodes, {num_connections} connections with seed {random_seed}"

    def run_experiment(self, seed_range, nodes_range, connections_range, num_steps, display_interval, metrics_interval, connections_as_density=False):
        # Start thread to check for early termination
        threading.Thread(target=self.monitor_input_early_termination, daemon=True).start()

        start = time.time()

        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = []
            for num_nodes, num_connections, seed in itertools.product(nodes_range, connections_range, seed_range):
                if connections_as_density:   # decimal values for num_connections represent network density, and are converted
                    self.logger.debug(f"Converting density {num_connections} to connections for {num_nodes} nodes")
                    num_connections = int(num_connections * (num_nodes * (num_nodes - 1) / 2))
                scenario_dir = os.path.join(f"seed_{seed}", f"nodes_{num_nodes}", f"edges_{num_connections}")
                if metrics_interval and self.is_scenario_completed(scenario_dir, num_steps / metrics_interval):
                    self.logger.debug(f"Skipping completed scenario: {scenario_dir}")
                    continue

                # Add simulation to queue
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
                if self.termination_flag.is_set():
                    self.logger.info("Stopping pending simulations.")
                    break
                result = future.result()
                self.logger.info(result + f" at time {time.time() - start}")

        total_time = time.time() - start
        self.logger.info(f"End time: {total_time} s = {total_time / 3600} h")
