from concurrent.futures import ProcessPoolExecutor
from itertools import product
from network_simulation.visualization import ColorBy
import logging
import os
from pandas import read_csv
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

    def is_scenario_completed(self, simulation_dir, expected_num_rows):
        """Check if the scenario is completed based on existing metrics."""
        if not os.path.exists(simulation_dir):
            return False
        for file_name in os.listdir(simulation_dir):
            if file_name.startswith("summary_metrics") and file_name.endswith(".csv"):
                metrics_path = os.path.join(simulation_dir, file_name)
                if os.path.exists(metrics_path):
                    try:
                        metrics_df = read_csv(metrics_path)
                        return len(metrics_df) >= expected_num_rows    # if the metrics file is long enough
                    except Exception:
                        pass
                    shutil.rmtree(simulation_dir)             # remove incomplete or broken simulation
                    return False
        return False

    def run_one_simulation(self, num_nodes, num_connections, simulation_dir, num_steps, display_interval, metrics_interval, random_seed, color_by=ColorBy.ACTIVITY, skip=False):
        if self.termination_flag.is_set():
            return f"Simulation {random_seed, num_nodes, num_connections} terminated by user"

        # Skip if scenario is already completed
        full_dir = os.path.join(self.experiment_folder, simulation_dir)
        if skip and metrics_interval and self.is_scenario_completed(full_dir, num_steps / metrics_interval):
            return f"Skipping completed scenario: {random_seed, num_nodes, num_connections}"

        from network_simulation.simulation import Simulation  # Import inside to ensure clean process
        sim = Simulation(num_nodes=num_nodes, num_connections=num_connections, output_dir=full_dir, color_by=color_by, random_seed=random_seed)
        self.logger.info(f"Starting with {random_seed, num_nodes, num_connections}")
        sim.run(num_steps=num_steps, display_interval=display_interval, metrics_interval=metrics_interval)
        return f"Simulation completed for {random_seed, num_nodes, num_connections}"

    def run_experiment(self, seed_range, nodes_range, connections_range, num_steps, display_interval, metrics_interval, max_workers=os.cpu_count()):
        # Start thread that checks for early termination
        threading.Thread(target=self.monitor_input_early_termination, daemon=True).start()

        start = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for num_nodes, num_connections, seed in product(nodes_range, connections_range, seed_range):
                if isinstance(num_connections, float):  # decimal values treated as network density percentages
                    self.logger.debug(f"Converting density {num_connections} to connections for {num_nodes} nodes")
                    num_connections = int(num_connections * (num_nodes * (num_nodes - 1) / 2))

                # Add simulation to queue
                futures.append(
                    executor.submit(
                        self.run_one_simulation,
                        num_nodes=num_nodes,
                        num_connections=num_connections,
                        simulation_dir=os.path.join(f"seed_{seed}", f"nodes_{num_nodes}", f"edges_{num_connections}"),
                        num_steps=num_steps,
                        display_interval=display_interval,
                        metrics_interval=metrics_interval,
                        random_seed=seed,
                        skip=True,
                    )
                )

            # Monitor progress and handle exceptions
            for future in futures:
                try:
                    result = future.result()
                    self.logger.info(result + f" at time {time.time() - start}")
                except Exception as e:
                    self.logger.error(f"Error in simulation: {repr(e)}")

        total_time = time.time() - start
        self.logger.info(f"End time: {total_time} s = {total_time / 3600} h")
