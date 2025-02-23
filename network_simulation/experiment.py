from multiprocessing import Process, Manager
from itertools import product
from network_simulation.visualization import ColorBy
import logging
import os
import time
import threading

class Experiment:
    def __init__(self):
        self.termination_flag = Manager().Event()
        self.logger = logging.getLogger(__name__)

    def monitor_input_early_termination(self):
        """Listen for 'quit' or 'exit' to terminate all runs."""
        while True:
            user_input = input("Enter 'quit' or 'exit' to stop the experiment:\n").strip().lower()
            if user_input in {"quit", "exit"}:
                self.termination_flag.set()
                self.logger.info("Termination signal received. Finishing running simulations before exiting...")
                break

    def run_one_simulation(self, num_nodes, num_connections, simulation_dir, num_steps, display_interval, metrics_interval, random_seed, color_by=ColorBy.ACTIVITY, process_num=0):
        if self.termination_flag.is_set():
            self.logger.info(f"Simulation {random_seed, num_nodes, num_connections} terminated by user")
            return

        from network_simulation.simulation import Simulation  # Import inside to ensure clean process
        self.logger.info(f"Starting simulation {random_seed, num_nodes, num_connections}")
        sim = Simulation(num_nodes=num_nodes, num_connections=num_connections, simulation_dir=simulation_dir, color_by=color_by, random_seed=random_seed)
        sim.run(num_steps=num_steps, display_interval=display_interval, metrics_interval=metrics_interval)
        self.logger.info(f"Completed simulation {random_seed, num_nodes, num_connections}")

    def run_experiment(self, seed_range, nodes_range, connections_range, num_steps, display_interval, metrics_interval, color_by=ColorBy.ACTIVITY, experiment_dir="/mnt/d/OneDrive - Vrije Universiteit Amsterdam/Y3-Thesis/code/output"):
        # Start thread that checks for early termination
        threading.Thread(target=self.monitor_input_early_termination, daemon=True).start()

        max_processes = 11  # Maximum allowed concurrent processes
        processes = []
        manager = Manager()
        queue = manager.Queue()

        for num_nodes, num_connections, seed in product(nodes_range, connections_range, seed_range):
            # Decimal values are treated as density percentages
            if isinstance(num_connections, float):  
                self.logger.debug(f"Converting density {num_connections} to connections for {num_nodes} nodes")
                num_connections = int(num_connections * (num_nodes * (num_nodes - 1) / 2))
 
            # Skip if scenario is already completed
            simulation_dir = os.path.join(experiment_dir, f"seed_{seed}", f"nodes_{num_nodes}", f"edges_{num_connections}")
            if os.path.exists(simulation_dir):
                self.logger.info(f"Skipping completed scenario: {seed, num_nodes, num_connections}")
                continue

            # Wait for available slots if max processes are running
            while len(processes) >= max_processes:
                for process in processes:
                    if not process.is_alive():  # Remove completed processes
                        processes.remove(process)
                time.sleep(5)  # Short delay to prevent busy waiting

            # Start a new process
            process = Process(
                target=self.run_one_simulation,
                args=(num_nodes, num_connections, simulation_dir, num_steps, display_interval, metrics_interval, seed, color_by, queue)
            )
            process.start()
            processes.append(process)

        # Wait for all remaining processes to finish
        for process in processes:
            process.join()

        # Collect results
        while not queue.empty():
            result = queue.get()
            if "Error" in result:
                self.logger.error(result)
            else:
                self.logger.info(result)