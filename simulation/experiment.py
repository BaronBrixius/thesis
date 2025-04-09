import multiprocessing
import concurrent.futures
from itertools import product
from runtime_output.visualization import ColorBy
import logging
import os
import threading
import random

def _monitor_input_early_termination(executor):
    """Listen for 'quit' or 'exit' to terminate future runs."""
    while True:
        user_input = input("Enter 'quit' or 'exit' to stop the experiment:\n").strip().lower()
        if user_input in {"quit", "exit"}:
            logging.info("Termination signal received. Finishing started simulations before exiting...")
            executor.shutdown(cancel_futures=True, wait=True)   # finishes all current work and max_workers extra tasks before terminating, not sure why, seems fine enough
            break

def run_one_simulation(num_nodes, num_edges, simulation_dir, num_steps, display_interval, metrics_interval, random_seed, color_by, process_num=0):
    from simulation.simulation import Simulation  # Import inside to ensure clean process
    print(f"Starting simulation {random_seed, num_nodes, num_edges}") # I don't want to reimport the logger just for this, remove this for final?
    if random_seed == 0:
        color_by = ColorBy.ACTIVITY
    elif random_seed < 3:
        color_by = ColorBy.DEGREE
    sim = Simulation(num_nodes=num_nodes, num_edges=num_edges, simulation_dir=simulation_dir, color_by=color_by, display_interval=display_interval, metrics_interval=metrics_interval, random_seed=random_seed, process_num=process_num)
    if random_seed == 0:
        color_by = ColorBy.ACTIVITY
    elif random_seed < 3:
        color_by = ColorBy.COMMUNITY
    else:
        color_by = ColorBy.DEGREE
    try:
        sim.run(num_steps=num_steps)
    except Exception as e:
        return f"Simulation {random_seed, num_nodes, num_edges} failed with error: \n{e}"
    return f"Completed simulation {random_seed, num_nodes, num_edges}"

def run_experiment(seed_range, nodes_range, edges_range, num_steps, display_interval, metrics_interval, color_by, experiment_dir="/mnt/d/OneDrive - Vrije Universiteit Amsterdam/Y3-Thesis/code/output"):
    with concurrent.futures.ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'), max_workers=4) as executor:
        futures = []

        # Random order, to avoid running a lot of large simulations at the same time
        param_combinations = list(product(nodes_range, edges_range, seed_range))
        random.shuffle(param_combinations)
        for num_nodes, num_edges, seed in param_combinations:
            # Decimal values are treated as density percentages
            if isinstance(num_edges, float):
                logging.debug(f"Converting density {num_edges} to edges for {num_nodes} nodes")
                num_edges = int(num_edges * (num_nodes * (num_nodes - 1) / 2))

            # Skip if scenario is already completed
            simulation_dir = os.path.join(experiment_dir, f"seed_{seed}", f"nodes_{num_nodes}", f"edges_{num_edges}")
            if os.path.exists(simulation_dir):
                logging.debug(f"Skipping completed scenario: {seed, num_nodes, num_edges}")
                continue

            # Submit the simulation task to the executor
            futures.append(executor.submit(run_one_simulation, num_nodes, num_edges, simulation_dir, num_steps, display_interval, metrics_interval, seed, color_by, len(futures)))

        # Start a thread to monitor early termination, after task submission so submission doesn't break
        threading.Thread(target=_monitor_input_early_termination, args=(executor,), daemon=True).start()

        # Wait for simulations to complete
        for future in futures:
            if future.cancelled():
                continue
            logging.info(future.result())
