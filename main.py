import logging
from experiment.aggregator import aggregate_metrics
from experiment.analyzer import analyze_metrics
from experiment.experiment import run_experiment, run_one_simulation
from file_generation.visualization import ColorBy
from timer_util import get_times
import cProfile
import pstats
import os
# from gui.app import NetworkControlApp

BASE_DIR = "/mnt/d/OneDrive - Vrije Universiteit Amsterdam/Y3-Thesis/code/output"

if __name__ == "__main__":
    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )

    profiler = None #cProfile.Profile()
    if profiler: profiler.enable()

    ## Run in GUI
    # NetworkControlApp()

    ## Quick Run
    # run_one_simulation(
    #     num_nodes=200,
    #     num_edges=2000,
    #     simulation_dir=os.path.join(BASE_DIR, "test"),
    #     num_steps=40_000,
    #     display_interval=10_000,
    #     metrics_interval=10_000,
    #     random_seed=42,
    #     color_by=ColorBy.COMMUNITY
    # )

    # Experiment Run
    experiment_folder = os.path.join(BASE_DIR, "test")
    run_experiment(
        seed_range=range(1),
        nodes_range=[200],
        edges_range=range(1000, 10001, 1000),
        num_steps=5_000,
        display_interval=1_000,
        metrics_interval=1_000,
        color_by=ColorBy.COMMUNITY,
        experiment_dir=experiment_folder
    )
    # aggregate_metrics([experiment_folder, os.path.join(base_dir, "hybrid_rainbow_zero")])
    # analyze_metrics(experiment_folder)

    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)

    print(get_times())   #TODO Remove for final version
