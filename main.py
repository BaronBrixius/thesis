import logging
from network_simulation.analyzer import PostRunAnalyzer
from network_simulation.experiment import Experiment
from network_simulation.visualization import ColorBy
from network_simulation.utils import get_times
import cProfile
import pstats
import os
# from gui.app import NetworkControlApp

base_dir = "/mnt/d/OneDrive - Vrije Universiteit Amsterdam/Y3-Thesis/code/output"

if __name__ == "__main__":
    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )

    profiler = cProfile.Profile()
    if profiler: profiler.enable()

    ## Run in GUI
    # NetworkControlApp()

    ## Quick Run
    Experiment().run_one_simulation(num_nodes=200, num_connections=2000, simulation_dir=os.path.join(base_dir, "maybegpu"), 
                                        num_steps=10_000, display_interval=1_000, metrics_interval=1000, random_seed=0, color_by=ColorBy.DEGREE)

    # Experiment Run
    # experiment_folder = os.path.join(base_dir, "gpu_parallel")
    # Experiment().run_experiment(
    #                         seed_range=range(1),
    #                         nodes_range=[200],
    #                         connections_range=range(2000, 10001, 1000),      # [x / 100.0 for x in range(2, 40, 2)],
    #                         num_steps=10_000,
    #                         display_interval=1000,
    #                         metrics_interval=1_000,
    #                         color_by=ColorBy.DEGREE,
    #                         experiment_dir=experiment_folder
    #                     )
    # PostRunAnalyzer(experiment_folder).aggregate_metrics(experiment_folder)

    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(10)

    print(get_times())   #TODO Remove for final version
    # print(Metrics.get_cluster_assignments.cache_info())
