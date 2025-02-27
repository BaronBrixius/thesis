import logging
from network_simulation.aggregator import aggregate_metrics
from network_simulation.analyzer import analyze_metrics
from network_simulation.experiment import Experiment
from network_simulation.visualization import ColorBy
from network_simulation.timer_util import get_times
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

    profiler = None #cProfile.Profile()
    if profiler: profiler.enable()

    ## Run in GUI
    # NetworkControlApp()

    ## Quick Run
    Experiment().run_one_simulation(num_nodes=200, num_edges=2000, simulation_dir=os.path.join(base_dir, "color"), 
                                        num_steps=40_000, display_interval=10_000, metrics_interval=10_000, random_seed=0, color_by=ColorBy.COMMUNITY)

    # Experiment Run
    # experiment_folder = os.path.join(base_dir, "little_buster/seed_2")
    # Experiment().run_experiment(
    #                         seed_range=[1],
    #                         nodes_range=[200],
    #                         edges_range=range(1000, 19901, 10),      # [x / 100.0 for x in range(2, 40, 2)],
    #                         num_steps=10_000_000,
    #                         display_interval=1_000_000,
    #                         metrics_interval=1_000,
    #                         color_by=ColorBy.DEGREE,
    #                         experiment_dir=experiment_folder
    #                     )
    # aggregate_metrics(experiment_folder)
    # analyze_metrics(experiment_folder)

    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)

    print(get_times())   #TODO Remove for final version
