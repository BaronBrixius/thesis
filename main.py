from itertools import product
from network_simulation.output import Output
from network_simulation.analyzer import PostRunAnalyzer
from network_simulation.experiment import Experiment
from network_simulation.utils import print_times
from network_simulation.visualization import ColorBy
import cProfile
import pstats
import os
from gui.app import NetworkControlApp

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
