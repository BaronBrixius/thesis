from network_simulation.analyzer import PostRunAnalyzer
from network_simulation.experiment import Experiment
from network_simulation.utils import print_times
from network_simulation.visualization import ColorBy
import cProfile
import pstats
import os
from gui.app import NetworkControlApp

if __name__ == "__main__":
    profiler = cProfile.Profile()
    if profiler: profiler.enable()

    ## Run in GUI
    # NetworkControlApp()

    ## Quick Run
    # Experiment("output").run_simulation(num_nodes=200, num_connections=2_000, output_dir="foo",  num_steps=5_000, display_interval=1_000, metrics_interval=1_000, random_seed=42)

    ## Experiment Run
    experiment_folder = "cluster_tracked"
    experiment = Experiment(experiment_folder)
    experiment.run_experiment(
                            seed_range=range(3),
                            nodes_range=[200],
                            connections_range=[1500, 1800, 2100, 2300, 2500, 2800, 3000, 3500, 4500, 5000, 6000, 7000, 8500, 10000],
                            num_steps=5_000_000,
                            display_interval=250_000,
                            metrics_interval=1_000
                        )
    PostRunAnalyzer(experiment_folder).aggregate_metrics(os.path.join("output", experiment_folder), starting_step=4_000_000)

    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(40)

    print_times()   #TODO Remove for final version
