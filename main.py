import logging
from network_simulation.analyzer import PostRunAnalyzer
from network_simulation.experiment import Experiment
from network_simulation.metrics import Metrics
from network_simulation.visualization import ColorBy
from network_simulation.utils import get_times
import cProfile
import pstats
import os
from gui.app import NetworkControlApp

base_dir = "output"

if __name__ == "__main__":
    logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )

    profiler = cProfile.Profile()
    if profiler: profiler.enable()

    ## Run in GUI
    # NetworkControlApp()

    ## Quick Run
    # Experiment(base_dir).run_simulation(num_nodes=200, num_connections=2000, output_dir="graphtools", 
    #                                     num_steps=20000, display_interval=5000, metrics_interval=1000, random_seed=42, color_by=ColorBy.ACTIVITY)

    # Experiment Run
    experiment_folder = os.path.join(base_dir, "new_graphtools_stuff")
    experiment = Experiment(experiment_folder)
    experiment.run_experiment(
                            seed_range=range(5),
                            nodes_range=[200],
                            connections_range=range(250, 19_901, 250),      # [x / 100.0 for x in range(2, 40, 2)],
                            connections_as_density = False,
                            num_steps=3_000_000,
                            display_interval=100_000,
                            metrics_interval=1_000,
                        )
    # PostRunAnalyzer(experiment_folder).aggregate_metrics(experiment_folder, starting_step=2_000_000)

    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(40)

    print(get_times())   #TODO Remove for final version
    # print(Metrics.get_cluster_assignments.cache_info())
