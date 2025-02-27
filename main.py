import logging
from network_simulation.aggregator import aggregate_metrics
from network_simulation.analyzer import analyze_metrics
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
    
    profiler = None #cProfile.Profile()
    if profiler: profiler.enable()

    ## Run in GUI
    # NetworkControlApp()

    ## Quick Run
    # Experiment("output").run_simulation(num_nodes=200, num_connections=2_000, output_dir="foo",  num_steps=5_000, display_interval=1_000, metrics_interval=1_000, random_seed=42)

    ## Experiment Run
    experiment_folder = os.path.join(base_dir, "foo")
    experiment = Experiment(experiment_folder)
    experiment.run_experiment(
                            seed_range=range(2),
                            nodes_range=[200],
                            connections_range=range(2000, 3000, 50),      # [x / 100.0 for x in range(2, 40, 2)],
                            connections_as_density = False,
                            num_steps=2_0_000,
                            display_interval=5_000,
                            metrics_interval=1_000,
                        )
    PostRunAnalyzer(experiment_folder).aggregate_metrics(experiment_folder, starting_step=1_0_000)

    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)

    print(get_times())   #TODO Remove for final version
