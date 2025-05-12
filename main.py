import logging
from postrun_output.aggregator import aggregate_metrics
from postrun_output.data_processor import process_metrics
from postrun_output.plotter import generate_scatterplots, scatterplot_clustering_vs_edges, long_long_plots, real_big_plots
from simulation.experiment import run_experiment, run_one_simulation
from runtime_output.visualization import ColorBy
import cProfile
import pstats
import os

BASE_DIR = "/mnt/d/OneDrive - Vrije Universiteit Amsterdam/Y3-Thesis/output"

if __name__ == "__main__":
    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )

    profiler = None #cProfile.Profile()
    if profiler: profiler.enable()

    # Experiment Run
    experiment_folder = os.path.join(BASE_DIR, "test")
    run_experiment(
        seed_range=range(5),
        nodes_range=[200],
        edges_range=range(10, 19901, 10),
        num_steps=10_000_000,
        display_interval=1_000_000,
        metrics_interval=1_000,
        color_by=ColorBy.COMMUNITY,
        experiment_dir=experiment_folder,
    )
    # aggregate_metrics([experiment_folder, os.path.join(BASE_DIR, "hybrid_rainbow_zero")])
    # process_metrics(experiment_folder)
    # generate_scatterplots(experiment_folder)
    # scatterplot_clustering_vs_edges(os.path.join(experiment_folder, "aggregated_metrics.csv"))
    # long_long_plots(os.path.join(BASE_DIR, "the_long_long_runs"))
    # real_big_plots(os.path.join(BASE_DIR, "realbig"))

    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)
