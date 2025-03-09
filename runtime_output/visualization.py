import os
from enum import Enum
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    from graph_tool.draw import arf_layout
matplotlib.use('Agg')
from runtime_output.physics import Physics

class ColorBy(Enum):
    ACTIVITY = "cividis"
    COMMUNITY = "tab20"
    DEGREE = "inferno"

class Visualization:
    def __init__(self, adjacency_matrix, activities, graph, community_assignments, output_dir="foo", color_by=ColorBy.ACTIVITY):
        self.output_dir = os.path.join(output_dir, "images")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.color_by = color_by
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.positions = None
        self.physics = Physics(.015)
        self.positions = np.random.rand(len(adjacency_matrix), 2) * 100
        self._compute_layout(adjacency_matrix, max_iter=100)
        # self._compute_layout(adjacency_matrix)
        self._initialize_plot(adjacency_matrix, activities, self.positions, community_assignments)

    def _compute_layout(self, adjacency_matrix, max_iter=25):
        self.positions = self.physics.apply_forces(adjacency_matrix, self.positions, max_iterations=max_iter)
        return self.positions

    def _compute_vertex_colors(self, adjacency_matrix, activities, community_assignments):
        if self.color_by == ColorBy.ACTIVITY:
            return activities
        elif self.color_by == ColorBy.COMMUNITY:
            unique_communities, community_indices = np.unique(community_assignments, return_inverse=True)
            return community_indices
        elif self.color_by == ColorBy.DEGREE:
            return np.sum(adjacency_matrix, axis=1)

    def _compute_lines(self, positions, adjacency_matrix):
        rows, cols = np.where(np.triu(adjacency_matrix, 1))
        return np.array([[positions[i], positions[j]] for i, j in zip(rows, cols)])

    def _initialize_plot(self, adjacency_matrix, activities, positions_array, community_assignments):
        # Set up axes
        self.ax.set(xlim=(0, 100), ylim=(0, 100), aspect='equal', xticks=[], yticks=[])

        # Initialize scatter plot for nodes
        self.scatter = self.ax.scatter(
            *positions_array.T,
            c=self._compute_vertex_colors(adjacency_matrix, activities, community_assignments),
            cmap=self.color_by.value,
            s=5,
            zorder=2
        )

        # Initialize lines (edges)
        lines = self._compute_lines(positions_array, adjacency_matrix)
        if len(lines) == 0:
            return

        network_density = len(lines) / (len(adjacency_matrix) * (len(adjacency_matrix) - 1) / 2)
        self.lines = LineCollection(lines, colors=[0.5, 0.5, 0.5], linewidths=0.15 - 0.1 * (network_density ** 0.5),
                                            alpha=0.25 - 0.2 * (network_density ** 0.5), zorder=1)
        self.ax.add_collection(self.lines)

    def draw_visual(self, adjacency_matrix, activities, graph, community_assignments, step, max_iter=0):
        # try:
        # Update positions, colors, lines
        positions_array = self._compute_layout(adjacency_matrix, max_iter=25)
        self.scatter.set_offsets(positions_array)
        self.scatter.set_array(self._compute_vertex_colors(adjacency_matrix, activities, community_assignments))
        self.scatter.set_clim([min(self.scatter.get_array()), max(self.scatter.get_array())])   # Rescale the colormap in case our min/max have changed (common with community coloring)
        self.lines.set_segments(self._compute_lines(positions_array, adjacency_matrix))

        # Redraw the canvas
        self.fig.canvas.draw_idle()
        self.ax.figure.canvas.flush_events()

        # Save the figure
        image_path = os.path.join(self.output_dir, f"{step}.png")
        self.fig.savefig(image_path)
        # except Exception as e:
        #     print(f"Error drawing visual: {e}") # We often don't mind if the image breaks, so just print the error

    def close(self):
        plt.close(self.fig)
