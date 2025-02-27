import os
from enum import Enum
import matplotlib
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    from graph_tool.draw import arf_layout
matplotlib.use('Agg')

class ColorBy(Enum):
    ACTIVITY = "cividis"
    COMMUNITY = "Set1"
    DEGREE = "inferno"

class Visualization:
    def __init__(self, adjacency_matrix, activities, graph, community_assignments, output_dir="foo", color_by=ColorBy.ACTIVITY):
        self.output_dir = os.path.join(output_dir, "images")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.color_by = color_by
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.positions = None
        positions_array = self._compute_layout(graph)
        self._initialize_plot(adjacency_matrix, activities, positions_array, community_assignments)

    def _compute_layout(self, graph, max_iter=0):
        self.positions = arf_layout(g=graph, dt=1e-4, epsilon=10_000, max_iter=min(max_iter, 1000), pos=self.positions)
        positions_array = self.positions.get_2d_array().T
        # Normalize positions to be within (-0.9, 0.9)
        normalized_positions_array = -0.9 + 1.8 * (positions_array - positions_array.min()) / (positions_array.max() - positions_array.min())
        normalized_positions_array -= (normalized_positions_array.max(axis=0) + normalized_positions_array.min(axis=0)) / 2 # TODO this centers, but maybe only on one axis?
        return normalized_positions_array

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
        self.ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect='equal', xticks=[], yticks=[])

        # Initialize scatter plot for nodes
        self.scatter = self.ax.scatter(
            *positions_array.T,
            c=self._compute_vertex_colors(adjacency_matrix, activities, community_assignments),
            cmap=self.color_by.value,
            s=10,
            zorder=2
        )

        # Initialize lines (edges)
        lines = self._compute_lines(positions_array, adjacency_matrix)
        if len(lines) == 0:
            return

        network_density = len(lines) / (len(adjacency_matrix) * (len(adjacency_matrix) - 1) / 2)
        self.lines = LineCollection(lines, colors=[0.4, 0.4, 0.4], linewidths=0.3 - 0.2 * (network_density ** 0.5),
                                            alpha=0.7 - 0.4 * (network_density ** 0.5), zorder=1)
        self.ax.add_collection(self.lines)

    def draw_visual(self, adjacency_matrix, activities, graph, community_assignments, step, max_iter=0):
        try:    # Update positions, colors, lines
            positions_array = self._compute_layout(graph, max_iter=max_iter)
            self.scatter.set_offsets(positions_array)
            self.scatter.set_array(self._compute_vertex_colors(adjacency_matrix, activities, community_assignments))
            self.lines.set_segments(self._compute_lines(positions_array, adjacency_matrix))

            # Redraw the canvas
            self.fig.canvas.draw_idle()
            self.ax.figure.canvas.flush_events()

            # Save the figure
            image_path = os.path.join(self.output_dir, f"{step}.png")
            self.fig.savefig(image_path)
        except Exception as e:
            print(f"Error drawing visual: {e}") # We often don't mind if the image breaks, so just print the error for checking
