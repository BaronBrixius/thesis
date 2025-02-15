from network_simulation.network import NodeNetwork
from network_simulation.physics import Physics
from graph_tool.draw import arf_layout
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import logging
from enum import Enum
import os
from matplotlib import cm
from graph_tool.all import Graph
from network_simulation.utils import start_timing, stop_timing
class ColorBy(Enum):
    ACTIVITY = "cividis"
    CLUSTER = "Set1"
    DEGREE = "inferno"

class Visualization:
    def __init__(self, network, output_dir="foo", color_by=ColorBy.ACTIVITY):
        self.logger = logging.getLogger(__name__)
        self.color_by = color_by
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.output_dir = os.path.join(output_dir, "images")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.positions = np.random.uniform([0.1, 0.1], [0.9, 0.9], (network.num_nodes, 2))
        self.physics = Physics(normal_distance=(0.5 * np.sqrt(network.num_connections + network.num_nodes) / network.num_nodes))

        self.initialize_plot(self.positions, network.activities, network.adjacency_matrix)

    def _compute_layout(self, graph, max_iter=0):
        start_timing("arf_layout")
        pos = graph.vertex_properties.get("pos", graph.new_vertex_property("vector<double>"))
        for v in graph.vertices():
            pos[v] = self.positions[int(v)]
        pos = arf_layout(graph, pos=pos, epsilon=10000, max_iter=max_iter)
        positions = np.array([pos[v] for v in graph.vertices()])
        self.positions = positions
        stop_timing("arf_layout")
        return self.positions

    def _compute_vertex_colors(self, adjacency_matrix, activities, cluster_assignments=None):
        if self.color_by == ColorBy.ACTIVITY:
            colors = np.copy(activities)
        elif self.color_by == ColorBy.CLUSTER:
            if cluster_assignments is not None:
                unique_values = np.unique(cluster_assignments)
                value_map = {old: new for new, old in enumerate(unique_values)}
                colors = np.array([value_map[x] for x in cluster_assignments])
            else:
                colors = np.zeros(len(adjacency_matrix), dtype=int)  # Default to zero if no assignments are available
        elif self.color_by == ColorBy.DEGREE:
            colors = np.sum(adjacency_matrix, axis=1)
        else:
            raise ValueError(f"Unsupported color_by option: {self.color_by}")

        return colors

    def _compute_lines(self, positions, adjacency_matrix):
        rows, cols = np.where(np.triu(adjacency_matrix, 1))
        connections = np.array([[positions[i], positions[j]] for i, j in zip(rows, cols)])
        return connections

    def initialize_plot(self, positions, activities, adjacency_matrix):
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Initialize scatter plot for nodes
        self.scatter = self.ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=self._compute_vertex_colors(adjacency_matrix, activities, None),
            cmap=self.color_by.value,
            s=10,
            zorder=2
        )

        # Initialize lines (connections)
        lines = self._compute_lines(positions, adjacency_matrix)
        if len(lines) > 0:
            density = len(lines) / (len(activities) * (len(activities) - 1) / 2)
            edge_pen_width=0.3 - 0.2 * (density ** 0.5)
            edge_color=[0.4, 0.4, 0.4]
            alpha = 0.7 - 0.4 * (density ** 0.5)
            self.line_collection = LineCollection(lines, colors=edge_color, linewidths=edge_pen_width, alpha=alpha, zorder=1)
            self.ax.add_collection(self.line_collection)

    def draw_visual(self, adjacency_matrix, activities, community_assignments, graph, step, max_iter=0, title=None):
        self.ax.set_title(title)

        self.positions = self._compute_layout(graph, max_iter=max_iter)

        # Update node colors and positions
        self.scatter.set_offsets(self.positions)
        self.scatter.set_array(self._compute_vertex_colors(adjacency_matrix, activities, community_assignments))

        # Update connection lines
        lines = self._compute_lines(self.positions, adjacency_matrix)
        self.line_collection.set_segments(lines)

        # Redraw the canvas
        self.fig.canvas.draw_idle()
        self.ax.figure.canvas.flush_events()

        image_path = os.path.join(self.output_dir, f"{step}.png")
        self.fig.savefig(image_path)
        self.logger.info(f"Saved network visualization for step {step} to {image_path}")

        # artist = graph_draw(
        #     network.graph,
        #     pos=self.positions,
        #     output=output_path,
        #     mplfig=ax,
        #     vertex_size=7.0,
        #     edge_pen_width=0.3 - 0.2 * (density ** 0.5),
        #     edge_color=[0.4, 0.4, 0.4, 0.7 - 0.4 * (density ** 0.5)],
        #     vertex_fill_color=self.colors,
        #     vertex_pen_width=0,
        #     bg_color=[1, 1, 1, 1],
        # )
        # if ax:
        #     artist.fit_view()
        #     ax.set_xlim(-1, 1)
        #     ax.set_ylim(-1, 1)