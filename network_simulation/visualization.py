from network_simulation.network import NodeNetwork
from graph_tool.draw import graph_draw, arf_layout
import numpy as np
import logging
from enum import Enum
import os
from matplotlib import cm

class ColorBy(Enum):
    ACTIVITY = "cividis"
    CLUSTER = "Set1"
    DEGREE = "inferno"

class Visualization:
    def __init__(self, network:NodeNetwork, output_dir="foo", color_by=ColorBy.ACTIVITY):
        self.logger = logging.getLogger(__name__)
        self.color_by = color_by

        self.output_dir = os.path.join(output_dir, "images")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.positions = self._compute_layout(network.graph)
        self.colors = network.graph.new_vertex_property("vector<float>")

        # array where the first 50 values are "square" and the next 250 are "circle"
        self.vertex_shapes = network.graph.new_vertex_property("int")
        self.vertex_shapes.a = [2 if i < 50 else 0 for i in range(network.num_nodes)]

    def _compute_layout(self, graph, old_positions=None, max_iter=0):
        try:
            return arf_layout(graph, pos=old_positions, epsilon=10000, max_iter=max_iter)
        except Exception as e:
            self.logger.error(f"Layout computation failed: {e}")
            return None

    def _compute_vertex_colors(self, network: NodeNetwork, step):
        colormap = cm.get_cmap(self.color_by.value)

        if self.color_by == ColorBy.ACTIVITY:
            min_val, max_val = np.min(network.activities.a), np.max(network.activities.a)
            if max_val == min_val:
                normalized_activities = np.zeros_like(network.activities.a)  # All the same
            else:
                normalized_activities = (network.activities.a - min_val) / (max_val - min_val)
            for v in network.graph.vertices():
                self.colors[v] = colormap(normalized_activities[int(v)])[:3]  # RGB only

        elif self.color_by == ColorBy.CLUSTER:
            cluster_assignments = network.metrics.get_cluster_assignments(network.graph, step)
            unique_clusters = np.unique(cluster_assignments)
            color_map = {cluster: colormap(i / len(unique_clusters))[:3] for i, cluster in enumerate(unique_clusters)}

            for v in network.graph.vertices():
                self.colors[v] = color_map[cluster_assignments[int(v)]]

        elif self.color_by == ColorBy.DEGREE:
            min_degree, max_degree = np.min(network.degrees.a), np.max(network.degrees.a)
            if max_degree == min_degree:
                normalized_degrees = np.zeros_like(network.degrees.a)  # All the same
            else:
                normalized_degrees = (network.degrees.a - min_degree) / (max_degree - min_degree)
            for v in network.graph.vertices():
                self.colors[v] = colormap(normalized_degrees[int(v)])[:3]

        else:
            raise ValueError(f"Unsupported color_by option: {self.color_by}")

    def draw_visual(self, network:NodeNetwork, step, max_iter=0, ax=None):
        self.positions = self._compute_layout(network.graph, self.positions, max_iter)
        self._compute_vertex_colors(network, step)
        density = network.num_connections / (network.num_nodes * (network.num_nodes - 1) / 2)
        # Create edge color property based on weights
        edge_colors = network.graph.new_edge_property("vector<float>")
        for e in network.graph.edges():
            source, target = int(e.source()), int(e.target())

            is_minority_edge = (source < 50 and target < 50)
            is_mixed_edge = (source < 50) != (target < 50)

            if is_minority_edge:
                edge_colors[e] = [0.4, 0.4, 1, 0.5]  # blue
            elif is_mixed_edge:
                edge_colors[e] = [0.4, 1, 0.4, 0.5]  # green
            else:   # majority edge
                edge_colors[e] = [1, 0.4, 0.4, 0.5]  # red

        output_path = os.path.join(self.output_dir, f"{step}.png")
        artist = graph_draw(
            network.graph,
            pos=self.positions,
            output=output_path,
            mplfig=ax,
            vertex_size=7.0,
            vertex_shape=self.vertex_shapes,
            edge_pen_width=0.35 - 0.2 * (density ** 0.5),
            edge_color=edge_colors,
            vertex_fill_color=self.colors,
            vertex_pen_width=0,
            bg_color=[1, 1, 1, 1],
        )
        if ax:
            artist.fit_view()
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
