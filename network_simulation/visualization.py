from network_simulation.network import NodeNetwork
from graph_tool.draw import graph_draw, arf_layout, fruchterman_reingold_layout
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

        self.positions = self._compute_layout(network.graph, None, 100)
        self.colors = network.graph.new_vertex_property("vector<float>")

    def _compute_layout(self, graph, old_positions=None, max_iter=25):
        """
        Compute or update the layout positions using the chosen algorithm.
        """
        try:
            layout = arf_layout(graph, pos=old_positions, epsilon=10000, max_iter=0)
            return layout
        except Exception as e:
            self.logger.error(f"Layout computation failed: {e}")
            return None

    def _compute_vertex_colors(self, network: NodeNetwork, step):
        """
        Update vertex colors based on the specified color mode.
        """
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

    def refresh_visual(self, network:NodeNetwork, step, max_iter=25):
        """Recompute positions and vertex colors to reflect graph changes."""
        self.positions = self._compute_layout(network.graph, self.positions, max_iter)
        self._compute_vertex_colors(network, step)

    def draw_visual(self, network:NodeNetwork, step, display_interval, ax=None):
        """Save a static visual of the graph."""
        self.refresh_visual(network, step, max_iter=min(25, display_interval))
        density = network.num_connections / (network.num_nodes * (network.num_nodes - 1) / 2)
        output_path = os.path.join(self.output_dir, f"{step}.png")
        artist = graph_draw(
            network.graph,
            pos=self.positions,
            output=output_path,
            mplfig=ax,
            vertex_size=7.0,
            edge_pen_width=0.3 - 0.2 * (density ** 0.5),
            edge_color=[0.4, 0.4, 0.4, 0.7 - 0.4 * (density ** 0.5)],
            vertex_fill_color=self.colors,
            vertex_pen_width=0,
            bg_color=[1, 1, 1, 1],
        )
        if ax:
            artist.fit_view()
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
