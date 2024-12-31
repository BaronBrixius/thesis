from network_simulation.network import NodeNetwork
from graph_tool.draw import graph_draw, sfdp_layout, arf_layout, fruchterman_reingold_layout
import numpy as np
import logging
from enum import Enum
import os
from matplotlib import cm

class ColorBy(Enum):
    ACTIVITY = cm.get_cmap("cividis")
    CLUSTER = cm.get_cmap("Set1")
    CONNECTIONS = cm.get_cmap("inferno")

class Visualization:
    def __init__(self, network:NodeNetwork, output_dir="sim", color_by=ColorBy.ACTIVITY):
        self.logger = logging.getLogger(__name__)
        self.color_by = color_by

        self.output_dir = os.path.join(output_dir, "images")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.positions = self.compute_layout(network.graph, None, 100)
        self.colors = network.graph.new_vertex_property("vector<float>")
        self.colors = self._compute_vertex_colors(network, 0)

    def compute_layout(self, graph, old_positions=None, max_iter=25):
        """
        Compute or update the layout positions using the chosen algorithm.
        """
        try:
            return arf_layout(graph, pos=old_positions, max_iter=max_iter)
        except Exception as e:
            self.logger.error(f"Layout computation failed: {e}")
            return None

    def _compute_vertex_colors(self, network:NodeNetwork, step):
        """
        Update vertex colors based on the specified color mode.
        """
        colors = network.graph.new_vertex_property("vector<float>")
        colormap = self.color_by.value

        if self.color_by == ColorBy.ACTIVITY:
            normalized_activities = (network.activities.a - np.min(network.activities.a)) / (
                np.max(network.activities.a) - np.min(network.activities.a)
            )
            for v in network.graph.vertices():
                colors[v] = [normalized_activities[int(v)], 0.5, 1 - normalized_activities[int(v)]]  # RGB mapping

        elif self.color_by == ColorBy.CLUSTER:
            cluster_assignments = network.metrics.get_cluster_assignments(network.graph, step)
            unique_clusters = np.unique(cluster_assignments)
            color_map = {cluster: colormap(i)[:3] for i, cluster in enumerate(unique_clusters)}

            for v in network.vertices:
                colors[v] = color_map[cluster_assignments[int(v)]]

        elif self.color_by == ColorBy.CONNECTIONS:
            degrees = network.degrees.a
            max_degree = max(degrees)
            min_degree = min(degrees)
            normalized_degrees = (degrees - min_degree) / (max_degree - min_degree)
            for v in network.vertices:
                colors[v] = [normalized_degrees[int(v)], 0.5, 0.5]
        else:
            raise ValueError(f"Unsupported color_by option: {self.color_by}")
        
        return colors

    def refresh_visual(self, network:NodeNetwork, step, max_iter=25):
        """Recompute positions and vertex colors to reflect graph changes."""
        self.positions = self.compute_layout(network.graph, self.positions, max_iter)
        self.colors = self._compute_vertex_colors(network, step)

    def save_visual(self, network:NodeNetwork, step):
        """Save a static visual of the graph."""
        output_path = os.path.join(self.output_dir, f"{step}.png")
        graph_draw(
            network.graph,
            pos=self.positions,
            vertex_fill_color=self.colors.a,  # Truncate RGBA to RGB
            output=output_path,
            # output_size=(800, 600),
        )
        self.logger.info(f"Saved visualization at step {step} to {output_path}")
