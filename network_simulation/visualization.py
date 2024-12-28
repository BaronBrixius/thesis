from graph_tool.draw import graph_draw, sfdp_layout, arf_layout, fruchterman_reingold_layout
import numpy as np
import logging
from enum import Enum
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

class ColorBy(Enum):
    ACTIVITY = "cividis"
    CLUSTER = "Set1"
    CONNECTIONS = "inferno"

class Visualization:
    def __init__(self, graph, activities, cluster_assignments, output_dir="visuals", color_by=ColorBy.CLUSTER):
        self.logger = logging.getLogger(__name__)
        self.graph = graph
        self.activities = activities
        self.cluster_assignments = cluster_assignments
        self.output_dir = output_dir
        self.color_by = color_by

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.positions = arf_layout(graph)  #fruchterman_reingold_layout(graph)
        
        # Compute vertex colors
        self.vertex_colors = self._compute_vertex_colors(self.graph, self.cluster_assignments, self.color_by, self.activities)

    def _compute_vertex_colors(self, graph, cluster_assignments, color_by, activities):
        """
        Compute vertex colors based on the specified color mode.
        """
        colors = graph.new_vertex_property("vector<float>")

        if color_by == ColorBy.ACTIVITY:
            normalized_activities = (activities.a - np.min(activities.a)) / (
                np.max(activities.a) - np.min(activities.a)
            )
            for v in graph.vertices():
                colors[v] = [normalized_activities[int(v)], 0.5, 1 - normalized_activities[int(v)]]  # RGB mapping

        elif color_by == ColorBy.CLUSTER:
            unique_clusters = np.unique(cluster_assignments)
            colormap = cm.get_cmap("Set1", len(unique_clusters))  # Use Set1 colormap for vibrant cluster colors
            color_map = {cluster: colormap(i)[:3] for i, cluster in enumerate(unique_clusters)}

            for v in graph.vertices():
                colors[v] = color_map[cluster_assignments[int(v)]]

        elif color_by == ColorBy.CONNECTIONS:
            degrees = graph.get_total_degrees(graph.get_vertices())
            max_degree = max(degrees)
            min_degree = min(degrees)
            normalized_degrees = (degrees - min_degree) / (max_degree - min_degree)
            for v in graph.vertices():
                colors[v] = [normalized_degrees[int(v)], 0.5, 0.5]
        else:
            raise ValueError(f"Unsupported color_by option: {color_by}")

        return colors

    def refresh_visual(self):
        """Recompute positions and vertex colors to reflect graph changes."""
        self.positions = arf_layout(self.graph, pos=self.positions, max_iter=25)    # fruchterman_reingold_layout(self.graph, pos=self.positions, n_iter=10)
        self.vertex_colors = self._compute_vertex_colors(self.graph, self.cluster_assignments, self.color_by, self.activities)

    def save_visual(self, step):
        """Save a static visual of the graph."""
        output_path = os.path.join(self.output_dir, f"{step}.png")
        try:
            graph_draw(
                self.graph,
                pos=self.positions,
                vertex_fill_color=self.vertex_colors,
                output=output_path
            )
            self.logger.info(f"Saved visualization at step {step} to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save visualization at step {step}: {e}")
