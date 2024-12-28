from graph_tool.draw import graph_draw, sfdp_layout, arf_layout, fruchterman_reingold_layout
import numpy as np
import logging
from enum import Enum
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ColorBy(Enum):
    ACTIVITY = "cividis"
    CLUSTER = "Set1"
    CONNECTIONS = "inferno"

class Visualization:
    def __init__(self, graph, activities, cluster_assignments, layout_type="fr", output_dir="visuals", color_by=ColorBy.ACTIVITY):
        self.logger = logging.getLogger(__name__)
        self.graph = graph
        self.activities = activities
        self.cluster_assignments = cluster_assignments
        self.output_dir = output_dir
        self.color_by = color_by

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # Choose layout type
        if layout_type.lower() == "sfdp":
            self.positions = sfdp_layout(graph)
        elif layout_type.lower() == "arf":
            self.positions = arf_layout(graph)
        elif layout_type.lower() == "fr":
            self.positions = fruchterman_reingold_layout(graph)
        else:
            self.logger.error(f"Unsupported layout type: {layout_type}")
            raise ValueError("Unsupported layout type. Choose 'sfdp', 'arf', or 'fr'.")

        # Compute vertex colors
        self.vertex_colors = self._compute_vertex_colors()

    def _compute_vertex_colors(self):
        """Compute vertex colors based on activity or cluster assignments."""
        colors = self.graph.new_vertex_property("vector<float>")

        if self.color_by == ColorBy.ACTIVITY:
            normalized_activities = (self.activities.a - np.min(self.activities.a)) / (
                np.max(self.activities.a) - np.min(self.activities.a)
            )
            for v in self.graph.vertices():
                colors[v] = [normalized_activities[int(v)], 0.5, 1 - normalized_activities[int(v)]]  # RGB mapping
        elif self.color_by == ColorBy.CLUSTER:
            unique_clusters = np.unique(self.cluster_assignments)
            color_map = {cluster: np.random.rand(3) for cluster in unique_clusters}
            for v in self.graph.vertices():
                colors[v] = color_map[self.cluster_assignments[int(v)]]
        elif self.color_by == ColorBy.CONNECTIONS:
            degrees = self.graph.get_total_degrees(self.graph.get_vertices())
            max_degree = max(degrees)
            min_degree = min(degrees)
            normalized_degrees = (degrees - min_degree) / (max_degree - min_degree)
            for v in self.graph.vertices():
                colors[v] = [normalized_degrees[int(v)], 0.5, 0.5]
        else:
            self.logger.error(f"Unsupported color_by option: {self.color_by}")
            raise ValueError(f"Unsupported color_by option: {self.color_by}")

        return colors

    def update_positions(self, layout_type="arf"):
        """Update the positions of the vertices based on the selected layout."""
        if layout_type.lower() == "sfdp":
            self.positions = sfdp_layout(self.graph)
        elif layout_type.lower() == "arf":
            self.positions = arf_layout(self.graph)
        elif layout_type.lower() == "fr":
            self.positions = fruchterman_reingold_layout(self.graph)
        else:
            self.logger.error(f"Unsupported layout type for update: {layout_type}")
            raise ValueError("Unsupported layout type. Choose 'sfdp', 'arf', or 'fr'.")
        self.logger.info(f"Updated positions using layout: {layout_type}")

    def refresh_visual(self):
        """Recompute positions and vertex colors to reflect graph changes."""
        self.update_positions()
        self.vertex_colors = self._compute_vertex_colors()

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
