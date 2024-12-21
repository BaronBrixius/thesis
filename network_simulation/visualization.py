from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import numpy as np

class ColorBy(Enum):
    ACTIVITY = "cividis"
    CLUSTER = "Set1"
    CONNECTIONS = "inferno"

class Visualization:
    def __init__(self, positions, activities, adjacency_matrix, cluster_assignments, draw_lines=True, show=True, color_by:ColorBy=ColorBy.ACTIVITY):
        self.color_by = color_by
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        # Initialize marked nodes (hardcoding for now)
        self.marked = np.zeros(len(positions), dtype=bool)
        # self.marked[:3] = True  # Hardcode first three nodes as marked
        # for i in [192, 193, 69, 70, 75, 140, 12, 88, 26, 92, 93, 158, 31, 30, 161, 34, 168, 106, 46, 113, 49, 180, 123, 61]:
            # self.marked[i] = True

        self.custom_colormap = self.create_custom_colormap(self.color_by.value)

        self.initialize_plot(positions, activities, adjacency_matrix, cluster_assignments, draw_lines=draw_lines)
        if show:
            plt.ion()
            plt.show()

    def create_custom_colormap(self, base_colormap_name):
        base_colormap = plt.get_cmap(base_colormap_name)
        new_colors = base_colormap(np.linspace(0, 1, 256))
        new_colors[0] = np.array([1, 0, 0, 1])  # Red in RGBA, for marked nodes
        return ListedColormap(new_colors)

    def compute_lines(self, positions, adjacency_matrix):
        rows, cols = np.where(np.triu(adjacency_matrix, 1))
        connections = np.array([[positions[i], positions[j]] for i, j in zip(rows, cols)])
        return connections, rows, cols

    def compute_node_colors(self, adjacency_matrix, activities, cluster_assignments=None):
        if self.color_by == ColorBy.CONNECTIONS:
            colors = np.sum(adjacency_matrix, axis=1)
        elif self.color_by == ColorBy.ACTIVITY:
            colors = np.copy(activities)
        elif self.color_by == ColorBy.CLUSTER:
            colors = cluster_assignments / np.max(cluster_assignments) * 1.75 - 0.75  # Normalize cluster assignments to [-1, ~1]
        else:
            raise ValueError(f"Unsupported color_by value: {self.color_by}")


        colors[self.marked] = -1  # Set marked nodes to -1 to map to red in the custom colormap
        # print(colors)
        return colors


    def initialize_plot(self, positions, activities, adjacency_matrix, cluster_assignments=None, draw_lines=True):
        self.ax.set_xlim(-0.05, 1.05)
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Initialize scatter plot for nodes
        self.scatter = self.ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=self.compute_node_colors(adjacency_matrix, activities, cluster_assignments),
            cmap=self.custom_colormap,
            s=10,
            zorder=2
        )

        # Initialize lines (connections)
        if draw_lines:
            lines, rows, cols = self.compute_lines(positions, adjacency_matrix)
            if len(lines) > 0:
                edge_colors = ['red' if self.marked[row] or self.marked[col] else 'gray' for row, col in zip(rows, cols)]
                self.line_collection = LineCollection(lines, colors=edge_colors, linewidths=0.5, alpha=0.6, zorder=1)
                self.ax.add_collection(self.line_collection)

    def update_plot(self, positions, activities, adjacency_matrix, cluster_assignments, title, draw_lines=True):
        self.ax.set_title(title)

        # Check for invalid values
        if not np.all(np.isfinite(positions)):
            print("Error: Invalid positions detected (NaN or inf).")
            return

        # Update node colors and positions
        self.scatter.set_offsets(positions)
        self.scatter.set_array(self.compute_node_colors(adjacency_matrix, activities, cluster_assignments))

        # Update connection lines
        if draw_lines:
            lines, rows, cols = self.compute_lines(positions, adjacency_matrix)
            edge_colors = ['red' if self.marked[row] or self.marked[col] else 'gray' for row, col in zip(rows, cols)]
            self.line_collection.set_segments(lines)
            self.line_collection.set_color(edge_colors)

        # Redraw the canvas
        self.fig.canvas.draw_idle()
        self.ax.figure.canvas.flush_events()

    def close(self):
        plt.close(self.fig)
