from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

class ColorBy(Enum):
    ACTIVITY = "activity"
    CONNECTIONS = "connections"

class Visualization:
    COLOR_MAPS = {
        ColorBy.ACTIVITY: 'cividis',
        ColorBy.CONNECTIONS: 'inferno'
    }

    def __init__(self, positions, activities, adjacency_matrix, draw_lines=True, show=True, color_by:ColorBy=ColorBy.ACTIVITY):
        self.color_by = color_by  # 'activity' or 'connections'
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.initialize_plot(positions, activities, adjacency_matrix, draw_lines=draw_lines)
        if show:
            plt.ion()
            plt.show()

    def compute_lines(self, positions, adjacency_matrix):
        rows, cols = np.where(np.triu(adjacency_matrix, 1))
        connections = np.array([[positions[i], positions[j]] for i, j in zip(rows, cols)])
        return connections

    def compute_node_colors(self, adjacency_matrix, activities):
        if self.color_by == ColorBy.CONNECTIONS:
            node_degrees = np.sum(adjacency_matrix, axis=1)
            return node_degrees
        elif self.color_by == ColorBy.ACTIVITY:
            return activities
        else:
            raise ValueError(f"Unsupported color_by value: {self.color_by}")

    def initialize_plot(self, positions, activities, adjacency_matrix, draw_lines=True):
        self.ax.set_xlim(-0.05, 1.05)
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Initialize scatter plot for nodes
        self.scatter = self.ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=self.compute_node_colors(adjacency_matrix, activities),
            cmap=self.COLOR_MAPS[self.color_by],
            s=10,
            zorder=2
        )

        # Initialize lines (connections)
        if draw_lines:
            lines = self.compute_lines(positions, adjacency_matrix)
            if len(lines) > 0:
                self.line_collection = LineCollection(lines, colors='gray', linewidths=0.5, alpha=0.6, zorder=1)
                self.ax.add_collection(self.line_collection)

    def update_plot(self, positions, activities, adjacency_matrix, title, draw_lines=True):
        self.ax.set_title(title)

        # Check for invalid values
        if not np.all(np.isfinite(positions)):
            print("Error: Invalid positions detected (NaN or inf).")
            return

        # Update node colors and positions
        self.scatter.set_offsets(positions)
        self.scatter.set_array(self.compute_node_colors(adjacency_matrix, activities))

        # Update connection lines
        if draw_lines:
            self.line_collection.set_segments(self.compute_lines(positions, adjacency_matrix))

        # Redraw the canvas
        self.fig.canvas.draw_idle()
        self.ax.figure.canvas.flush_events()

    def close(self):
        plt.close(self.fig)
