from network_simulation.visualization import Visualization, ColorBy
import matplotlib.pyplot as plt
import numpy as np

class NetworkVisualizer:
    def __init__(self, network, color_by=ColorBy.ACTIVITY):
        """
        Initialize the NetworkVisualizer.
        :param network: NodeNetwork instance
        :param color_by: ColorBy Enum (e.g., ColorBy.ACTIVITY or ColorBy.CONNECTIONS)
        """
        self.network = network
        self.color_by = color_by
        self.visualizer = None

    def show(self, step):
        """
        Display or update the visualization.
        """
        positions = self.network.physics.positions
        activities = self.network.activities
        adjacency_matrix = self.network.adjacency_matrix

        if self.visualizer is None:
            # Initialize the visualization if not already created
            self.visualizer = Visualization(
                positions=positions,
                activities=activities,
                adjacency_matrix=adjacency_matrix,
                color_by=self.color_by,
                draw_lines=True,
                show=True
            )
        else:
            # Update the visualization with new data
            step = step
            epsilon = self.network.epsilon
            title = f"Step: {step}, Epsilon: {epsilon:.3f}"
            self.visualizer.update_plot(
                positions=positions,
                activities=activities,
                adjacency_matrix=adjacency_matrix,
                title=title,
                draw_lines=True
            )

    def close(self):
        """
        Close the visualization.
        """
        if self.visualizer:
            self.visualizer.close()
            self.visualizer = None
