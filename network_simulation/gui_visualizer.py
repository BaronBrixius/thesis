from network_simulation.visualization import Visualization, ColorBy
import matplotlib.pyplot as plt
import numpy as np

class GUIVisualizer:
    def __init__(self, network, color_by=ColorBy.ACTIVITY):
        """
        Initialize the NetworkVisualizer.
        :param network: NodeNetwork instance
        :param color_by: ColorBy Enum (e.g., ColorBy.ACTIVITY or ColorBy.CONNECTIONS)
        """
        self.network = network
        self.color_by = color_by
        self.visualizer = Visualization(
                positions=network.physics.positions,
                activities=network.activities,
                adjacency_matrix=network.adjacency_matrix,
                color_by=self.color_by,
                draw_lines=True,
                show=False
            )

    def show(self, step):
        """
        Display or update the visualization.
        """
        # Update the visualization with new data
        step = step
        epsilon = self.network.epsilon
        title = f"Step: {step}, Epsilon: {epsilon:.3f}"
        self.visualizer.update_plot(
            positions=self.network.physics.positions,
            activities=self.network.activities,
            adjacency_matrix=self.network.adjacency_matrix,
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
