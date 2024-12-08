from network_simulation.visualization import Visualization, ColorBy

class NetworkVisualizer:
    def __init__(self, network):
        self.visualizer = Visualization(
            positions=network.positions,
            activities=network.activities,
            adjacency_matrix=network.adjacency_matrix,
            color_by=ColorBy.ACTIVITY,
            draw_lines=True,
            show=False
        )

    def update(self, network, title=None):
        self.visualizer.update_plot(
            positions=network.positions,
            activities=network.activities,
            adjacency_matrix=network.adjacency_matrix,
            title=title,
            draw_lines=True
        )
