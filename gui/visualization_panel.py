from tkinter import ttk
from network_simulation.network import NodeNetwork
from network_simulation.visualization import Visualization, ColorBy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import networkx as nx
import numpy as np

matplotlib.use("TkAgg")

class VisualizationPanel:
    def __init__(self, root, network:NodeNetwork):
        self.root = root
        self.network = network
        self.visualizer = Visualization(
            positions=network.positions,
            activities=network.activities,
            adjacency_matrix=network.get_adjacency_matrix(),
            cluster_assignments=network.metrics.get_cluster_assignments(network.get_adjacency_matrix(), step=0),
            color_by=ColorBy.CLUSTER,
            draw_lines=True,
            show=False
        )
        self.create_canvas()

    def create_canvas(self):
        frame = ttk.Frame(self.root)
        frame.grid(row=0, column=1, sticky="NSEW")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.canvas = FigureCanvasTkAgg(self.visualizer.fig, master=frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="NSEW")
        self.canvas.draw()

    def update(self, step):
        self.visualizer.update_plot(
            positions=self.network.positions,
            activities=self.network.activities,
            adjacency_matrix=self.network.get_adjacency_matrix(),
            cluster_assignments=self.network.metrics.get_cluster_assignments(self.network.get_adjacency_matrix(), step),
            title=f"Nodes: {self.network.num_nodes}, Connections: {self.network.num_connections}, Step: {step}",
            draw_lines=True
        )
        self.canvas.draw()
