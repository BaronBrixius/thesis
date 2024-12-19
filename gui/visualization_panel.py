from tkinter import ttk
from network_simulation.visualization import Visualization, ColorBy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import networkx as nx
import numpy as np

matplotlib.use("TkAgg")

class VisualizationPanel:
    def __init__(self, root, network):
        self.root = root
        self.network = network
        self.visualizer = Visualization(
            positions=network.positions,
            activities=network.activities,
            adjacency_matrix=network.adjacency_matrix,
            cluster_assignments=self.detect_communities(nx.from_numpy_array(network.adjacency_matrix)),
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

    def update(self, network, step):
        self.visualizer.update_plot(
            positions=network.positions,
            activities=network.activities,
            adjacency_matrix=network.adjacency_matrix,
            cluster_assignments=self.detect_communities(nx.from_numpy_array(network.adjacency_matrix)),
            title=f"Nodes: {network.num_nodes}, Connections: {network.num_connections}, Step: {step}",
            draw_lines=True
        )
        self.canvas.draw()

    def update_network(self, network):
        self.network = network

    def detect_communities(self, graph):    #FIXME This really cannot be here haha, I'm just too tired right now
        """
        Louvain Method For Community Detection: Assigns nodes to communities for modularity and cluster stability calculations.
        """
        communities = nx.algorithms.community.louvain_communities(graph)
        cluster_assignments = np.zeros(len(graph.nodes), dtype=int)

        for cluster_id, community in enumerate(communities):
            cluster_assignments[list(community)] = cluster_id
        return cluster_assignments