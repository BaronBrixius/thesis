from tkinter import ttk
from network_simulation.network_gpu import NodeNetwork
from network_simulation.visualization import Visualization, ColorBy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib

class VisualizationPanel:
    def __init__(self, root, network: NodeNetwork, output_dir="sim"):
        matplotlib.use("TkAgg")
        self.root = root
        self.network = network
        self.visualizer = Visualization(network, output_dir=output_dir, color_by=ColorBy.COMMUNITY)

        self.create_canvas()
        self.update(step=0, max_iter=1)

    def create_canvas(self):
        frame = ttk.Frame(self.root)
        frame.grid(row=0, column=1, sticky="NSEW")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        # Create a Matplotlib figure for embedding visualization
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="NSEW")
        self.canvas.get_tk_widget().config(width=800, height=600)  # Set canvas size
        self.canvas.draw()

    def update(self, step, max_iter):
        """Update the visualization with the current step."""
        self.visualizer.draw_visual(self.network, step, max_iter, ax=self.ax)
        self.ax.set_title(f"Nodes: {self.network.num_nodes}, Edges: {self.network.num_edges}, Step: {step}")
        self.canvas.draw()
