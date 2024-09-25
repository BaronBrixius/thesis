import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm
from matplotlib.animation import FuncAnimation

class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.position = np.random.uniform(0, 1, 2)  # Random x, y position
        self.activity = np.random.uniform(-1, 1)    # Random initial activity
        self.connections = []

    def update_activity(self, a=1.7):
        self.activity = 1 - a * self.activity**2    # logistic map: x(n+1) = f(x(n)) = 1 - ax(n)²

def add_connection(node_a, node_b):
    if node_b not in node_a.connections:
        node_a.connections.append(node_b)
    if node_a not in node_b.connections:
        node_b.connections.append(node_a)

def remove_connection(node_a, node_b):
    if node_b in node_a.connections:
        node_a.connections.remove(node_b)
    if node_a in node_b.connections:
        node_b.connections.remove(node_a)

class NodeNetwork:
    def __init__(self, num_nodes=100, num_initial_connections=20, a=1.7, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        
        self.num_nodes = num_nodes
        self.a = a
        self.nodes = self.initialize_nodes()            # Create the nodes
        self.initialize_connections(num_initial_connections)    # Add connections between nodes
    
    # Initialize the nodes with random positions and activity
    def initialize_nodes(self):
        nodes = []
        for i in range(self.num_nodes):
            node = Node(node_id=i)
            nodes.append(node)
        return nodes
    
    # Initialize random connections between the nodes
    def initialize_connections(self, num_connections):        
        possible_pairs = [(i, j) for i in range(self.num_nodes) for j in range(i+1, self.num_nodes)]
        np.random.shuffle(possible_pairs)

        for i, j in possible_pairs[:num_connections]:
            add_connection(self.nodes[i], self.nodes[j])
    
    # Update the activity of all nodes
    def update_network(self):
        for node in self.nodes:
            node.update_activity(self.a)

class NetworkPlot:
    def __init__(self, ax, nodes, connections):
        self.ax = ax
        self.cmap = cm.get_cmap('cividis')  # Color map for node activity
        self.circles = []
        self.texts = []
        self.lines = []

        self.initialize_plot(nodes, connections)

    # Function to initialize the plot with node positions and connections
    def initialize_plot(self, nodes, connections):
        self.circles = []
        self.texts = []
        self.lines = []

        # Draw connections
        for i, j in connections:
            x1, y1 = nodes[i].position
            x2, y2 = nodes[j].position
            line, = self.ax.plot([x1, x2], [y1, y2], 'gray', lw=0.5, alpha=0.6)
            self.lines.append(line)

        # Draw nodes and their activities
        for node in nodes:
            x, y = node.position
            activity = node.activity
            color = self.cmap((activity + 1) / 2)
            circle = Circle((x, y), 0.02, color=color, ec='black')
            self.ax.add_patch(circle)
            self.circles.append(circle)
            formatted_activity = self.format_activity(activity)
            text = self.ax.text(x, y, formatted_activity, fontsize=7, ha='center', va='center', color='white')
            self.texts.append(text)

    # Function to update the plot during animation
    def update_plot(self, nodes, connections):
        # Update node colors and activities
        for i, (circle, text) in enumerate(zip(self.circles, self.texts)):
            activity = nodes[i].activity
            color = self.cmap((activity + 1) / 2)
            circle.set_facecolor(color)
            text.set_text(self.format_activity(activity))

        # Update connection lines
        for line, (i, j) in zip(self.lines, connections):
            x1, y1 = nodes[i].position
            x2, y2 = nodes[j].position
            line.set_data([x1, x2], [y1, y2])

    # Helper function to format activity values
    def format_activity(self, activity):
        return f'{activity: .2f}'.replace('0.', '.').replace('-0.', '-.')

# Parent class manages both the Network and its Plotter representation
class Simulation:
    def __init__(self, num_nodes=100, num_initial_connections=20, a=1.7, random_seed=None):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.network = NodeNetwork(num_nodes=num_nodes, num_initial_connections=num_initial_connections, a=a, random_seed=random_seed)
        self.plot = NetworkPlot(self.ax, self.network.nodes, self.get_connections())

    # Get list of connections as (i, j) pairs for the plotter
    def get_connections(self):
        connections = []
        for i, node in enumerate(self.network.nodes):
            for connected_node in node.connections:
                connections.append((i, connected_node.node_id))
        return connections

    # This function handles updating both the network and the plot for each frame
    def update(self, frame):
        self.network.update_network()  # Update network logic
        self.plot.update_plot(self.network.nodes, self.get_connections())  # Update the plot based on new state

    # Run the animation
    def run(self, num_steps=100):
        anim = FuncAnimation(
            self.fig, self.update, frames=num_steps,
            repeat=True, interval=250  # Interval for speed (in milliseconds)
        )
        plt.show()

# Run the simulation
if __name__ == "__main__":
    sim = Simulation(num_nodes=100, num_initial_connections=20, a=1.7, random_seed=42)
    sim.run(num_steps=100)
