import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm
from matplotlib.animation import FuncAnimation

NUM_NODES = 100
NUM_CONNECTIONS = 20
ALPHA = 1.7

NUM_STEPS = 100
RANDOM_SEED = 42

class NodeNetwork:
    def __init__(self, num_nodes, num_connections, a=1.7, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        
        self.num_nodes = num_nodes
        self.a = a
        self.nodes = self.initialize_nodes()            # Create the nodes
        self.connections = []                           # List to hold all connections (i, j)
        self.initialize_connections(num_connections)    # Add initial connections
    
    # Initialize the nodes with random positions and activity
    def initialize_nodes(self):
        return [self.Node(node_id=i) for i in range(self.num_nodes)]
    
    # Initialize random connections between the nodes
    def initialize_connections(self, num_connections):
        possible_pairs = [(i, j) for i in range(self.num_nodes) for j in range(i+1, self.num_nodes)]
        np.random.shuffle(possible_pairs)

        for i, j in possible_pairs[:num_connections]:
            self.add_connection(self.nodes[i], self.nodes[j])

    def add_connection(self, node_a, node_b):
        node_a.connections.append(node_b)
        node_b.connections.append(node_a)
        self.connections.append((node_a.node_id, node_b.node_id))

    def remove_connection(self, node_a, node_b):
        node_a.connections.remove(node_b)
        node_b.connections.remove(node_a)
        self.connections.remove((node_a.node_id, node_b.node_id))

    # Update the activity of all nodes
    def update_network(self):
        for node in self.nodes:
            node.update_activity(self.a)

    class Node:
        def __init__(self, node_id):
            self.node_id = node_id
            self.position = np.random.uniform(0, 1, 2)  # Random x, y position
            self.activity = np.random.uniform(-1, 1)    # Random initial activity
            self.connections = []

        def update_activity(self, a=1.7):
            self.activity = 1 - a * self.activity**2    # logistic map: x(n+1) = f(x(n)) = 1 - ax(n)²

class NetworkPlot:
    def __init__(self, ax, nodes, connections):
        self.ax = ax
        self.cmap = cm.get_cmap('cividis')  # Color map for node activity
        self.circles = []
        self.texts = []
        self.lines = []

        self.initialize_plot(nodes, connections)

    def initialize_plot(self, nodes, connections):
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

        # Draw connections
        for i, j in connections:
            x1, y1 = nodes[i].position
            x2, y2 = nodes[j].position
            line, = self.ax.plot([x1, x2], [y1, y2], 'gray', lw=0.5, alpha=0.6)
            self.lines.append(line)

    # Update the plot each frame
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
    def __init__(self, num_nodes, num_connections, a=1.7, random_seed=None):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.network = NodeNetwork(num_nodes=num_nodes, num_connections=num_connections, a=a, random_seed=random_seed)
        self.plot = NetworkPlot(self.ax, self.network.nodes, self.network.connections)

    # Update both the network and the plot for each frame
    def update(self, frame):
        self.network.update_network()  # Update network logic
        self.plot.update_plot(self.network.nodes, self.network.connections)  # Update the plot based on new state

    # Run the animation
    def run(self, num_steps):
        anim = FuncAnimation(
            self.fig, self.update, frames=num_steps,
            repeat=True, interval=250  # Interval for speed (in milliseconds)
        )
        plt.show()

# Run the simulation
if __name__ == "__main__":
    sim = Simulation(num_nodes=NUM_NODES, num_connections=NUM_CONNECTIONS, a=ALPHA, random_seed=RANDOM_SEED)
    sim.run(num_steps=NUM_STEPS)
