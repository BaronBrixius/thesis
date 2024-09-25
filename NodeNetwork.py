import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation

NUM_NODES = 100
NUM_CONNECTIONS = 50
ALPHA = 1.7
EPSILON = 0.4

NUM_STEPS = 10000
TIME_INTERVAL = 10  # Interval between steps (in milliseconds)
RANDOM_SEED = 44

class NodeNetwork:
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.epsilon = epsilon
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

        # try normal and reversed order, since we don't know which node was considered node_a at the the time of insertion
        if (node_a.node_id, node_b.node_id) in self.connections:
            self.connections.remove((node_a.node_id, node_b.node_id))
        else:
            self.connections.remove((node_b.node_id, node_a.node_id))


    def rewire(self):
        # 1. Pick a unit at random (henceforth: pivot)
        pivot = np.random.choice(self.nodes)

        # 2. From all other units, select the one that is most synchronized (henceforth: candidate
        other_nodes = [node for node in self.nodes if node != pivot]
        candidate = min(other_nodes, key=lambda node: abs(pivot.activity - node.activity))

        # 3a. If there is a connection between the pivot and the candidate already, do nothing
        if candidate in pivot.connections:
            return

        # 3b. If there is no connection between the pivot and the candidate, establish it, and break the connection between the pivot and its least synchronized neighbor.
        # TODO is it possible to break the connection we just created? Maybe let's only look at pivots that already have connections?
        self.add_connection(pivot, candidate)
        least_synchronized = max(pivot.connections, key=lambda node: abs(pivot.activity - node.activity))
        self.remove_connection(pivot, least_synchronized)

    # Update the activity of all nodes
    def update_network(self):
        for node in self.nodes:
            node.save_old_activity()
        
        for node in self.nodes:
            node.update_activity(self.alpha, self.epsilon)

        # TODO Initial transient time, means that rewiring only begins after some number of iterations?
        self.rewire()


    class Node:
        def __init__(self, node_id):
            self.node_id = node_id
            self.position = np.random.uniform(0, 1, 2)  # Random x, y position
            self.activity = np.random.uniform(-1, 1)    # Random initial activity
            self.old_activity = self.activity
            self.connections = []

        def update_activity(self, alpha=1.7, epsilon=0.4):
            own_activity = 1 - alpha * self.old_activity**2    # logistic map: x(n+1) = f(x(n)) = 1 - ax(n)²

            if self.connections:
                neighbor_activity = np.mean([node.old_activity for node in self.connections])
                self.activity = ((1 - epsilon) * own_activity) + (epsilon * neighbor_activity) # xᵢ(n+1) = (1 − ε) * f(xᵢ(n)) + (ε / Mᵢ) * ∑(f(xⱼ(n) for j in B(i))
            else:
                self.activity = own_activity
    
        def save_old_activity(self):
            self.old_activity = self.activity

class NetworkPlot:
    def __init__(self, ax, nodes, connections):
        self.ax = ax
        self.cmap = colormaps['cividis']  # Color map for node activity
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
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, random_seed=None):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.network = NodeNetwork(num_nodes=num_nodes, num_connections=num_connections, alpha=alpha, epsilon=epsilon, random_seed=random_seed)
        self.plot = NetworkPlot(self.ax, self.network.nodes, self.network.connections)

    # Update both the network and the plot for each frame
    def update(self, frame):
        self.network.update_network()  # Update network logic
        self.plot.update_plot(self.network.nodes, self.network.connections)  # Update the plot based on new state

    # Run the animation
    def run(self, num_steps, interval):
        anim = FuncAnimation(
            self.fig, self.update, frames=num_steps,
            repeat=True, interval=interval
        )
        plt.show()

# Run the simulation
if __name__ == "__main__":
    sim = Simulation(num_nodes=NUM_NODES, num_connections=NUM_CONNECTIONS, alpha=ALPHA, epsilon=EPSILON, random_seed=RANDOM_SEED)
    sim.run(num_steps=NUM_STEPS, interval=TIME_INTERVAL)
