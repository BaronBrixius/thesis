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

# Draw connections between nodes
def draw_connections(network, ax):
    for node in network.nodes:
        for connected_node in node.connections:
            x1, y1 = node.position
            x2, y2 = connected_node.position
            ax.plot([x1, x2], [y1, y2], 'gray', lw=0.5, alpha=0.6)

# Function to initialize the circles and text labels on the plot
def initialize_plot(ax, network, cmap):
    circles = []
    texts = []

    draw_connections(network, ax)
    
    # Create the circles and text labels only once
    for node in network.nodes:
        x, y = node.position
        activity = node.activity
        
        # Map activity to a color (-1 to 1 normalized range)
        color = cmap((activity + 1) / 2)
        
        # Create a circle for each node
        circle = Circle((x, y), 0.02, color=color, ec='black')
        circles.append(circle)
        ax.add_patch(circle)
        
        # Add text inside the node
        formatted_activity = format_activity(activity)
        text = ax.text(x, y, formatted_activity, fontsize=7, ha='center', va='center', color='white')
        texts.append(text)
        
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    return circles, texts

# Step function, updates the nodes and graph each step
def step(frame, network, circles, texts, cmap):
    network.update_network()  # Update the network activity
    
    # Update the color and text for each node
    for i, (circle, text) in enumerate(zip(circles, texts)):
        activity = network.nodes[i].activity
        color = cmap((activity + 1) / 2)
        circle.set_facecolor(color)                 # Update the color of the circle
        text.set_text(format_activity(activity))    # Update the activity value

def format_activity(activity):
    return f'{activity: .2f}'.replace('0.', '.').replace('-0.', '-.')

def run_simulation(num_nodes=100, num_initial_connections=20, num_steps=100, a=1.7, random_seed=None):
    network = NodeNetwork(num_nodes=num_nodes, num_initial_connections=num_initial_connections, a=a, random_seed=random_seed)

    fig, ax = plt.subplots(figsize=(8, 8))
    color_map = cm.get_cmap('cividis')

    # Initialize the plot
    circles, texts = initialize_plot(ax, network, color_map)
    
    # Create the animation, updating the colors and texts for each frame
    anim = FuncAnimation(
        fig, step, frames=num_steps, fargs=(network, circles, texts, color_map),
        repeat=True, interval=250  # Interval for speed (in milliseconds)
    )
    
    plt.show()

if __name__ == "__main__":
    run_simulation(num_nodes=100, num_initial_connections=20, num_steps=100, a=1.7, random_seed=42)
