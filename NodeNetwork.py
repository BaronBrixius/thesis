import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation
from scipy.sparse.csgraph import shortest_path

NUM_NODES = 100
NUM_CONNECTIONS = 600

NUM_STEPS = 10000000000
DISPLAY_INTERVAL = 100

ALPHA = 1.7
EPSILON = 0.4

NODE_ATTRACTION_FORCE = 0.006
NODE_REPULSION_FORCE = 0.0002
MIN_NODE_DISTANCE = 0.005
REPULSION_MAX_DISTANCE = 0.25

RANDOM_SEED = 42

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
        # 1. Pick a unit at random (henceforth: pivot). Note that zero-connection nodes cannot be pivots
        connected_nodes = [node for node in self.nodes if node.connections]
        pivot = np.random.choice(connected_nodes)

        # 2. From all other units, select the one that is most synchronized (henceforth: candidate
        other_nodes = [node for node in self.nodes if node != pivot]
        candidate = min(other_nodes, key=lambda node: abs(pivot.activity - node.activity))

        # 3a. If there is a connection between the pivot and the candidate already, do nothing
        if candidate in pivot.connections:
            return

        # 3b. If there is no connection between the pivot and the candidate, establish it, and break the connection between the pivot and its least synchronized neighbor.
        self.add_connection(pivot, candidate)
        least_synchronized = max(pivot.connections, key=lambda node: abs(pivot.activity - node.activity))
        self.remove_connection(pivot, least_synchronized)

    def apply_forces(self):
        for node in self.nodes:
            force = np.array([0.0, 0.0])

            # Attraction force: Pull nodes closer to connected neighbors
            for connected_node in node.connections:
                direction = connected_node.position - node.position
                distance = np.linalg.norm(direction)
                if MIN_NODE_DISTANCE * 3 < distance:
                    force += NODE_ATTRACTION_FORCE * direction / distance  # Normalize to create proportional force

            # Repulsion force: Push nodes away from all other nodes
            for other_node in self.nodes:
                if other_node != node:
                    direction = node.position - other_node.position
                    distance = np.linalg.norm(direction)
                    if distance == 0:
                        force += np.random.uniform(-0.01, 0.01, 2)  # Small random nudge
                        continue
                    
                    if MIN_NODE_DISTANCE * 2 < distance < REPULSION_MAX_DISTANCE:
                        force += NODE_REPULSION_FORCE * direction / (distance ** 2)  # Inverse square repulsion
                    elif distance < MIN_NODE_DISTANCE:                                 
                        force += 3 * NODE_REPULSION_FORCE * direction / (distance ** 2)    # Very hard repulsion if nodes are overlapping
                    
            force *= 0.7

            # Update node position based on the total force
            node.position += force

            # Ensure nodes stay within the bounds (0, 1) on the grid
            node.position = np.clip(node.position, 0, 1)

    # Update the activity of all nodes
    def update_network(self):
        for node in self.nodes:
            node.save_old_activity()
        
        for node in self.nodes:
            node.update_activity(self.alpha, self.epsilon)

        # We don't do an initial transient time

        self.rewire()

        #self.apply_forces()

    # Calculate Characteristic Path Length and Clustering Coefficient
    def calculate_metrics(self):
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i, j in self.connections:
            adj_matrix[i, j] = adj_matrix[j, i] = 1
        path_lengths = shortest_path(adj_matrix, directed=False, unweighted=True)
        char_path_length = np.mean(path_lengths[path_lengths != np.inf])
        clustering_coefficients = []
        for node in self.nodes:
            neighbors = node.connections
            if len(neighbors) < 2:
                clustering_coefficients.append(0)
                continue
            connections = sum(1 for n1 in neighbors for n2 in neighbors if n2 in n1.connections)
            clustering_coefficients.append(connections / (len(neighbors) * (len(neighbors) - 1)))
        avg_clustering = np.mean(clustering_coefficients)
        return char_path_length, avg_clustering

    class Node:
        def __init__(self, node_id):
            self.node_id = node_id
            self.position = np.random.uniform(0, 1, 2)  # Random x, y position
            self.activity = np.random.uniform(-1, 1)    # Random initial activity
            self.old_activity = self.activity
            self.connections = []

        def update_activity(self, alpha=1.7, epsilon=0.4):
            own_activity = 1 - alpha * self.old_activity**2     # logistic map: x(n+1) = f(x(n)) = 1 - ax(n)²
            if self.connections:                                # influences by connected nodes
                neighbor_activity = np.mean([node.old_activity for node in self.connections])
                self.activity = ((1 - epsilon) * own_activity) + (epsilon * neighbor_activity) # xᵢ(n+1) = (1 − ε) * f(xᵢ(n)) + (ε / Mᵢ) * ∑(f(xⱼ(n) for j in B(i))
            else:
                self.activity = own_activity
    
        def save_old_activity(self):
            self.old_activity = self.activity

class NetworkPlot:
    def __init__(self, nodes, connections):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.cmap = colormaps['cividis']  # Color map for node activity
        self.circles = []
        self.texts = []
        self.lines = []

        self.initialize_plot(nodes, connections)

    def initialize_plot(self, nodes, connections):
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_aspect('equal')

        # Draw nodes
        for node in nodes:
            circle = Circle((0, 0), 0.02, ec='black')
            self.ax.add_patch(circle)
            self.circles.append(circle)

            text = self.ax.text(0, 0, '', fontsize=7, ha='center', va='center', color='white')
            self.texts.append(text)

        # Draw connections
        for i, j in connections:
            line, = self.ax.plot([], [], 'gray', lw=0.5, alpha=0.6)
            self.lines.append(line)

    # Update the plot each frame
    def update_plot(self, nodes, connections, step, characteristic_path_length, clustering_coefficient):
        self.ax.set_title(f"Generation {step} - CPL: {characteristic_path_length:.2f}, CC: {clustering_coefficient:.2f}")
        # Update node colors, positions, and text values
        for i, (circle, text) in enumerate(zip(self.circles, self.texts)):
            activity = nodes[i].activity
            color = self.cmap((activity + 1) / 2)
            circle.set_facecolor(color)
            
            circle.set_center(nodes[i].position)

            text.set_text(self.format_activity(activity))
            text.set_position(nodes[i].position)

        # Update connection lines
        for line, (i, j) in zip(self.lines, connections):
            x1, y1 = nodes[i].position
            x2, y2 = nodes[j].position
            line.set_data([x1, x2], [y1, y2])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()   # Flush GUI events for immediate update without delay

    # Helper function to format activity values
    def format_activity(self, activity):
        return f'{activity: .2f}'.replace('0.', '.').replace('-0.', '-.')

# Parent class manages both the Network and its Plotter representation
class Simulation:
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, random_seed=None):
        self.network = NodeNetwork(num_nodes=num_nodes, num_connections=num_connections, alpha=alpha, epsilon=epsilon, random_seed=random_seed)
        self.plot = NetworkPlot(self.network.nodes, self.network.connections)

    def run(self, num_steps, display_interval):
        # Turn on plotting in interactive mode so it updates
        plt.ion()
        plt.show()
        for step in range(num_steps):
            self.network.update_network()
            if step % display_interval == 0 or step == num_steps - 1:
                characteristic_path_length, clustering_coefficient = self.network.calculate_metrics()
                print(f"Iteration {step}: CPL={characteristic_path_length:.2f}, Clustering={clustering_coefficient:.2f}")
                self.plot.update_plot(self.network.nodes, self.network.connections, step, characteristic_path_length, clustering_coefficient)
        
        # Turn off interactive mode to display the plot at the very end without it closing
        plt.ioff()
        plt.show()

# Run the simulation
if __name__ == "__main__":
    sim = Simulation(num_nodes=NUM_NODES, num_connections=NUM_CONNECTIONS, alpha=ALPHA, epsilon=EPSILON, random_seed=RANDOM_SEED)
    sim.run(num_steps=NUM_STEPS, display_interval=DISPLAY_INTERVAL)