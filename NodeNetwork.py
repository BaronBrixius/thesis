import cProfile
import pstats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation
from scipy.sparse.csgraph import shortest_path

NUM_NODES = 100
NUM_CONNECTIONS = 600

NUM_STEPS = 20000
DISPLAY_INTERVAL = 0

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
        self.activities = np.random.uniform(-1, 1, num_nodes)   # Random initial activity
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        self.positions = np.random.uniform(0, 1, (num_nodes, 2))

        self.initialize_connections(num_connections)    # Add initial connections
    
    # Initialize random connections between the nodes
    def initialize_connections(self, num_connections):
        possible_pairs = [(i, j) for i in range(self.num_nodes) for j in range(i+1, self.num_nodes)]
        np.random.shuffle(possible_pairs)

        for i, j in possible_pairs[:num_connections]:
            self.add_connection(i, j)

    def add_connection(self, a, b):
        self.adjacency_matrix[a, b] = self.adjacency_matrix[b, a] = 1

    def remove_connection(self, a, b):
        self.adjacency_matrix[a, b] = self.adjacency_matrix[b, a] = 0

    def rewire(self):
        # 1. Pick a unit at random (henceforth: pivot). Note that zero-connection nodes cannot be pivots
        connected_nodes = np.where(self.adjacency_matrix.sum(axis=1) > 0)[0]
        pivot = np.random.choice(connected_nodes)

        # 2. From all other units, select the one that is most synchronized (henceforth: candidate)
        activity_diff = np.abs(self.activities - self.activities[pivot])
        activity_diff[pivot] = np.inf                       # stop the pivot from connecting to itself
        candidate = np.argmin(activity_diff)                # most similar activity

        # 3a. If there is a connection between the pivot and the candidate already, do nothing
        if self.adjacency_matrix[pivot, candidate] == 1:
            return

        # 3b. If there is no connection between the pivot and the candidate, establish it, and break the connection between the pivot and its least synchronized neighbor.
        self.add_connection(pivot, candidate)

        pivot_connections = self.adjacency_matrix[pivot]
        activity_diff_connected = np.abs(self.activities - self.activities[pivot]) * pivot_connections
        least_synchronized = np.argmax(activity_diff_connected)
        self.remove_connection(pivot, least_synchronized)

    def apply_forces(self, attraction_force=0.006, repulsion_force=0.0002, min_distance=0.005, max_distance=0.25):
        forces = np.zeros((self.num_nodes, 2))  # Initialize force matrix for all nodes

        # Calculate attraction forces between connected nodes
        for i in range(self.num_nodes):
            connected_nodes = np.where(self.adjacency_matrix[i] == 1)[0]
            for j in connected_nodes:
                direction = self.positions[j] - self.positions[i]
                distance = np.linalg.norm(direction)
                if distance > min_distance:
                    forces[i] += attraction_force * direction / distance  # Normalize force

        # Calculate repulsion forces between all pairs
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                direction = self.positions[i] - self.positions[j]
                distance = np.linalg.norm(direction)
                if distance < max_distance and distance > 0:
                    force_magnitude = repulsion_force / (distance**2)
                    forces[i] += force_magnitude * direction / distance
                    forces[j] -= force_magnitude * direction / distance

        # Update positions based on total forces and keep them within bounds
        self.positions += forces
        self.positions = np.clip(self.positions, 0, 1)

    # Update the activity of all nodes
    def update_network(self):    
        # Calculate neighbor activities as a matrix multiplication of adjacency and old activities
        neighbor_sum = self.adjacency_matrix @ self.activities
        neighbor_counts = self.adjacency_matrix.sum(axis=1)
        connected_nodes = neighbor_counts > 0  # Boolean array indicating connected nodes
        neighbor_activities = np.zeros_like(self.activities)
        neighbor_activities[connected_nodes] = neighbor_sum[connected_nodes] / neighbor_counts[connected_nodes]

        # logistic map: x(n+1) = f(x(n)) = 1 - ax(n)²
        own_activities = 1 - self.alpha * self.activities**2                                    
        # xᵢ(n+1) = (1 − ε) * f(xᵢ(n)) + (ε / Mᵢ) * ∑(f(xⱼ(n) for j in B(i))
        self.activities[connected_nodes] = (1 - self.epsilon) * own_activities[connected_nodes] + self.epsilon * neighbor_activities[connected_nodes]
        # Unconnected nodes use only their own activity
        self.activities[~connected_nodes] = own_activities[~connected_nodes]  

        self.rewire()

        #self.apply_forces()

    def characteristic_path_length(self):
        path_lengths = shortest_path(self.adjacency_matrix, directed=False, unweighted=True)
        valid_lengths = path_lengths[(path_lengths != np.inf) & (path_lengths > 0)]
        return np.mean(valid_lengths)
    
    def clustering_coefficient(self):
        clustering_coefficients = []
        for i in range(self.num_nodes):
            neighbors = np.where(self.adjacency_matrix[i] == 1)[0]
            if len(neighbors) < 2:
                clustering_coefficients.append(0)
                continue
            neighbor_pairs = self.adjacency_matrix[neighbors][:, neighbors]
            connections = np.sum(neighbor_pairs) / 2  # Each edge is counted twice
            clustering_coefficients.append(connections / (len(neighbors) * (len(neighbors) - 1)))
        return np.mean(clustering_coefficients)

    def calculate_metrics(self):
        char_path_length = self.characteristic_path_length()
        avg_clustering = self.clustering_coefficient()
        return char_path_length, avg_clustering

class NetworkPlot:
    def __init__(self, positions, activities, adjacency_matrix):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.cmap = colormaps['cividis']  # Color map for node activity
        self.circles = []
        self.texts = []
        self.lines = []

        self.initialize_plot(positions, activities, adjacency_matrix)

    def initialize_plot(self, positions, activities, adjacency_matrix):
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_aspect('equal')

        # Draw nodes
        for i, (x, y) in enumerate(positions):
            color = self.cmap((activities[i] + 1) / 2)
            circle = Circle((x, y), 0.02, color=color, ec='black')
            self.ax.add_patch(circle)
            self.circles.append(circle)

            text = self.ax.text(x, y, self.format_activity(activities[i]), fontsize=7, ha='center', va='center', color='white')
            self.texts.append(text)

        # Draw connections based on adjacency matrix
        for i in range(adjacency_matrix.shape[0]):
            for j in range(i + 1, adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] == 1:
                    line, = self.ax.plot([positions[i][0], positions[j][0]], [positions[i][1], positions[j][1]], 'gray', lw=0.5, alpha=0.6)
                    self.lines.append(line)

    def update_plot(self, positions, activities, adjacency_matrix, step, characteristic_path_length, clustering_coefficient):
        self.ax.set_title(f"Generation {step} - CPL: {characteristic_path_length:.2f}, CC: {clustering_coefficient:.2f}")

        # Update node colors, positions, and text values
        for i, (circle, text) in enumerate(zip(self.circles, self.texts)):
            color = self.cmap((activities[i] + 1) / 2)
            circle.set_facecolor(color)
            circle.set_center(positions[i])

            text.set_text(self.format_activity(activities[i]))
            text.set_position(positions[i])

        # Update connection lines
        line_index = 0
        for i in range(adjacency_matrix.shape[0]):
            for j in range(i + 1, adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] == 1:
                    self.lines[line_index].set_data([positions[i][0], positions[j][0]], [positions[i][1], positions[j][1]])
                    line_index += 1

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()   # Flush GUI events for immediate update without delay

    # Helper function to format activity values
    def format_activity(self, activity):
        return f'{activity: .2f}'.replace('0.', '.').replace('-0.', '-.')

# Parent class manages both the Network and its Plotter representation
class Simulation:
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, random_seed=None):
        self.network = NodeNetwork(num_nodes=num_nodes, num_connections=num_connections, alpha=alpha, epsilon=epsilon, random_seed=random_seed)
        #self.plot = NetworkPlot(self.network.positions, self.network.activities, self.network.adjacency_matrix)

    def run(self, num_steps, display_interval):
        # Turn on plotting in interactive mode so it updates
        #plt.ion()
        #plt.show()
        for step in range(num_steps):
            self.network.update_network()
            #if (display_interval and step % display_interval == 0) or step == num_steps - 1:
            #    characteristic_path_length, clustering_coefficient = self.network.calculate_metrics()
            #    print(f"Iteration {step}: CPL={characteristic_path_length:.2f}, Clustering={clustering_coefficient:.2f}")
                #self.plot.update_plot(self.network.positions, self.network.activities, self.network.adjacency_matrix, step, characteristic_path_length, clustering_coefficient)
        
        # Turn off interactive mode to display the plot at the very end without it closing
        #plt.ioff()
        #plt.show()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    # Run the simulation
    sim = Simulation(num_nodes=NUM_NODES, num_connections=NUM_CONNECTIONS, alpha=ALPHA, epsilon=EPSILON, random_seed=RANDOM_SEED)
    sim.run(num_steps=NUM_STEPS, display_interval=DISPLAY_INTERVAL)

    profiler.disable()

    # Print profiler stats to sort by cumulative time
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)