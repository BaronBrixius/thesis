import cProfile
import pstats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation
from scipy.sparse.csgraph import shortest_path
import time

NUM_NODES = 50
CONNECTION_DENSITY = 0.1
NUM_CONNECTIONS = int(CONNECTION_DENSITY * (NUM_NODES * (NUM_NODES - 1) / 2)) # * total possible connections n*(n-1)/2
NUM_STEPS = 2_000_000

METRICS_INTERVAL = 1000
DISPLAY_INTERVAL = 1000
STABILIZATION_THRESHOLD = 0.0

NORMAL_DISTANCE_SCALING = 0.7  # You may need to adjust this for screen scaling
SCREEN_WIDTH = 2.0
SCREEN_HEIGHT = 2.0

ALPHA = 1.7
EPSILON = 0.4
RANDOM_SEED = 42

accumulated_times = {}
def start_timing(label):
    if label not in accumulated_times:
        accumulated_times[label] = {'start_time': 0, 'total_time': 0}
    accumulated_times[label]['start_time'] = time.perf_counter()

def stop_timing(label):
    if label in accumulated_times and accumulated_times[label]['start_time'] != 0:
        end_time = time.perf_counter()
        elapsed_time = end_time - accumulated_times[label]['start_time']
        accumulated_times[label]['total_time'] += elapsed_time
        accumulated_times[label]['start_time'] = 0  # Reset start time

class NodeNetwork:
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.epsilon = epsilon

        self.activities = np.random.uniform(-1, 1, num_nodes)   # Random initial activity
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        self.positions = np.random.uniform([0.2 * SCREEN_WIDTH, 0.2 * SCREEN_HEIGHT], [0.8 * SCREEN_WIDTH, 0.8 * SCREEN_HEIGHT], (num_nodes, 2))
        
        self.initialize_connections(num_connections)    # Add initial connections

        self.cpl_history = []
        self.breakup_count = 0
        self.cc_history = []
        self.stabilized = False # Relatable

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

    def update_activity(self):
        # Calculate neighbor activities as a matrix multiplication of adjacency and activities, then row-wise summing
        neighbor_sum = np.einsum('ij,j->i', self.adjacency_matrix, self.activities) # maybe faster than self.adjacency_matrix @ self.activities
        neighbor_counts = self.adjacency_matrix.sum(axis=1)
        connected_nodes = neighbor_counts > 0  # Boolean array indicating connected nodes

        # xᵢ(n+1) = (1 − ε) * f(xᵢ(n)) + (ε / Mᵢ) * ∑(f(xⱼ(n) for j in B(i))
        self.activities[connected_nodes] = (
            (1 - self.epsilon) * self.activities[connected_nodes]
            + self.epsilon * neighbor_sum[connected_nodes] / neighbor_counts[connected_nodes]
        )
        # logistic map: x(n+1) = f(x(n)) = 1 - ax(n)²
        self.activities = 1 - self.alpha * self.activities**2

    def rewire(self):
        # 1. Pick a unit at random (henceforth: pivot)
        pivot = np.random.randint(self.num_nodes)
        while not np.any(self.adjacency_matrix[pivot]): # zero-connection nodes cannot be pivots
            pivot = np.random.randint(self.num_nodes)

        # 2. From all other units, select the one that is most synchronized (henceforth: candidate) and least synchronized neighbor
        # TODO optimize with a loop to look for both at once?
        activity_diff = np.abs(self.activities - self.activities[pivot])
        activity_diff_neighbors = activity_diff * self.adjacency_matrix[pivot]

        activity_diff[pivot] = np.inf                       # stop the pivot from connecting to itself
        candidate = np.argmin(activity_diff)                # most similar activity
        least_synchronized_neighbor = np.argmax(activity_diff_neighbors)    # least similar neighbor

        # 3a. If there is a connection between the pivot and the candidate already, do nothing
        if self.adjacency_matrix[pivot, candidate] == 1:
            return

        # 3b. If there is no connection between the pivot and the candidate, establish it, and break the connection between the pivot and its least synchronized neighbor.
        self.add_connection(pivot, candidate)
        self.remove_connection(pivot, least_synchronized_neighbor)

    # Update the activity of all nodes
    def update_network(self):
        self.update_activity()
        self.rewire()

    def characteristic_path_length(self):
        path_lengths = shortest_path(self.adjacency_matrix, directed=False, unweighted=True)
        if np.isinf(path_lengths).any():
            self.breakup_count += 1
        valid_lengths = path_lengths[(path_lengths < np.inf) & (path_lengths > 0)]  # FIXME right now, upon breakup, it removes "infinite" distances then computes the average as if that were okay
        return np.mean(valid_lengths)
    
    def clustering_coefficient(self):
        clustering_coefficients = []
        for i in range(self.num_nodes):
            neighbors = np.where(self.adjacency_matrix[i] == 1)[0]
            if len(neighbors) < 2:
                clustering_coefficients.append(0)
                continue
            neighbor_pairs = self.adjacency_matrix[neighbors][:, neighbors]
            connections = np.sum(neighbor_pairs)
            possible_connections = len(neighbors) * (len(neighbors) - 1)
            clustering_coefficients.append(connections / possible_connections)
        return np.mean(clustering_coefficients)

    def calculate_stats(self):
        char_path_length = self.characteristic_path_length()
        avg_clustering = self.clustering_coefficient()

        if not STABILIZATION_THRESHOLD:
            return char_path_length, avg_clustering

        # Add the new CPL and CC values to history
        self.cpl_history.append(char_path_length)
        self.cc_history.append(avg_clustering)

        if len(self.cpl_history) > 100:
            self.cpl_history.pop(0)
        if len(self.cc_history) > 100:
            self.cc_history.pop(0)

        # Check stabilization for both CPL and CC
        if len(self.cpl_history) == 100 and len(self.cc_history) == 100:
            cpl_min, cpl_max = min(self.cpl_history), max(self.cpl_history)
            cpl_stable = (cpl_max - cpl_min) / cpl_max <= STABILIZATION_THRESHOLD

            cc_min, cc_max = min(self.cc_history), max(self.cc_history)
            cc_stable = (cc_max - cc_min) / cc_max <= STABILIZATION_THRESHOLD

            # If both CPL and CC are stable, mark the network as stabilized
            self.stabilized = cpl_stable and cc_stable

        return char_path_length, avg_clustering

    def apply_forces(self):
        # Initialize force matrix
        forces = np.zeros((self.num_nodes, 2))

        # Calculate normal distance based on screen size and number of nodes
        normal_distance = NORMAL_DISTANCE_SCALING * np.sqrt(SCREEN_WIDTH * SCREEN_HEIGHT / self.num_nodes)

        # Iterate over each pair of nodes
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                # Compute the vector from node i to node j and the distance
                direction = self.positions[j] - self.positions[i]
                distance = np.linalg.norm(direction)
                
                if distance == 0:  # Avoid division by zero
                    continue
                
                normalized_direction = direction / distance
                
                if self.adjacency_matrix[i, j] == 1:  # If nodes are connected
                    # Apply attraction/repulsion based on the target range for connected nodes
                    if distance < 0.2 * normal_distance:
                        forces[i] -= normalized_direction * (0.2 * normal_distance - distance)
                        forces[j] += normalized_direction * (0.2 * normal_distance - distance)
                    elif distance > 0.3 * normal_distance:
                        forces[i] += normalized_direction * (distance - 0.3 * normal_distance)
                        forces[j] -= normalized_direction * (distance - 0.3 * normal_distance)
                else:
                    # If nodes are not connected, apply repulsion if they're within a certain distance
                    if distance < 1.7 * normal_distance:
                        repulsion_force = (1.7 * normal_distance - distance) / distance
                        forces[i] -= normalized_direction * repulsion_force
                        forces[j] += normalized_direction * repulsion_force

        # Update positions based on the forces
        self.positions += forces * 0.01  # Adjust the multiplier for movement speed

        # Clip positions to keep nodes within screen bounds
        self.positions = np.clip(self.positions, 0, [SCREEN_WIDTH, SCREEN_HEIGHT])


class NetworkPlot:
    def __init__(self, positions, activities, adjacency_matrix):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.cmap = colormaps['cividis']  # Color map for node activity
        self.circles = []
        #self.texts = []
        self.lines = []

        self.initialize_plot(positions, activities, adjacency_matrix)

    def initialize_plot(self, positions, activities, adjacency_matrix):
        self.ax.set_xlim(0, SCREEN_WIDTH)
        self.ax.set_ylim(0, SCREEN_HEIGHT)
        self.ax.set_aspect('equal')

        # Draw connections based on adjacency matrix
        for i in range(adjacency_matrix.shape[0]):
            for j in range(i + 1, adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] == 1:
                    line, = self.ax.plot([positions[i][0], positions[j][0]], [positions[i][1], positions[j][1]], 'gray', lw=0.5, alpha=0.6)
                    self.lines.append(line)

        # Draw nodes
        for i, (x, y) in enumerate(positions):
            color = self.cmap((activities[i] + 1) / 2)
            circle = Circle((x, y), 0.02, color=color, ec='black', zorder=2)
            self.ax.add_patch(circle)
            self.circles.append(circle)

            #text = self.ax.text(x, y, self.format_activity(activities[i]), fontsize=7, ha='center', va='center', color='white')
            #self.texts.append(text)


    def update_plot(self, positions, activities, adjacency_matrix, step, characteristic_path_length, clustering_coefficient):
        self.ax.set_title(f"Generation {step}")# - CPL: {characteristic_path_length:.2f}, CC: {clustering_coefficient:.2f}") #TODO

        # Update connection lines
        line_index = 0
        for i in range(adjacency_matrix.shape[0]):
            for j in range(i + 1, adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] == 1:
                    self.lines[line_index].set_data([positions[i][0], positions[j][0]], [positions[i][1], positions[j][1]])
                    line_index += 1

        # Update node colors, positions, and text values
        for i, circle in enumerate(self.circles):               # for i, (circle, text) in enumerate(zip(self.circles, self.texts)):
            color = self.cmap((activities[i] + 1) / 2)
            circle.set_facecolor(color)
            circle.set_center(positions[i])

            #text.set_text(self.format_activity(activities[i]))
            #text.set_position(positions[i])


        self.fig.canvas.draw()
        self.fig.canvas.flush_events()   # Flush GUI events for immediate update without delay

    # Helper function to format activity values
    def format_activity(self, activity):
        return f'{activity: .2f}'.replace('0.', '.').replace('-0.', '-.')

class Simulation:
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, random_seed=None):
        self.network = NodeNetwork(num_nodes=num_nodes, num_connections=num_connections, alpha=alpha, epsilon=epsilon, random_seed=random_seed)

    def run(self, num_steps, display_interval, metrics_interval=METRICS_INTERVAL):
        if display_interval:
            self.plot = NetworkPlot(self.network.positions, self.network.activities, self.network.adjacency_matrix)
            # Turn on plotting in interactive mode so it updates
            plt.ion()
            plt.show()
        for step in range(num_steps):
            if self.network.stabilized == True:
                print(f"Stabilized after {step} iterations.")
                break
            self.network.update_network()
            if display_interval:
                self.network.apply_forces()

            if step % metrics_interval == 0:
                characteristic_path_length, clustering_coefficient = self.network.calculate_stats()
                print(f"Iteration {step}: CPL={characteristic_path_length}, CC={clustering_coefficient}, Breakups={self.network.breakup_count}")

            if display_interval and step % display_interval == 0:
                self.plot.update_plot(self.network.positions, self.network.activities, self.network.adjacency_matrix, step, characteristic_path_length, clustering_coefficient)

        characteristic_path_length, clustering_coefficient = self.network.calculate_stats()
        print(f"Iteration {step}: CPL={characteristic_path_length}, CC={clustering_coefficient}, Breakups={self.network.breakup_count}")
        
        if display_interval:
            self.network.apply_forces()
            self.plot.update_plot(self.network.positions, self.network.activities, self.network.adjacency_matrix, step, characteristic_path_length, clustering_coefficient)
            # Turn off interactive mode to display the plot at the very end without it closing
            #plt.ioff()
            #plt.show()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    if profiler: profiler.enable()

    # Run the simulation
    print("Nodes:", NUM_NODES, "Connections:", NUM_CONNECTIONS, "Steps:", NUM_STEPS)
    sim = Simulation(num_nodes=NUM_NODES, num_connections=NUM_CONNECTIONS, alpha=ALPHA, epsilon=EPSILON, random_seed=RANDOM_SEED)
    sim.run(num_steps=NUM_STEPS, display_interval=DISPLAY_INTERVAL)

    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)

    for label, times in accumulated_times.items():
        print(f"{label}: {times['total_time']:.4f} seconds")