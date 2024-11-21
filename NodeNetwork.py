import cProfile
import pstats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse.csgraph import shortest_path
import time
import os

NUM_NODES = 500
CONNECTION_DENSITY = 0.1
NUM_CONNECTIONS = int(CONNECTION_DENSITY * (NUM_NODES * (NUM_NODES - 1) / 2)) # * total possible connections n*(n-1)/2
NUM_STEPS = 1_00_000

METRICS_INTERVAL = 1000
DISPLAY_INTERVAL = 1000
DRAW_LINES = True
STABILIZATION_THRESHOLD = 0

NORMAL_DISTANCE_SCALING = 0.5
SCREEN_WIDTH = 1.0
SCREEN_HEIGHT = 1.0

OUTPUT_DIR = "network_results"

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
        np.random.seed(random_seed)
        
        self.num_nodes = num_nodes
        self.num_connections = num_connections
        self.alpha = alpha
        self.epsilon = epsilon

        self.activities = np.random.uniform(-1, 1, num_nodes)   # Random initial activity
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        positions = np.random.uniform([0.1 * SCREEN_WIDTH, 0.1 * SCREEN_HEIGHT], [0.9 * SCREEN_WIDTH, 0.9 * SCREEN_HEIGHT], (num_nodes, 2))
        normal_distance = NORMAL_DISTANCE_SCALING * np.sqrt(SCREEN_WIDTH * SCREEN_HEIGHT / self.num_nodes) * np.sqrt(1 + self.num_connections / self.num_nodes)
        self.physics = NetworkPhysics(self.adjacency_matrix, positions, normal_distance)

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

    def apply_forces(self, effective_iterations=1):
        self.physics.apply_forces(self.adjacency_matrix, effective_iterations)

class NetworkPhysics:
    def __init__(self, adjacency_matrix, positions, normal_distance):
        self.positions = positions
        self.adjacency_matrix = adjacency_matrix
        self.normal_distance = normal_distance

    def adjust_normal_distance(self, target_coverage=0.7, tolerance=0.05, adjustment_rate=0.015):
        lower_bounds = np.percentile(self.positions, 1, axis=0)
        upper_bounds = np.percentile(self.positions, 99, axis=0)
        width, height = upper_bounds - lower_bounds

        network_area = width * height
        total_area = SCREEN_WIDTH * SCREEN_HEIGHT
        area_coverage = network_area / total_area

        # Adjust normal_distance based on coverage
        if area_coverage > target_coverage + tolerance:
            self.normal_distance *= (1 - adjustment_rate)  # Reduce normal_distance to shrink the network
        elif area_coverage < target_coverage - tolerance:
            self.normal_distance *= (1 + adjustment_rate)  # Increase normal_distance to expand the network

    def apply_forces(self, adjacency_matrix, effective_iterations=1, central_force_strength=0.0002):
        self.adjust_normal_distance()

        for _ in range(effective_iterations):
            diffs = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
            distances = np.sqrt(np.einsum('ijk,ijk->ij', diffs, diffs))
            normalized_directions = np.divide(
                diffs, distances[:, :, np.newaxis], where=distances[:, :, np.newaxis] > 0
            )

            # Identify connected pairs
            connected = adjacency_matrix == 1

            # Attraction/repulsion masks for connected nodes
            too_close = connected & (distances < 0.2 * self.normal_distance)
            too_far = connected & (distances > 0.3 * self.normal_distance)

            # Calculate forces for close and far connected nodes
            close_force = (0.2 * self.normal_distance - distances) * too_close
            far_force = (distances - 0.3 * self.normal_distance) * too_far

            # Repulsion for non-connected nodes
            within_range = ~connected & (distances < 1.7 * self.normal_distance)
            repulsion_force = within_range * np.divide((1.7 * self.normal_distance - distances), distances, where=distances > 0)

            # Apply forces
            forces = np.einsum('ijk,ij->ik', normalized_directions, close_force - far_force + repulsion_force)

            # Update positions based on forces
            self.positions += forces * 0.003  # Adjust the multiplier for movement speed
        self.pull_all_nodes_towards_center(central_force_strength)
        self.positions = np.clip(self.positions, 0, [SCREEN_WIDTH, SCREEN_HEIGHT])

    def pull_all_nodes_towards_center(self, central_force_strength):
        center = np.array([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2])
        diffs = center - self.positions
        self.positions += diffs * (central_force_strength / self.normal_distance)

class NetworkPlot:
    def __init__(self, positions, activities, adjacency_matrix, draw_lines=DRAW_LINES):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.initialize_plot(positions, activities, adjacency_matrix, draw_lines=draw_lines)

    def compute_lines(self, positions, adjacency_matrix):
        rows, cols = np.where(np.triu(adjacency_matrix, 1) == 1)
        connections = np.array([[positions[i], positions[j]] for i, j in zip(rows, cols)])
        return connections

    def initialize_plot(self, positions, activities, adjacency_matrix, draw_lines=DRAW_LINES):
        self.ax.set_xlim(0, SCREEN_WIDTH)
        self.ax.set_ylim(0, SCREEN_HEIGHT)
        self.ax.set_aspect('equal')

        # Initialize scatter plot for nodes
        self.scatter = self.ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=activities,
            cmap='cividis',
            s=10,
            zorder=2
        )

        # Initialize lines (connections)
        if draw_lines:
            lines = self.compute_lines(positions, adjacency_matrix)
            if len(lines) > 0:
                self.line_collection = LineCollection(lines, colors='gray', linewidths=0.5, alpha=0.6, zorder=1)
                self.ax.add_collection(self.line_collection)

    def update_plot(self, positions, activities, adjacency_matrix, title, draw_lines=DRAW_LINES):
        self.ax.set_title(title)

        # Check for invalid values
        if not np.all(np.isfinite(positions)):
            print("Error: Invalid positions detected (NaN or inf).")
            return

        if not np.all(np.isfinite(activities)):
            print("Error: Invalid activities detected (NaN or inf).")
            return

        # Update node colors and positions
        self.scatter.set_offsets(positions)
        self.scatter.set_array(activities)

        # Update connection lines
        if draw_lines:
            self.line_collection.set_segments(self.compute_lines(positions, adjacency_matrix))

        # Redraw the canvas
        self.fig.canvas.draw_idle()
        self.ax.figure.canvas.flush_events()

    def plot_connection_distribution_single(self, adjacency_matrix, title="Connection Distribution"):
        # Count connections for each node (degree of the nodes)
        connections = np.sum(adjacency_matrix, axis=1)

        # Plot histogram
        plt.figure(figsize=(8, 6))
        plt.hist(connections, bins=np.arange(connections.min(), connections.max() + 2), color='black', edgecolor='white')
        plt.title(title)
        plt.xlabel("#connections per unit")
        plt.ylabel("#units")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

class Simulation:
    def __init__(self, num_nodes, num_connections, alpha=1.7, epsilon=0.4, random_seed=None, metrics_file=None, connection_matrix_dir=None):
        self.network = NodeNetwork(num_nodes=num_nodes, num_connections=num_connections, alpha=alpha, epsilon=epsilon, random_seed=random_seed)
        self.metrics_file = metrics_file
        self.connection_matrix_dir = connection_matrix_dir

        # Initialize the metrics file
        if self.metrics_file:
            with open(self.metrics_file, "w") as f:
                f.write("Iteration,APL,CC,Breakups,Time\n")

        # Create directory for connection matrices if specified
        if self.connection_matrix_dir and not os.path.exists(self.connection_matrix_dir):
            os.makedirs(self.connection_matrix_dir)

    def save_connection_matrix(self, step):
            if self.connection_matrix_dir:
                np.save(os.path.join(self.connection_matrix_dir, f"connection_matrix_{step}.npy"), self.network.adjacency_matrix)

    def save_connection_histogram(self, step):
        if self.connection_matrix_dir:
            connections = np.sum(self.network.adjacency_matrix, axis=1)
            histogram, bins = np.histogram(connections, bins=np.arange(0, self.network.num_nodes + 1))
            np.save(os.path.join(self.connection_matrix_dir, f"connection_histogram_{step}.npy"), histogram)

    def metrics(self, step, start):
        characteristic_path_length, clustering_coefficient = self.network.calculate_stats()
        time_since_start = time.time() - start
        print(f"Iteration {step}: CPL={characteristic_path_length}, CC={clustering_coefficient}, Breakups={self.network.breakup_count}, Time={time_since_start:.2f}")

        if self.metrics_file:
            with open(self.metrics_file, "a") as f:
                f.write(f"{step},{characteristic_path_length},{clustering_coefficient},{self.network.breakup_count},{time_since_start:.2f}\n")
        
        self.save_connection_matrix(step)
        self.save_connection_histogram(step)

    def run(self, num_steps, display_interval, metrics_interval=METRICS_INTERVAL, figure_file=None, show=True):
        start = time.time()

        if display_interval:
            self.plot = NetworkPlot(self.network.physics.positions, self.network.activities, self.network.adjacency_matrix)
            if show:
                plt.ion()
                plt.show()

        # Main Loop
        for step in range(num_steps):
            if self.network.stabilized == True:
                print(f"Stabilized after {step} iterations.")
                break

            self.network.update_network()            

            if step % metrics_interval == 0:
                self.metrics(step, start)

            if display_interval and step % display_interval == 0:
                self.network.apply_forces(min(25, display_interval))
                self.plot.update_plot(self.network.physics.positions, self.network.activities, self.network.adjacency_matrix, title=f"{self.network.num_nodes} Nodes, {self.network.num_connections} Connections, Generation {step}")

        # Finalization
        self.metrics(step, start)

        #if display_interval:
            #for _ in range(min(150, display_interval)):
            #    self.network.apply_forces()
            #    self.plot.update_plot(self.network.positions, self.network.activities, self.network.adjacency_matrix, title=f"{self.network.num_nodes} Nodes, {self.network.num_connections} Connections, Generation {step}")

            #if figure_file:
            #    self.plot.fig.savefig(figure_file, dpi=300)

            #self.plot.plot_connection_distribution_single(self.network.adjacency_matrix, title=f"{self.network.num_nodes} Nodes, {self.network.num_connections} Connections, Distribution")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    if profiler: profiler.enable()

    # # Run the simulation
    print("Nodes:", NUM_NODES, "Connections:", NUM_CONNECTIONS, "Steps:", NUM_STEPS)
    sim = Simulation(num_nodes=NUM_NODES, num_connections=NUM_CONNECTIONS, alpha=ALPHA, epsilon=EPSILON, random_seed=RANDOM_SEED, metrics_file=f"{NUM_NODES}_{NUM_CONNECTIONS}_{NUM_STEPS}.csv")
    sim.run(num_steps=NUM_STEPS, display_interval=DISPLAY_INTERVAL)

    # Task 1: Networks with varying connection densities
    # for num_connections in range(550, 551, 1):
    #     print("Connections:", num_connections)
    #     metrics_file = f"density_test_data/100_{num_connections}_{NUM_STEPS}.csv"
    #     sim = Simulation(num_nodes=100, num_connections=num_connections, alpha=ALPHA, epsilon=EPSILON, random_seed=RANDOM_SEED, metrics_file=metrics_file)
    #     sim.run(num_steps=10_00, display_interval=1000, figure_file=f"density_test_data/100_{num_connections}_{1000}.png", show=True)

    # Task 2: 600-unit networks with connection matrix and histogram
    # for i in range(5):
    #     print(f"600 nodes {i}")
    #     metrics_file = f"metrics_600_nodes_{i}.csv"
    #     connection_matrix_dir = "600_node_data"
    #     sim = Simulation(num_nodes=600, num_connections=18000, alpha=ALPHA, epsilon=EPSILON, random_seed=RANDOM_SEED, metrics_file=metrics_file, connection_matrix_dir=connection_matrix_dir)
    #     sim.run(num_steps=1_000_000, display_interval=1000, figure_file=f"600_nodes_{i}.png", show=False)

    if profiler: profiler.disable()

    # Print profiler stats to sort by cumulative time
    if profiler: pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)

    for label, times in accumulated_times.items():
        print(f"{label}: {times['total_time']:.4f} seconds")