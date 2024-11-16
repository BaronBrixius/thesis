import unittest
import numpy as np
from NodeNetwork import NodeNetwork, NetworkPlot
import matplotlib.pyplot as plt

# Constants for screen dimensions
SCREEN_WIDTH = 2.0
SCREEN_HEIGHT = 2.0

class TestNodeNetworkPhysics(unittest.TestCase):
    def setUp(self):
        # Common parameters for the tests
        self.num_nodes = 100
        self.alpha = 1.7
        self.epsilon = 0.4
        self.random_seed = 42

    # Helper method to display and iterate the network
    def display_and_iterate(self, network, title, steps=100, interval=0.000001):
        plot = NetworkPlot(network.positions, network.activities, network.adjacency_matrix)
        plot.ax.set_xlabel(f"{title}")
        for step in range(steps):
            network.apply_forces()
            plot.update_plot(network.positions, network.activities, network.adjacency_matrix, step, 0, 0)  # Metrics are placeholders here
            plt.pause(interval)

        plot.ax.set_title(f"{title} - Final State")
        plt.pause(0.5)
        plt.close()

    # Helper method to initialize positions and activities
    def initialize_positions_and_activities(self, network, clusters=None):
        # Initialize random positions
        network.positions = np.random.uniform(0, 1, (network.num_nodes, 2))
        network.positions[:, 0] *= SCREEN_WIDTH
        network.positions[:, 1] *= SCREEN_HEIGHT

        # Assign activities
        if clusters:
            num_clusters = len(clusters)
            activity_values = np.linspace(-1, 1, num_clusters)  # Spread activities evenly
            network.activities = np.zeros(network.num_nodes)

            for i, cluster in enumerate(clusters):
                activity = activity_values[i]
                for node in cluster:
                    network.activities[node] = activity
        else:
            network.activities = np.random.uniform(-1, 1, network.num_nodes)

    def test_two_connected_nodes_with_interloper(self):
        network = NodeNetwork(num_nodes=3, num_connections=0, alpha=self.alpha, epsilon=self.epsilon, random_seed=self.random_seed)
        network.add_connection(0, 1)
        self.initialize_positions_and_activities(network, clusters=[[0, 1], [2]])
        self.display_and_iterate(network, "Two Connected Nodes With Interloper", 75)

    def test_two_clusters_iterating(self):
        network = NodeNetwork(num_nodes=6, num_connections=0, alpha=self.alpha, epsilon=self.epsilon, random_seed=self.random_seed)
        network.add_connection(0, 1)
        network.add_connection(1, 2)
        network.add_connection(3, 4)
        network.add_connection(4, 5)
        self.initialize_positions_and_activities(network, clusters=[[0, 1, 2], [3, 4, 5]])
        self.display_and_iterate(network, "Two Clusters Iterating", 100)

    def test_clustering_behavior(self):
        num_nodes = 30
        network = NodeNetwork(num_nodes=num_nodes, num_connections=0, alpha=self.alpha, epsilon=self.epsilon, random_seed=self.random_seed)
        for i in range(10):
            for j in range(i + 1, 10):
                network.add_connection(i, j)
        for i in range(10, 20):
            for j in range(i + 1, 20):
                network.add_connection(i, j)
        for i in range(20, 30):
            for j in range(i + 1, 30):
                network.add_connection(i, j)
        self.initialize_positions_and_activities(network, clusters=[[i for i in range(10)], [i for i in range(10, 20)], [i for i in range(20, 30)]])
        self.display_and_iterate(network, "Clustering Behavior", 75)

    def test_ring_lattice(self):
        num_nodes = 30
        k = 4  # Each node connects to k nearest neighbors
        network = NodeNetwork(num_nodes=num_nodes, num_connections=0, alpha=self.alpha, epsilon=self.epsilon, random_seed=self.random_seed)
        for i in range(num_nodes):
            for j in range(1, k // 2 + 1):
                network.add_connection(i, (i + j) % num_nodes)
                network.add_connection(i, (i - j) % num_nodes)
        self.initialize_positions_and_activities(network)
        self.display_and_iterate(network, "Ring Lattice Structure", 150)

    def test_scale_free_network(self):
        num_nodes = 50
        m = 2  # New nodes connect to m existing nodes
        network = NodeNetwork(num_nodes=0, num_connections=0, alpha=self.alpha, epsilon=self.epsilon, random_seed=self.random_seed)
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                network.add_connection(i, j)
        self.initialize_positions_and_activities(network)
        for new_node in range(m + 1, num_nodes):
            network.positions = np.vstack([network.positions, np.random.uniform(0, 1, 2)])
            network.activities = np.append(network.activities, np.random.uniform(-1, 1))
            degree = network.adjacency_matrix.sum(axis=1)
            probs = degree / degree.sum()
            existing_nodes = np.random.choice(np.arange(len(degree)), size=m, replace=False, p=probs)
            for existing_node in existing_nodes:
                network.add_connection(new_node, existing_node)
        self.display_and_iterate(network, "Scale-Free Network", 150)


if __name__ == "__main__":
    unittest.main()
