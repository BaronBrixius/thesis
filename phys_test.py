import unittest
import numpy as np
from NodeNetwork import NodeNetwork, NetworkPlot
import matplotlib.pyplot as plt

# Constants for screen dimensions
SCREEN_WIDTH = 1.0
SCREEN_HEIGHT = 1.0
RANDOM_SEED = None

class TestNodeNetworkPhysics(unittest.TestCase):
    def helper_set_cluster_colors(self, network, clusters):
        activity_values = np.linspace(-1, 1, len(clusters))
        for activity, cluster in zip(activity_values, clusters):
            network.activities[cluster] = activity

    def helper_create_clusters(self, network, num_clusters, density=1.0):
        cluster_size = network.num_nodes // num_clusters
        clusters = [list(range(i * cluster_size, (i + 1) * cluster_size)) for i in range(num_clusters)]

        for cluster in clusters:
            possible_connections = [
                (i, j) for idx, i in enumerate(cluster) for j in cluster[idx + 1 :]
            ]
            num_connections = int(len(possible_connections) * density)
            selected_connections = np.random.choice(
                range(len(possible_connections)), num_connections, replace=False
            )
            for idx in selected_connections:
                node1, node2 = possible_connections[idx]
                network.add_connection(node1, node2)

        # Assign cluster colors automatically
        self.helper_set_cluster_colors(network, clusters)

        return clusters

    def helper_create_small_world(self, num_nodes, num_clusters, inter_cluster_edges, density=.7):
        network = NodeNetwork(num_nodes=num_nodes, num_connections=0, random_seed=RANDOM_SEED)
        
        # Main clusters
        clusters = self.helper_create_clusters(network, num_clusters, density=density)

        # Sparse inter-cluster connections
        np.random.seed(RANDOM_SEED)
        for _ in range(inter_cluster_edges):
            cluster_a, cluster_b = np.random.choice(num_clusters, 2, replace=False)
            node_a = np.random.choice(clusters[cluster_a])
            node_b = np.random.choice(clusters[cluster_b])
            network.add_connection(node_a, node_b)

        return network

    def helper_display_and_iterate_network(self, network, title, steps=400, draw_lines=True):
        plot = NetworkPlot(network.physics.positions, network.activities, network.adjacency_matrix, draw_lines=draw_lines)
        plot.ax.set_xlabel(f"{title}")

        # Callback to set a flag when the plot window is closed
        plot_closed = False
        def on_close(event):
            nonlocal plot_closed
            plot_closed = True
        plot.fig.canvas.mpl_connect('close_event', on_close)

        for step in range(steps):
            if plot_closed:
                break  # Exit the loop if the plot is closed
            network.apply_forces()
            plot.update_plot(network.physics.positions, network.activities, network.adjacency_matrix, title=f"Generation {step}", draw_lines=draw_lines)
            plt.pause(0.000001)

        plot.ax.set_title(f"{title} - Final State")
        plt.pause(0.5)
        plt.close()

    # Test Cases
    def test_a_two_nodes_with_interloper(self):
        network = NodeNetwork(num_nodes=3, num_connections=1, random_seed=RANDOM_SEED)
        network.activities = np.full((3), 1.0)
        self.helper_display_and_iterate_network(network, "Two Connected Nodes With Interloper")

    def test_a_two_trios(self):
        network = NodeNetwork(num_nodes=6, num_connections=0, random_seed=RANDOM_SEED)
        self.helper_create_clusters(network, num_clusters=2, density=1.0)
        self.helper_display_and_iterate_network(network, "Two Clusters Iterating")

    def test_b_clustering_behavior(self):
        network = NodeNetwork(num_nodes=30, num_connections=0, random_seed=RANDOM_SEED)
        self.helper_create_clusters(network, num_clusters=3, density=.8)
        self.helper_display_and_iterate_network(network, "Basic Clusters")

    def test_c_loose_clustered_network(self):
        network = NodeNetwork(num_nodes=300, num_connections=0, random_seed=RANDOM_SEED)
        self.helper_create_clusters(network, num_clusters=10, density=.5)
        self.helper_display_and_iterate_network(network, "Loose Clusters")

    def test_c_tight_clustered_network(self):
        network = NodeNetwork(num_nodes=300, num_connections=0, random_seed=RANDOM_SEED)
        self.helper_create_clusters(network, num_clusters=10, density=.9)
        self.helper_display_and_iterate_network(network, "Tight Clusters")

    def test_d_small_world_100_nodes(self):
        network = self.helper_create_small_world(num_nodes=100, num_clusters=4, inter_cluster_edges=40)
        self.helper_display_and_iterate_network(network, title="Small-World Network (100 Nodes)")

    def test_d_small_world_300_nodes(self):
        network = self.helper_create_small_world(num_nodes=300, num_clusters=6, inter_cluster_edges=70)
        self.helper_display_and_iterate_network(network, title="Small-World Network (300 Nodes)")

    def test_d_small_world_500_nodes(self):
        network = self.helper_create_small_world(num_nodes=500, num_clusters=10, inter_cluster_edges=90)
        self.helper_display_and_iterate_network(network, title="Small-World Network (500 Nodes)")

    def test_e_small_world_1000_nodes(self):
        network = self.helper_create_small_world(num_nodes=1000, num_clusters=20, inter_cluster_edges=120)
        self.helper_display_and_iterate_network(network, title="Small-World Network (1000 Nodes)")

    def test_e_small_world_1000_nodes_no_lines(self):
        network = self.helper_create_small_world(num_nodes=1000, num_clusters=20, inter_cluster_edges=120)
        self.helper_display_and_iterate_network(network, title="Small-World Network (1000 Nodes)", draw_lines=False)

    def test_z_ring_lattice(self):
        num_nodes = 30
        network = NodeNetwork(num_nodes=num_nodes, num_connections=0, random_seed=RANDOM_SEED)
        for i in range(num_nodes):
            for j in range(1, 3):
                network.add_connection(i, (i + j) % num_nodes)
                network.add_connection(i, (i - j) % num_nodes)
        self.helper_display_and_iterate_network(network, "Ring Lattice Structure", 500)

if __name__ == "__main__":
    unittest.main()
