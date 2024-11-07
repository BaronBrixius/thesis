import unittest
import numpy as np
from NodeNetwork import NodeNetwork


class TestNetworkMetrics(unittest.TestCase):
    
    def helper_metrics(self, num_nodes, connections, expected_cpl, expected_cc, alpha=1.7, epsilon=0.4, random_seed=None):
        # Setup network and calculate metrics
        network = NodeNetwork(num_nodes, 0, alpha, epsilon, random_seed)
        for i, j in connections:
            network.add_connection(i, j)
        calculated_cpl, calculated_cc = network.calculate_stats()

        self.assertAlmostEqual(calculated_cpl, expected_cpl, delta=0.01, msg="CPL does not match expected value")
        self.assertAlmostEqual(calculated_cc, expected_cc, delta=0.01, msg="CC does not match expected value")

    def test_simple_two_nodes(self):
        self.helper_metrics(num_nodes=2, connections=[(0, 1)], expected_cpl=1.0, expected_cc=0.0)

    def test_simple_three_nodes(self):
        self.helper_metrics(num_nodes=3, connections=[(0, 1), (1, 2)], expected_cpl=1.33, expected_cc=0.0)    

    def test_star_network(self):
        self.helper_metrics(num_nodes=5, connections=[(0, 1), (0, 2), (0, 3), (0, 4)], expected_cpl=1.6, expected_cc=0.0)

    def test_fully_connected(self):
        self.helper_metrics(num_nodes=4, connections=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], expected_cpl=1.0, expected_cc=1.0)

    def test_ring_network(self):
        self.helper_metrics(num_nodes=4, connections=[(0, 1), (1, 2), (2, 3), (3, 0)], expected_cpl=1.3333, expected_cc=0.0)

    # Tests for `update_activity`
    def helper_update_activity(self, initial_activities, expected_activities, connections, alpha=1.7, epsilon=0.4):
        # Setup network
        num_nodes = len(initial_activities)
        network = NodeNetwork(num_nodes, 0, alpha, epsilon)
        network.activities = np.array(initial_activities)
        for i, j in connections:
            network.add_connection(i, j)
        
        # Perform activity update
        network.update_activity()

        # Check each activity value matches the expected value within a tolerance
        for i, expected in enumerate(expected_activities):
            self.assertAlmostEqual(network.activities[i], expected, delta=0.01, msg=f"Activity at node {i} does not match expected value")

    def test_activity_no_connections(self):
        # No connections, each node should update independently
        initial_activities = [0.5, -0.5]
        expected_activities = [1 - 1.7 * (0.5)**2, 1 - 1.7 * (-0.5)**2]  # Using only the logistic map
        self.helper_update_activity(initial_activities, expected_activities, connections=[])

    def test_activity_with_connections(self):
        # Nodes connected to each other, expect mutual influence
        initial_activities = [0.5, 0.8]
        connections = [(0, 1)]
        
        # Calculate expected activities manually
        alpha, epsilon = 1.7, 0.4
        own_activities = [1 - alpha * (a**2) for a in initial_activities]
        neighbor_activity = initial_activities[1]  # Node 0's neighbor has activity 0.8
        expected_activities = [
            (1 - epsilon) * own_activities[0] + epsilon * neighbor_activity,  # Node 0 influenced by Node 1
            (1 - epsilon) * own_activities[1] + epsilon * initial_activities[0]  # Node 1 influenced by Node 0
        ]
        
        self.helper_update_activity(initial_activities, expected_activities, connections)

    def test_activity_mixed_connections(self):
        # Mixed scenario: One node connected, one isolated
        initial_activities = [0.3, -0.7, 0.5]
        connections = [(0, 2)]
        
        # Expected values for each node
        alpha, epsilon = 1.7, 0.4
        own_activities = [1 - alpha * (a**2) for a in initial_activities]
        neighbor_activity = initial_activities[2]  # Node 0's neighbor has activity 0.5
        expected_activities = [
            (1 - epsilon) * own_activities[0] + epsilon * neighbor_activity,  # Node 0 influenced by Node 2
            own_activities[1],  # Node 1 is isolated
            (1 - epsilon) * own_activities[2] + epsilon * initial_activities[0]  # Node 2 influenced by Node 0
        ]

        self.helper_update_activity(initial_activities, expected_activities, connections)


if __name__ == "__main__":
    unittest.main()
