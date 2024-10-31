import unittest
from NodeNetwork import NodeNetwork

def setup_network(num_nodes, connections, alpha=1.7, epsilon=0.4, random_seed=None):
    """
    Helper function to initialize a network with specific connections.
    """
    network = NodeNetwork(num_nodes=num_nodes, num_connections=0, alpha=alpha, epsilon=epsilon, random_seed=random_seed)
    for i, j in connections:
        network.add_connection(i, j)
    return network

class TestNetworkMetrics(unittest.TestCase):
    
    def test_simple_two_nodes(self):
        num_nodes = 2
        connections = [(0, 1)]
        expected_cpl = 1
        expected_cc = 0

        # Setup network and calculate metrics
        network = setup_network(num_nodes, connections)
        calculated_cpl, calculated_cc = network.calculate_metrics()

        self.assertAlmostEqual(calculated_cpl, expected_cpl, delta=0.01, msg="CPL does not match expected value")
        self.assertAlmostEqual(calculated_cc, expected_cc, delta=0.01, msg="CC does not match expected value")

    def test_star_network(self):
        num_nodes = 5
        connections = [(0, 1), (0, 2), (0, 3), (0, 4)]
        expected_cpl = 1.6
        expected_cc = 0.0

        # Setup network and calculate metrics
        network = setup_network(num_nodes, connections)
        calculated_cpl, calculated_cc = network.calculate_metrics()

        self.assertAlmostEqual(calculated_cpl, expected_cpl, delta=0.01, msg="CPL does not match expected value")
        self.assertAlmostEqual(calculated_cc, expected_cc, delta=0.01, msg="CC does not match expected value")

if __name__ == "__main__":
    unittest.main()
