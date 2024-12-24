import unittest
import numpy as np
import networkx as nx
from network_simulation.metrics import Metrics

class TestCalculator(unittest.TestCase):
    def setUp(self):
        """Set up reusable test data."""
        self.calculator = Metrics()

        # Create a small adjacency matrix for testing
        self.adjacency_matrix = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0]
        ])
        self.activities = np.array([0.2, 0.5, 0.8, 0.1])

    def test_calculate_rewiring_chance(self):
        """Test the rewiring chance calculation."""
        chance = self.calculator.calculate_rewiring_chance(self.adjacency_matrix, self.activities)
        self.assertAlmostEqual(chance, 0.5)

    def test_calculate_edge_persistence(self):
        """Test edge persistence calculation."""
        previous_matrix = np.array([
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 0]
        ])
        persistence = self.calculator.calculate_edge_persistence(self.adjacency_matrix, previous_matrix)
        self.assertAlmostEqual(persistence, 0.75)

    def test_detect_communities(self):
        """Test community detection."""
        cluster_assignments = self.calculator.detect_communities(nx.from_numpy_array(self.adjacency_matrix))
        self.assertEqual(len(set(cluster_assignments)), 2)  # Two communities expected in the test graph

    def test_calculate_cluster_membership_stability(self):
        """Test cluster membership stability."""
        previous_assignments = np.array([0, 1, 1, 0, 1])
        current_assignments = np.array([0, 1, 0, 1, 1])
        stability = self.calculator.calculate_cluster_membership_stability(current_assignments, previous_assignments)
        self.assertAlmostEqual(stability, -0.25)

    def test_calculate_cluster_size_variance(self):
        """Test cluster size variance."""
        cluster_assignments = np.array([0, 1, 1, 0])
        variance = self.calculator.calculate_cluster_size_variance(cluster_assignments)
        self.assertAlmostEqual(variance, 0.0)  # Two equal-sized clusters

if __name__ == "__main__":
    unittest.main()
