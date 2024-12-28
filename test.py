import unittest
from graph_tool.all import Graph
from network_simulation.network import NodeNetwork

class TestNodeNetwork(unittest.TestCase):

    def setUp(self):
        self.num_nodes = 10
        self.num_connections = 20
        self.alpha = 1.7
        self.epsilon = 0.4
        self.random_seed = 42
        self.network = NodeNetwork(
            num_nodes=self.num_nodes,
            num_connections=self.num_connections,
            alpha=self.alpha,
            epsilon=self.epsilon,
            random_seed=self.random_seed
        )

    def test_initialization(self):
        """Test if the network initializes correctly."""
        self.assertEqual(self.network.num_nodes, self.num_nodes)
        self.assertEqual(self.network.graph.num_vertices(), self.num_nodes)
        self.assertEqual(self.network.graph.num_edges(), self.num_connections)
        self.assertAlmostEqual(self.network.alpha, self.alpha)
        self.assertAlmostEqual(self.network.epsilon, self.epsilon)

    def test_activity_update(self):
        """Test if the activity update function works as expected."""
        initial_activities = self.network.activities.a.copy()
        self.network.update_activity()
        updated_activities = self.network.activities.a
        self.assertFalse((initial_activities == updated_activities).all())
        self.assertTrue(((updated_activities >= -1) & (updated_activities <= 1)).all())

    def test_rewire(self):
        """Test if the rewiring function modifies the graph structure."""
        initial_edges = set(tuple(sorted((int(e.source()), int(e.target())))) for e in self.network.graph.edges())
        self.network.rewire(step=1)
        updated_edges = set(tuple(sorted((int(e.source()), int(e.target())))) for e in self.network.graph.edges())
        self.assertNotEqual(initial_edges, updated_edges)

    def test_rewiring_consistency(self):
        """Test rewiring consistency to ensure no duplicate or self-loops."""
        self.network.rewire(step=1)
        for e in self.network.graph.edges():
            self.assertNotEqual(e.source(), e.target())  # No self-loops

        edge_set = set(tuple(sorted((int(e.source()), int(e.target())))) for e in self.network.graph.edges())
        self.assertEqual(len(edge_set), self.network.graph.num_edges())  # No duplicates

if __name__ == "__main__":
    unittest.main()
