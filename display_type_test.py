from graph_tool.all import Graph, Edge, lattice
from graph_tool.draw import graph_draw
from network_simulation.visualization import Visualization
import numpy as np
import os

def create_small_world_network(num_nodes, rewire_prob, target_density=0.2):
    """
    Create a small-world network by rewiring edges in a lattice with added random edges.
    """
    graph: Graph = lattice(shape=[num_nodes], periodic=True)

    # Calculate target number of edges based on desired density
    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    target_edges = int(target_density * max_possible_edges)

    # Add random edges until target density is reached
    while graph.num_edges() < target_edges:
        v1 = np.random.randint(0, num_nodes)
        v2 = np.random.randint(0, num_nodes)
        if v1 != v2 and not graph.edge(v1, v2):
            graph.add_edge(v1, v2)

    # Perform rewiring to simulate small-world behavior
    edges = list(graph.edges())
    for edge in edges:
        if np.random.rand() < rewire_prob:
            v1 = edge.source()
            v2 = edge.target()
            graph.remove_edge(edge)
            new_v2 = np.random.choice(list(set(range(num_nodes)) - {int(v1)}))
            graph.add_edge(v1, graph.vertex(new_v2))

    return graph

def test_visualizations():
    """
    Generate and save small-world networks with different layouts and rewiring probabilities.
    """
    # Parameters for the small-world network
    num_nodes = 100
    rewire_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Different rewiring probabilities
    shape = [num_nodes]  # 1D lattice
    output_dir = "test_visuals"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test various layouts
    layouts = ["sfdp", "arf", "fr"]
    
    for rewire_prob in rewire_probs:
        for layout in layouts:
            print(f"Generating visualization with rewiring probability: {rewire_prob}, layout: {layout}")
            
            # Create small-world network
            graph = create_small_world_network(num_nodes, rewire_prob)
            
            # Assign random activities and clusters for testing
            activities = graph.new_vertex_property("float")
            cluster_assignments = graph.new_vertex_property("int")
            for v in graph.vertices():
                activities[v] = rewire_prob  # Use rewiring probability for simplicity
                cluster_assignments[v] = int(v) % 3  # Divide nodes into 3 arbitrary clusters
            
            # Initialize visualization
            viz = Visualization(
                graph=graph,
                activities=activities,
                cluster_assignments=cluster_assignments,
                layout_type=layout,  # Dynamic layout selection
                output_dir=output_dir
            )
            
            # Handle normalization issue for `activities`
            if np.ptp(activities.a) > 0:  # Check if range is not zero
                viz.vertex_colors = viz._compute_vertex_colors()  # Recompute safely
            
            # Save graph to file
            output_path = os.path.join(output_dir, f"small_world_p{rewire_prob}_{layout}.png")
            graph_draw(graph, pos=viz.positions, vertex_fill_color=viz.vertex_colors, output=output_path)
            print(f"Saved: {output_path}")

# Run the test suite
test_visualizations()
