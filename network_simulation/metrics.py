from graph_tool.all import Graph, local_clustering, shortest_distance
import networkx as nx
import numpy as np
from typing import Optional, Dict

def compute_metrics(adjacency_matrix, graph, entropy, step, community_assignments):
    community_metrics = calculate_community_metrics(community_assignments, graph, entropy)

    nx_graph = nx.from_numpy_array(adjacency_matrix)

    # Compute row data
    row = {
        "Step": step,
        "Clustering Coefficient": calculate_clustering_coefficient(graph),
        "Average Path Length": calculate_average_path_length(graph),
        "Rich Club Coefficients": calculate_rich_club_coefficients(nx_graph),
    }

    # Add community metrics
    row.update(community_metrics)

    return row

def calculate_clustering_coefficient(graph: Graph) -> float:
    """
    Clustering Coefficient (CC): Tendency of nodes to form tightly knit groups (triangles).
    """
    return local_clustering(graph).get_array().mean()

def calculate_average_path_length(graph: Graph) -> Optional[float]:
    """
    Average Path Length (APL): Average shortest path length between all pairs of nodes in the network.
    """
    distances = shortest_distance(graph, directed=False)
    num_vertex_pairs = (graph.num_vertices()**2 - graph.num_vertices())
    ave_path_length = 2 * sum([sum(row[j + 1:]) for j, row in enumerate(distances)]) / num_vertex_pairs  # Only sum the upper triangle of the distance matrix, and double it
    return ave_path_length

def calculate_rich_club_coefficients(nx_graph) -> Dict[int, float]:
    return nx.rich_club_coefficient(nx_graph, normalized=False)

def calculate_community_metrics(community_assignments, graph, entropy) -> Dict[str, object]:
    unique_communities, community_sizes = np.unique(community_assignments, return_counts=True)
    intra_community_densities, intra_community_edges = calculate_community_densities(graph, community_assignments, unique_communities)

    return {
        "Community Count": len(unique_communities),
        "Community Sizes": dict(zip(unique_communities, community_sizes)),
        "Community Densities": intra_community_densities,
        "Community Size Variance": np.var(community_sizes),
        "SBM Entropy Normalized": (entropy / graph.num_edges()) if graph.num_edges() > 0 else 0,
        "Intra-Community Edges": intra_community_edges,
        "Community Membership": community_assignments,
        "Node Degrees": graph.get_total_degrees(graph.get_vertices()),
    }

def calculate_community_densities(graph, community_assignments, unique_communities):
    intra_community_densities = {}
    intra_community_edges = 0

    for community in unique_communities:
        # Filter to only see the current community
        community_nodes = np.where(community_assignments == community)[0]
        graph.set_vertex_filter(graph.new_vertex_property("bool", vals=[int(v) in community_nodes for v in graph.vertices()]), inverted=False)

        # Calculate density
        num_community_edges = graph.num_edges()
        num_possible_community_edges = len(community_nodes) * (len(community_nodes) - 1) / 2
        intra_community_densities[community] = num_community_edges / num_possible_community_edges

        # Update total intra-community edges
        intra_community_edges += num_community_edges

        graph.set_vertex_filter(None)

    return intra_community_densities, intra_community_edges
