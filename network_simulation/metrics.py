from graph_tool.all import Graph, local_clustering, shortest_distance
import networkx as nx
import numpy as np
from typing import Optional, Dict

def compute_metrics(adjacency_matrix, graph, entropy, community_assignments, step):
    community_metrics = calculate_community_metrics(graph, entropy, community_assignments, adjacency_matrix)
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

def calculate_community_metrics(graph, entropy, community_assignments, adjacency_matrix) -> Dict[str, object]:
    unique_communities, community_sizes = np.unique(community_assignments, return_counts=True)
    intra_community_densities, intra_community_edges = calculate_community_densities(adjacency_matrix, community_assignments, unique_communities)

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

def calculate_community_densities(adjacency_matrix, community_assignments, unique_communities):
    intra_community_densities = {}
    intra_community_edges = 0

    for community in unique_communities:
        # Get node indices for the community
        community_nodes = np.where(community_assignments == community)[0]

        if len(community_nodes) > 1:
            # Extract subgraph adjacency matrix
            subgraph_matrix = adjacency_matrix[np.ix_(community_nodes, community_nodes)]
            
            # Count edges in the community
            num_community_edges = np.sum(subgraph_matrix) / 2  # Since it's undirected, divide by 2
            num_possible_community_edges = len(community_nodes) * (len(community_nodes) - 1) / 2

            # Compute density
            intra_community_densities[community] = num_community_edges / num_possible_community_edges if num_possible_community_edges > 0 else 0
            intra_community_edges += num_community_edges
        else:
            intra_community_densities[community] = 0  # A single-node community has density 0

    return intra_community_densities, intra_community_edges
