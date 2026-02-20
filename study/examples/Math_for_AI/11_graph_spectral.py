"""
Graph Theory and Spectral Graph Theory

This script demonstrates:
- Graph construction (adjacency, degree, Laplacian matrices)
- Spectral decomposition of graph Laplacian
- Spectral clustering algorithm
- Simple Graph Neural Network (GNN) message passing
- PageRank computation

Spectral graph theory connects graph structure to eigenvalues/eigenvectors
of graph matrices. It's foundational for:
- Graph clustering
- Graph Neural Networks (GNNs)
- Graph signal processing
- Community detection
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import torch
import torch.nn.functional as F


def construct_graph_matrices(edges, num_nodes):
    """
    Construct graph matrices from edge list.

    Args:
        edges: List of (source, target) tuples
        num_nodes: Number of nodes in graph

    Returns:
        A: Adjacency matrix
        D: Degree matrix
        L: Laplacian matrix (D - A)
        L_norm: Normalized Laplacian
    """
    # Adjacency matrix
    A = np.zeros((num_nodes, num_nodes))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1  # Undirected graph

    # Degree matrix
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)

    # Laplacian matrix: L = D - A
    L = D - A

    # Normalized Laplacian: L_norm = D^(-1/2) L D^(-1/2) = I - D^(-1/2) A D^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))
    L_norm = np.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

    return A, D, L, L_norm


def spectral_decomposition(L):
    """
    Compute eigenvalues and eigenvectors of Laplacian.

    The eigenvalues of the Laplacian encode important graph properties:
    - Number of zero eigenvalues = number of connected components
    - Second smallest eigenvalue (Fiedler value) measures connectivity
    - Eigenvectors can be used for embedding and clustering
    """
    eigenvalues, eigenvectors = eigh(L)

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def spectral_clustering(L_norm, num_clusters=2):
    """
    Spectral clustering algorithm.

    Steps:
    1. Compute k smallest eigenvectors of normalized Laplacian
    2. Use these as feature representations
    3. Apply k-means clustering

    Args:
        L_norm: Normalized Laplacian matrix
        num_clusters: Number of clusters

    Returns:
        Cluster assignments
    """
    # Get k smallest eigenvectors
    eigenvalues, eigenvectors = spectral_decomposition(L_norm)
    features = eigenvectors[:, :num_clusters]

    # Simple k-means clustering (manual implementation)
    # Initialize centroids randomly
    num_nodes = features.shape[0]
    centroid_indices = np.random.choice(num_nodes, num_clusters, replace=False)
    centroids = features[centroid_indices]

    # Iterative assignment and update
    max_iters = 100
    for _ in range(max_iters):
        # Assign to nearest centroid
        distances = np.linalg.norm(features[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([features[assignments == k].mean(axis=0) for k in range(num_clusters)])

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return assignments


def simple_gnn_forward_pass(A, X, W):
    """
    Simple Graph Neural Network (GNN) forward pass.

    Message passing: H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))

    Args:
        A: Adjacency matrix
        X: Node features (num_nodes x feature_dim)
        W: Weight matrix (feature_dim x output_dim)

    Returns:
        Updated node features
    """
    # Add self-loops
    A_tilde = A + np.eye(A.shape[0])

    # Degree matrix
    D_tilde = np.diag(np.sum(A_tilde, axis=1))

    # Normalized adjacency: D^(-1/2) A D^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde) + 1e-10))
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    # Message passing
    H = A_norm @ X @ W

    # Apply activation (ReLU)
    H = np.maximum(0, H)

    return H


def pagerank(A, damping=0.85, max_iters=100, tol=1e-6):
    """
    PageRank algorithm.

    PageRank models a random walk on the graph:
    r = (1-d)/N * 1 + d * A^T D^(-1) r

    where d is damping factor, N is number of nodes.

    Args:
        A: Adjacency matrix
        damping: Damping factor (probability of following links)
        max_iters: Maximum iterations
        tol: Convergence tolerance

    Returns:
        PageRank scores
    """
    num_nodes = A.shape[0]

    # Out-degree (column sums for directed graph, row sums for undirected)
    out_degree = np.sum(A, axis=1)
    out_degree[out_degree == 0] = 1  # Avoid division by zero

    # Transition matrix: P = A^T D^(-1)
    P = (A.T / out_degree).T

    # Initialize ranks uniformly
    r = np.ones(num_nodes) / num_nodes

    # Power iteration
    for i in range(max_iters):
        r_new = (1 - damping) / num_nodes + damping * P.T @ r

        # Check convergence
        if np.linalg.norm(r_new - r, 1) < tol:
            print(f"PageRank converged in {i+1} iterations")
            break

        r = r_new

    return r


def visualize_graph_clustering():
    """
    Create a graph with clear community structure and apply spectral clustering.
    """
    print("=== Spectral Clustering Example ===\n")

    # Create a graph with two communities
    # Community 1: nodes 0-4
    # Community 2: nodes 5-9
    edges_c1 = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (2, 4)]
    edges_c2 = [(5, 6), (5, 7), (6, 7), (6, 8), (7, 8), (8, 9), (7, 9)]
    edges_between = [(4, 5)]  # Weak connection between communities

    all_edges = edges_c1 + edges_c2 + edges_between
    num_nodes = 10

    # Construct matrices
    A, D, L, L_norm = construct_graph_matrices(all_edges, num_nodes)

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {len(all_edges)}")
    print(f"Average degree: {np.sum(A) / num_nodes:.2f}\n")

    # Spectral decomposition
    eigenvalues, eigenvectors = spectral_decomposition(L_norm)
    print(f"Smallest 5 eigenvalues: {eigenvalues[:5]}")
    print(f"Fiedler value (2nd smallest): {eigenvalues[1]:.4f}\n")

    # Spectral clustering
    assignments = spectral_clustering(L_norm, num_clusters=2)
    print(f"Cluster assignments: {assignments}\n")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Adjacency matrix
    im1 = axes[0].imshow(A, cmap='Blues', interpolation='nearest')
    axes[0].set_title('Adjacency Matrix')
    axes[0].set_xlabel('Node')
    axes[0].set_ylabel('Node')
    plt.colorbar(im1, ax=axes[0])

    # Plot 2: First 3 eigenvectors
    axes[1].plot(eigenvectors[:, 0], 'o-', label='λ=0 (constant)')
    axes[1].plot(eigenvectors[:, 1], 's-', label=f'λ={eigenvalues[1]:.3f} (Fiedler)')
    axes[1].plot(eigenvectors[:, 2], '^-', label=f'λ={eigenvalues[2]:.3f}')
    axes[1].set_title('First 3 Eigenvectors of Laplacian')
    axes[1].set_xlabel('Node Index')
    axes[1].set_ylabel('Eigenvector Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Spectral embedding (2D)
    colors = ['red' if a == 0 else 'blue' for a in assignments]
    axes[2].scatter(eigenvectors[:, 1], eigenvectors[:, 2], c=colors, s=100, alpha=0.6)
    for i in range(num_nodes):
        axes[2].annotate(str(i), (eigenvectors[i, 1], eigenvectors[i, 2]),
                        ha='center', va='center', fontsize=8, color='white', weight='bold')
    axes[2].set_title('Spectral Embedding (Fiedler vs 3rd eigenvector)')
    axes[2].set_xlabel('2nd Eigenvector (Fiedler)')
    axes[2].set_ylabel('3rd Eigenvector')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/graph_spectral_clustering.png', dpi=150, bbox_inches='tight')
    print("Plot saved to /tmp/graph_spectral_clustering.png\n")
    plt.close()


def demonstrate_gnn():
    """
    Demonstrate simple GNN message passing.
    """
    print("=== Graph Neural Network (GNN) Message Passing ===\n")

    # Small graph
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    num_nodes = 4
    A, _, _, _ = construct_graph_matrices(edges, num_nodes)

    # Random node features (4 nodes, 3 features each)
    X = np.random.randn(num_nodes, 3)
    print(f"Input node features shape: {X.shape}")
    print(f"Input features:\n{X}\n")

    # Random weight matrix (3 input features -> 2 output features)
    W = np.random.randn(3, 2)

    # Forward pass
    H = simple_gnn_forward_pass(A, X, W)
    print(f"Output node features shape: {H.shape}")
    print(f"Output features:\n{H}\n")

    print("Message passing aggregates information from neighbors,")
    print("weighted by the normalized adjacency matrix.\n")


def demonstrate_pagerank():
    """
    Demonstrate PageRank on a simple directed graph.
    """
    print("=== PageRank Algorithm ===\n")

    # Create a simple directed graph
    # Node 0 links to 1, 2
    # Node 1 links to 2
    # Node 2 links to 0
    # Node 3 links to 2 (dangling node pointing in)
    A = np.array([
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ], dtype=float)

    print("Adjacency matrix (directed graph):")
    print(A)
    print()

    # Compute PageRank
    ranks = pagerank(A, damping=0.85)

    print("PageRank scores:")
    for i, r in enumerate(ranks):
        print(f"  Node {i}: {r:.4f}")
    print(f"  Sum: {np.sum(ranks):.4f} (should be 1.0)\n")

    # Visualize
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(ranks)), ranks, color='steelblue', alpha=0.7)
    ax.set_title('PageRank Scores')
    ax.set_xlabel('Node')
    ax.set_ylabel('PageRank Score')
    ax.set_xticks(range(len(ranks)))
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate values
    for i, (bar, rank) in enumerate(zip(bars, ranks)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{rank:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('/tmp/pagerank.png', dpi=150, bbox_inches='tight')
    print("Plot saved to /tmp/pagerank.png\n")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Graph Theory and Spectral Graph Theory")
    print("=" * 60)
    print()

    # Set random seed
    np.random.seed(42)

    # Run demonstrations
    visualize_graph_clustering()

    demonstrate_gnn()

    demonstrate_pagerank()

    print("=" * 60)
    print("Summary:")
    print("- Graph Laplacian eigenvalues encode connectivity structure")
    print("- Spectral clustering uses eigenvectors as node embeddings")
    print("- GNNs aggregate neighbor information via message passing")
    print("- PageRank measures node importance via random walks")
    print("=" * 60)
