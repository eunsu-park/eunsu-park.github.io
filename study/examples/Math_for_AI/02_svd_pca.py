"""
Singular Value Decomposition (SVD) and Principal Component Analysis (PCA)

Demonstrates:
- SVD decomposition with NumPy
- PCA from scratch (centering → covariance → eigendecomposition)
- Comparison with sklearn PCA
- Explained variance visualization
- ML application: dimensionality reduction

Dependencies: numpy, matplotlib, sklearn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.datasets import load_iris


def demonstrate_svd():
    """Demonstrate Singular Value Decomposition"""
    print("=" * 60)
    print("SINGULAR VALUE DECOMPOSITION (SVD)")
    print("=" * 60)

    # Create a matrix
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]], dtype=float)

    print(f"\nOriginal matrix A (4x3):\n{A}")
    print(f"Shape: {A.shape}")
    print(f"Rank: {np.linalg.matrix_rank(A)}")

    # Perform SVD: A = U @ S @ V^T
    U, s, Vt = np.linalg.svd(A, full_matrices=True)

    print(f"\n--- SVD Components ---")
    print(f"U shape: {U.shape} (left singular vectors)")
    print(f"s shape: {s.shape} (singular values)")
    print(f"V^T shape: {Vt.shape} (right singular vectors)")

    print(f"\nSingular values: {s}")

    # Reconstruct S matrix (diagonal)
    S = np.zeros((U.shape[1], Vt.shape[0]))
    S[:len(s), :len(s)] = np.diag(s)

    print(f"\nS matrix (diagonal):\n{S}")

    # Reconstruct A
    A_reconstructed = U @ S @ Vt
    print(f"\nReconstructed A:\n{A_reconstructed}")
    print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.10f}")

    # Low-rank approximation
    print("\n--- Low-Rank Approximation ---")
    k = 2  # Keep only top 2 singular values

    S_k = S.copy()
    S_k[:, k:] = 0  # Zero out smaller singular values

    A_approx = U @ S_k @ Vt
    print(f"\nRank-{k} approximation:\n{A_approx}")

    error = np.linalg.norm(A - A_approx)
    print(f"Approximation error (Frobenius norm): {error:.4f}")

    # Energy preserved
    energy_original = np.sum(s**2)
    energy_kept = np.sum(s[:k]**2)
    print(f"\nEnergy preserved: {energy_kept/energy_original*100:.2f}%")


def pca_from_scratch(X, n_components=2):
    """
    Implement PCA from scratch

    Steps:
    1. Center the data (subtract mean)
    2. Compute covariance matrix
    3. Compute eigenvalues and eigenvectors
    4. Sort by eigenvalues (descending)
    5. Project data onto top k eigenvectors
    """
    print("\n--- PCA From Scratch ---")

    # Step 1: Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    print(f"Data shape: {X.shape}")
    print(f"Mean: {mean}")

    # Step 2: Compute covariance matrix
    n_samples = X.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    print(f"\nCovariance matrix shape: {cov_matrix.shape}")
    print(f"Covariance matrix:\n{cov_matrix}")

    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Step 4: Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print(f"\nEigenvalues: {eigenvalues}")
    print(f"\nTop {n_components} eigenvectors (principal components):")
    print(eigenvectors[:, :n_components])

    # Step 5: Project data
    X_pca = X_centered @ eigenvectors[:, :n_components]

    # Explained variance
    explained_variance = eigenvalues / np.sum(eigenvalues)
    explained_variance_cumsum = np.cumsum(explained_variance)

    print(f"\nExplained variance ratio: {explained_variance[:n_components]}")
    print(f"Cumulative explained variance: {explained_variance_cumsum[:n_components]}")

    return X_pca, eigenvectors[:, :n_components], explained_variance, mean


def pca_with_sklearn(X, n_components=2):
    """Use sklearn PCA for comparison"""
    print("\n--- PCA with sklearn ---")

    pca = SklearnPCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    print(f"Transformed shape: {X_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")
    print(f"\nPrincipal components (shape {pca.components_.shape}):")
    print(pca.components_.T)

    return X_pca, pca


def demonstrate_pca_on_iris():
    """Demonstrate PCA on Iris dataset"""
    print("\n" + "=" * 60)
    print("PCA ON IRIS DATASET")
    print("=" * 60)

    # Load Iris dataset
    iris = load_iris()
    X = iris.data  # 150 samples, 4 features
    y = iris.target
    feature_names = iris.feature_names

    print(f"\nIris dataset:")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Feature names: {feature_names}")
    print(f"First 5 samples:\n{X[:5]}")

    # Apply custom PCA
    print("\n" + "=" * 60)
    print("CUSTOM PCA")
    print("=" * 60)
    X_pca_custom, components_custom, explained_var, mean = pca_from_scratch(X, n_components=2)

    # Apply sklearn PCA
    print("\n" + "=" * 60)
    print("SKLEARN PCA")
    print("=" * 60)
    X_pca_sklearn, pca_model = pca_with_sklearn(X, n_components=2)

    # Verify they match (may differ in sign)
    print("\n--- Verification ---")
    print(f"Custom PCA result (first 5):\n{X_pca_custom[:5]}")
    print(f"Sklearn PCA result (first 5):\n{X_pca_sklearn[:5]}")

    # Check if they're the same (up to sign flip)
    diff = np.abs(X_pca_custom) - np.abs(X_pca_sklearn)
    print(f"\nMax absolute difference: {np.max(np.abs(diff)):.10f}")

    return X_pca_sklearn, y, pca_model, X, feature_names


def visualize_pca_results(X_pca, y, pca_model, X_original, feature_names):
    """Visualize PCA results"""
    print("\n" + "=" * 60)
    print("VISUALIZING PCA RESULTS")
    print("=" * 60)

    fig = plt.figure(figsize=(15, 5))

    # Plot 1: PCA projection (2D)
    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis',
                         edgecolors='k', s=50, alpha=0.7)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('PCA Projection (2D)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Class')

    # Plot 2: Explained variance
    ax2 = fig.add_subplot(132)
    explained_var = pca_model.explained_variance_ratio_
    cumsum_var = np.cumsum(explained_var)

    x_pos = np.arange(len(explained_var))
    ax2.bar(x_pos, explained_var, alpha=0.7, label='Individual')
    ax2.plot(x_pos, cumsum_var, 'ro-', label='Cumulative')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('Explained Variance by Component')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'PC{i+1}' for i in x_pos])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Component loadings
    ax3 = fig.add_subplot(133)
    components = pca_model.components_.T  # Transpose to get (features, components)

    x_pos = np.arange(len(feature_names))
    width = 0.35

    bars1 = ax3.bar(x_pos - width/2, components[:, 0], width,
                    label='PC1', alpha=0.7)
    bars2 = ax3.bar(x_pos + width/2, components[:, 1], width,
                    label='PC2', alpha=0.7)

    ax3.set_xlabel('Features')
    ax3.set_ylabel('Loading')
    ax3.set_title('Principal Component Loadings')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([name.split()[0] for name in feature_names], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Math_for_AI/pca_visualization.png', dpi=150)
    print("Visualization saved to pca_visualization.png")
    plt.close()


def svd_vs_pca():
    """Show relationship between SVD and PCA"""
    print("\n" + "=" * 60)
    print("RELATIONSHIP BETWEEN SVD AND PCA")
    print("=" * 60)

    # Generate random data
    np.random.seed(42)
    X = np.random.randn(100, 5)

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    print(f"Data shape: {X.shape}")

    # Method 1: PCA via eigendecomposition of covariance
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print("\n--- PCA via Eigendecomposition ---")
    print(f"Eigenvalues: {eigenvalues}")

    # Method 2: PCA via SVD
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Singular values squared relate to eigenvalues
    eigenvalues_from_svd = (s**2) / (X.shape[0] - 1)

    print("\n--- PCA via SVD ---")
    print(f"Eigenvalues from SVD: {eigenvalues_from_svd}")

    # V^T from SVD gives principal components
    print("\n--- Comparison ---")
    print(f"Eigenvalues match: {np.allclose(eigenvalues, eigenvalues_from_svd)}")
    print(f"Eigenvectors match (up to sign): {np.allclose(np.abs(eigenvectors), np.abs(Vt.T))}")

    print("\nKey insight:")
    print("- SVD of centered data X = U @ S @ V^T")
    print("- V^T contains principal components (eigenvectors of X^T X)")
    print("- Singular values s relate to eigenvalues: eigenvalue = s^2 / (n-1)")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_svd()

    print("\n")
    X_pca, y, pca_model, X_original, feature_names = demonstrate_pca_on_iris()

    visualize_pca_results(X_pca, y, pca_model, X_original, feature_names)

    svd_vs_pca()

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
