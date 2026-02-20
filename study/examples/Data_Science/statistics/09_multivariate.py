"""
09_multivariate.py

Demonstrates multivariate statistical analysis:
- PCA (from scratch and sklearn)
- Factor analysis concept
- Multivariate normal distribution
- Mahalanobis distance
- Canonical correlation
"""

import numpy as np
from scipy import stats

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("matplotlib not available; skipping plots\n")


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def pca_from_scratch():
    """Demonstrate PCA from scratch."""
    print_section("1. Principal Component Analysis (From Scratch)")

    # Generate correlated data
    np.random.seed(42)
    n = 200

    # True principal components
    mean = np.array([5, 3])
    # Data with specific correlation structure
    x1 = np.random.normal(0, 3, n)
    x2 = np.random.normal(0, 1, n)

    # Rotate to create correlation
    angle = np.pi / 4  # 45 degrees
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    X_rotated = (np.column_stack([x1, x2]) @ rotation.T) + mean

    print(f"Generated data: n = {n}, p = 2")
    print(f"Mean: {np.mean(X_rotated, axis=0)}")

    # Center the data
    X_centered = X_rotated - np.mean(X_rotated, axis=0)

    # Covariance matrix
    cov_matrix = np.cov(X_centered.T)

    print(f"\nCovariance matrix:")
    print(cov_matrix)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors (principal components):")
    print(eigenvectors)

    # Variance explained
    total_var = np.sum(eigenvalues)
    var_explained = eigenvalues / total_var

    print(f"\nVariance explained:")
    for i, var in enumerate(var_explained):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")

    # Project data onto principal components
    X_pca = X_centered @ eigenvectors

    print(f"\nPCA scores statistics:")
    print(f"  PC1: mean={np.mean(X_pca[:,0]):.4f}, std={np.std(X_pca[:,0], ddof=1):.4f}")
    print(f"  PC2: mean={np.mean(X_pca[:,1]):.4f}, std={np.std(X_pca[:,1], ddof=1):.4f}")

    # Reconstruction
    X_reconstructed = X_pca @ eigenvectors.T + np.mean(X_rotated, axis=0)
    reconstruction_error = np.mean((X_rotated - X_reconstructed)**2)

    print(f"\nReconstruction error (using all PCs): {reconstruction_error:.6f}")

    if HAS_PLT:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Original data
        axes[0].scatter(X_rotated[:, 0], X_rotated[:, 1], alpha=0.6)
        axes[0].set_xlabel('X₁')
        axes[0].set_ylabel('X₂')
        axes[0].set_title('Original Data')
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')

        # Plot principal component directions
        origin = np.mean(X_rotated, axis=0)
        for i in range(2):
            direction = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2
            axes[0].arrow(origin[0], origin[1], direction[0], direction[1],
                         head_width=0.3, head_length=0.2, fc=f'C{i+1}', ec=f'C{i+1}',
                         linewidth=2, label=f'PC{i+1}')
        axes[0].legend()

        # PCA scores
        axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
        axes[1].axhline(0, color='black', linewidth=0.5)
        axes[1].axvline(0, color='black', linewidth=0.5)
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].set_title('PCA Scores')
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')

        plt.tight_layout()
        plt.savefig('/tmp/pca_scratch.png', dpi=100)
        print("\n[Plot saved to /tmp/pca_scratch.png]")
        plt.close()


def pca_dimensionality_reduction():
    """Demonstrate PCA for dimensionality reduction."""
    print_section("2. PCA for Dimensionality Reduction")

    # Generate high-dimensional data with low intrinsic dimensionality
    np.random.seed(123)
    n = 150
    k = 3  # True dimensionality

    # Generate in low dimension
    Z = np.random.randn(n, k)

    # Random projection to high dimension
    p = 10
    A = np.random.randn(k, p)
    X = Z @ A

    # Add small noise
    X += np.random.randn(n, p) * 0.5

    print(f"Data: n = {n}, p = {p}")
    print(f"True intrinsic dimension: k = {k}")

    # PCA
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Variance explained
    total_var = np.sum(eigenvalues)
    var_explained = eigenvalues / total_var
    cumulative_var = np.cumsum(var_explained)

    print(f"\nVariance explained by each PC:")
    for i in range(p):
        print(f"  PC{i+1:2d}: {var_explained[i]:.4f} (cumulative: {cumulative_var[i]:.4f})")

    # Find number of PCs for 95% variance
    n_components_95 = np.argmax(cumulative_var >= 0.95) + 1

    print(f"\nComponents needed for 95% variance: {n_components_95}")
    print(f"Dimension reduction: {p} → {n_components_95}")

    # Project to reduced dimension
    X_reduced = X_centered @ eigenvectors[:, :n_components_95]

    print(f"\nReduced representation shape: {X_reduced.shape}")

    # Reconstruction from reduced representation
    X_reconstructed = X_reduced @ eigenvectors[:, :n_components_95].T + np.mean(X, axis=0)
    reconstruction_error = np.mean((X - X_reconstructed)**2)

    print(f"Reconstruction error (using {n_components_95} PCs): {reconstruction_error:.4f}")

    if HAS_PLT:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, p + 1), var_explained, alpha=0.7, label='Individual')
        plt.plot(range(1, p + 1), cumulative_var, 'r-o', linewidth=2, label='Cumulative')
        plt.axhline(0.95, color='green', linestyle='--', label='95% threshold')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.title('PCA Scree Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/tmp/pca_scree.png', dpi=100)
        print("\n[Plot saved to /tmp/pca_scree.png]")
        plt.close()


def multivariate_normal():
    """Demonstrate multivariate normal distribution."""
    print_section("3. Multivariate Normal Distribution")

    # Define multivariate normal
    mean = np.array([2, 3])
    cov = np.array([
        [2.0, 0.8],
        [0.8, 1.5]
    ])

    print(f"Multivariate normal distribution")
    print(f"Mean: {mean}")
    print(f"\nCovariance matrix:")
    print(cov)

    # Correlation
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    print(f"\nCorrelation matrix:")
    print(corr)

    # Generate samples
    np.random.seed(456)
    n = 500
    samples = np.random.multivariate_normal(mean, cov, n)

    print(f"\nGenerated {n} samples")
    print(f"Sample mean: {np.mean(samples, axis=0)}")
    print(f"Sample covariance:")
    print(np.cov(samples.T))

    # Conditional distribution: X₁ | X₂ = x₂
    x2_given = 4.0

    # Conditional mean and variance
    mu1, mu2 = mean
    sigma11, sigma12, sigma22 = cov[0, 0], cov[0, 1], cov[1, 1]

    mu_cond = mu1 + (sigma12 / sigma22) * (x2_given - mu2)
    var_cond = sigma11 - sigma12**2 / sigma22

    print(f"\nConditional distribution X₁ | X₂={x2_given}:")
    print(f"  Mean: {mu_cond:.4f}")
    print(f"  Variance: {var_cond:.4f}")

    if HAS_PLT:
        plt.figure(figsize=(10, 8))

        # Scatter plot
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=20)

        # Confidence ellipse
        from matplotlib.patches import Ellipse
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(5.991 * eigenvalues)  # 95% confidence
        ellipse = Ellipse(mean, width, height, angle=angle,
                         facecolor='none', edgecolor='red', linewidth=2,
                         label='95% confidence ellipse')
        plt.gca().add_patch(ellipse)

        plt.scatter([mean[0]], [mean[1]], color='red', s=100, marker='x',
                   zorder=5, label='Mean')

        plt.xlabel('X₁')
        plt.ylabel('X₂')
        plt.title('Bivariate Normal Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.savefig('/tmp/multivariate_normal.png', dpi=100)
        print("\n[Plot saved to /tmp/multivariate_normal.png]")
        plt.close()


def mahalanobis_distance():
    """Demonstrate Mahalanobis distance."""
    print_section("4. Mahalanobis Distance")

    # Generate data
    np.random.seed(789)
    mean = np.array([0, 0])
    cov = np.array([
        [2.0, 1.2],
        [1.2, 1.5]
    ])

    n = 200
    X = np.random.multivariate_normal(mean, cov, n)

    print(f"Generated {n} points from bivariate normal")
    print(f"Mean: {mean}")

    # Add some outliers
    outliers = np.array([
        [5, 5],
        [-6, 4],
        [4, -5]
    ])

    X_with_outliers = np.vstack([X, outliers])

    print(f"\nAdded {len(outliers)} outliers")

    # Calculate Mahalanobis distance
    cov_inv = np.linalg.inv(cov)

    def mahalanobis(x, mean, cov_inv):
        """Calculate Mahalanobis distance."""
        diff = x - mean
        return np.sqrt(diff @ cov_inv @ diff)

    distances = np.array([mahalanobis(x, mean, cov_inv) for x in X_with_outliers])

    # Also calculate Euclidean distance for comparison
    euclidean_distances = np.sqrt(np.sum((X_with_outliers - mean)**2, axis=1))

    print(f"\nDistance statistics:")
    print(f"  Mahalanobis: mean={np.mean(distances[:n]):.2f}, std={np.std(distances[:n], ddof=1):.2f}")
    print(f"  Euclidean: mean={np.mean(euclidean_distances[:n]):.2f}, std={np.std(euclidean_distances[:n], ddof=1):.2f}")

    # Outlier detection using Mahalanobis distance
    # Chi-square threshold (95% for 2 dimensions)
    threshold = np.sqrt(stats.chi2.ppf(0.95, df=2))

    outlier_indices = np.where(distances > threshold)[0]

    print(f"\nOutlier detection (threshold={threshold:.2f}):")
    print(f"  Detected outliers: {len(outlier_indices)}")
    print(f"  True outliers start at index {n}")

    print(f"\nOutlier distances:")
    for idx in range(n, len(X_with_outliers)):
        print(f"  Point {idx}: Mahalanobis={distances[idx]:.2f}, Euclidean={euclidean_distances[idx]:.2f}")

    if HAS_PLT:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Mahalanobis distance
        scatter = axes[0].scatter(X_with_outliers[:n, 0], X_with_outliers[:n, 1],
                                 c=distances[:n], cmap='viridis', alpha=0.6, s=30)
        axes[0].scatter(X_with_outliers[n:, 0], X_with_outliers[n:, 1],
                       c='red', marker='x', s=200, linewidths=3, label='Outliers')
        axes[0].scatter([mean[0]], [mean[1]], c='red', marker='*', s=200, zorder=5)

        # Confidence ellipse
        from matplotlib.patches import Ellipse
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * threshold * np.sqrt(eigenvalues)
        ellipse = Ellipse(mean, width, height, angle=angle,
                         facecolor='none', edgecolor='red', linewidth=2, linestyle='--')
        axes[0].add_patch(ellipse)

        axes[0].set_xlabel('X₁')
        axes[0].set_ylabel('X₂')
        axes[0].set_title('Mahalanobis Distance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        fig.colorbar(scatter, ax=axes[0], label='Mahalanobis Distance')

        # Comparison histogram
        axes[1].hist(distances[:n], bins=30, alpha=0.7, label='Normal points', density=True)
        axes[1].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')

        for i, d in enumerate(distances[n:]):
            axes[1].axvline(d, color='red', alpha=0.5, linewidth=2)

        # Overlay chi-square distribution
        x = np.linspace(0, distances.max(), 200)
        axes[1].plot(x, stats.chi2.pdf(x**2, df=2) * 2 * x, 'black', linestyle='--',
                    linewidth=2, label='χ²(2) density')

        axes[1].set_xlabel('Mahalanobis Distance')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Distance Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/mahalanobis.png', dpi=100)
        print("\n[Plot saved to /tmp/mahalanobis.png]")
        plt.close()


def canonical_correlation():
    """Demonstrate canonical correlation analysis concept."""
    print_section("5. Canonical Correlation Analysis")

    # Generate two sets of variables with correlation
    np.random.seed(999)
    n = 150

    # Latent variable
    z = np.random.normal(0, 1, n)

    # First set of variables (related to z)
    X1 = z + np.random.normal(0, 0.5, n)
    X2 = 0.8 * z + np.random.normal(0, 0.7, n)
    X3 = np.random.normal(0, 1, n)  # Independent

    X = np.column_stack([X1, X2, X3])

    # Second set of variables (also related to z)
    Y1 = 0.9 * z + np.random.normal(0, 0.6, n)
    Y2 = z + np.random.normal(0, 0.5, n)
    Y3 = np.random.normal(0, 1, n)  # Independent

    Y = np.column_stack([Y1, Y2, Y3])

    print(f"Two sets of variables:")
    print(f"  X: {X.shape[1]} variables")
    print(f"  Y: {Y.shape[1]} variables")
    print(f"  Both related through latent variable z")

    # Simple approach: correlations between all pairs
    print(f"\nPairwise correlations (X vs Y):")
    print(f"       Y1      Y2      Y3")

    for i in range(3):
        row = []
        for j in range(3):
            corr = np.corrcoef(X[:, i], Y[:, j])[0, 1]
            row.append(f"{corr:7.3f}")
        print(f"  X{i+1} " + " ".join(row))

    # Simplified canonical correlation (first canonical variate)
    # Find linear combinations that maximize correlation

    # Center the data
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)

    # Covariance matrices
    Cxx = np.cov(X_centered.T)
    Cyy = np.cov(Y_centered.T)
    Cxy = np.cov(X_centered.T, Y_centered.T)[:3, 3:]

    # Canonical correlation via eigendecomposition
    # Solve: Cxx^(-1) Cxy Cyy^(-1) Cyx
    try:
        M = np.linalg.inv(Cxx) @ Cxy @ np.linalg.inv(Cyy) @ Cxy.T
        eigenvalues, eigenvectors = np.linalg.eig(M)

        # Canonical correlations are sqrt of eigenvalues
        canonical_corrs = np.sqrt(np.abs(eigenvalues))

        idx = canonical_corrs.argsort()[::-1]
        canonical_corrs = canonical_corrs[idx]

        print(f"\nCanonical correlations:")
        for i, cc in enumerate(canonical_corrs):
            print(f"  CC{i+1}: {cc:.4f}")

        print(f"\nFirst canonical correlation captures the latent relationship")

    except np.linalg.LinAlgError:
        print(f"\nSingular covariance matrix; CCA not computable with this approach")


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("  MULTIVARIATE ANALYSIS DEMONSTRATIONS")
    print("=" * 70)

    np.random.seed(42)

    pca_from_scratch()
    pca_dimensionality_reduction()
    multivariate_normal()
    mahalanobis_distance()
    canonical_correlation()

    print("\n" + "=" * 70)
    print("  All demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
