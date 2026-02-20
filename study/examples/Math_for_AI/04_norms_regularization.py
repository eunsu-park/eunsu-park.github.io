"""
Norms, Distance Metrics, and Regularization

Demonstrates:
- Lp norms visualization (L1, L2, L∞)
- Distance metrics (Euclidean, Manhattan, Cosine, Mahalanobis)
- L1 vs L2 regularization effects on linear regression
- Sparsity demonstration
- ML applications: feature scaling, regularization

Dependencies: numpy, matplotlib, sklearn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def demonstrate_lp_norms():
    """Demonstrate L1, L2, and L∞ norms"""
    print("=" * 60)
    print("Lp NORMS")
    print("=" * 60)

    print("\nLp norm: ||x||_p = (Σ|x_i|^p)^(1/p)")

    x = np.array([3, -4, 0, 2])

    # L1 norm (Manhattan): sum of absolute values
    l1_norm = np.linalg.norm(x, ord=1)
    l1_manual = np.sum(np.abs(x))

    # L2 norm (Euclidean): square root of sum of squares
    l2_norm = np.linalg.norm(x, ord=2)
    l2_manual = np.sqrt(np.sum(x**2))

    # L∞ norm (Maximum): maximum absolute value
    linf_norm = np.linalg.norm(x, ord=np.inf)
    linf_manual = np.max(np.abs(x))

    print(f"\nVector x = {x}")
    print(f"\nL1 norm (Manhattan):  ||x||_1 = {l1_norm:.4f} (manual: {l1_manual:.4f})")
    print(f"L2 norm (Euclidean):  ||x||_2 = {l2_norm:.4f} (manual: {l2_manual:.4f})")
    print(f"L∞ norm (Maximum):    ||x||_∞ = {linf_norm:.4f} (manual: {linf_manual:.4f})")

    # Properties
    print("\n--- Norm Properties ---")
    print("1. Non-negativity: ||x|| ≥ 0")
    print("2. Definiteness: ||x|| = 0 iff x = 0")
    print("3. Homogeneity: ||αx|| = |α| · ||x||")
    print("4. Triangle inequality: ||x + y|| ≤ ||x|| + ||y||")

    # Verify triangle inequality
    y = np.array([1, 2, -1, 3])
    lhs = np.linalg.norm(x + y, ord=2)
    rhs = np.linalg.norm(x, ord=2) + np.linalg.norm(y, ord=2)
    print(f"\nTriangle inequality verification (L2):")
    print(f"||x + y|| = {lhs:.4f}")
    print(f"||x|| + ||y|| = {rhs:.4f}")
    print(f"||x + y|| ≤ ||x|| + ||y||: {lhs <= rhs}")


def visualize_unit_balls():
    """Visualize unit balls for different norms"""
    print("\n" + "=" * 60)
    print("VISUALIZING UNIT BALLS")
    print("=" * 60)

    print("\nUnit ball: {x : ||x||_p ≤ 1}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Generate points
    theta = np.linspace(0, 2*np.pi, 1000)

    # L∞ norm: square
    ax = axes[0]
    square_x = np.array([1, 1, -1, -1, 1])
    square_y = np.array([1, -1, -1, 1, 1])
    ax.plot(square_x, square_y, 'b-', linewidth=2)
    ax.fill(square_x, square_y, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('L∞ Norm (||x||∞ ≤ 1)')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')

    # L2 norm: circle
    ax = axes[1]
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.plot(circle_x, circle_y, 'g-', linewidth=2)
    ax.fill(circle_x, circle_y, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('L2 Norm (||x||₂ ≤ 1)')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')

    # L1 norm: diamond
    ax = axes[2]
    diamond_x = np.array([1, 0, -1, 0, 1])
    diamond_y = np.array([0, 1, 0, -1, 0])
    ax.plot(diamond_x, diamond_y, 'r-', linewidth=2)
    ax.fill(diamond_x, diamond_y, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('L1 Norm (||x||₁ ≤ 1)')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Math_for_AI/unit_balls.png', dpi=150)
    print("Unit ball visualization saved to unit_balls.png")
    plt.close()


def distance_metrics():
    """Demonstrate various distance metrics"""
    print("\n" + "=" * 60)
    print("DISTANCE METRICS")
    print("=" * 60)

    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])

    print(f"Vector x: {x}")
    print(f"Vector y: {y}")

    # Euclidean distance (L2)
    euclidean = np.linalg.norm(x - y, ord=2)
    print(f"\n1. Euclidean distance (L2): {euclidean:.4f}")
    print(f"   d(x, y) = ||x - y||₂ = sqrt(Σ(x_i - y_i)²)")

    # Manhattan distance (L1)
    manhattan = np.linalg.norm(x - y, ord=1)
    print(f"\n2. Manhattan distance (L1): {manhattan:.4f}")
    print(f"   d(x, y) = ||x - y||₁ = Σ|x_i - y_i|")

    # Chebyshev distance (L∞)
    chebyshev = np.linalg.norm(x - y, ord=np.inf)
    print(f"\n3. Chebyshev distance (L∞): {chebyshev:.4f}")
    print(f"   d(x, y) = ||x - y||∞ = max|x_i - y_i|")

    # Cosine similarity/distance
    cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    cos_dist = 1 - cos_sim
    print(f"\n4. Cosine similarity: {cos_sim:.4f}")
    print(f"   cos(θ) = (x·y) / (||x|| ||y||)")
    print(f"   Cosine distance: {cos_dist:.4f}")

    # Mahalanobis distance
    print("\n5. Mahalanobis distance:")
    print("   Accounts for covariance structure")

    # Generate correlated data
    np.random.seed(42)
    mean = [0, 0]
    cov = [[2, 1], [1, 2]]  # Covariance matrix
    data = np.random.multivariate_normal(mean, cov, size=100)

    # Compute covariance and its inverse
    cov_matrix = np.cov(data.T)
    cov_inv = np.linalg.inv(cov_matrix)

    # Two points
    p1 = np.array([1, 1])
    p2 = np.array([2, 2])

    # Mahalanobis distance
    diff = p1 - p2
    mahal_dist = np.sqrt(diff.T @ cov_inv @ diff)

    print(f"   Point 1: {p1}")
    print(f"   Point 2: {p2}")
    print(f"   Mahalanobis distance: {mahal_dist:.4f}")
    print(f"   Euclidean distance:   {np.linalg.norm(p1 - p2):.4f}")


def regularization_comparison():
    """Compare L1 (Lasso) and L2 (Ridge) regularization"""
    print("\n" + "=" * 60)
    print("L1 vs L2 REGULARIZATION")
    print("=" * 60)

    # Generate synthetic data with many features
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    # Only first 5 features are truly relevant
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    true_coef[:5] = [3.0, -2.0, 1.5, -1.0, 2.5]

    y = X @ true_coef + np.random.randn(n_samples) * 0.5

    print(f"\nDataset:")
    print(f"Samples: {n_samples}, Features: {n_features}")
    print(f"True non-zero coefficients: 5")

    # Linear regression (no regularization)
    lr = LinearRegression()
    lr.fit(X, y)

    # Ridge (L2 regularization)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)

    # Lasso (L1 regularization)
    lasso = Lasso(alpha=0.1)
    lasso.fit(X, y)

    print("\n--- Coefficient Analysis ---")
    print(f"Linear Regression - Non-zero coefs: {np.sum(np.abs(lr.coef_) > 0.01)}")
    print(f"Ridge (L2) - Non-zero coefs: {np.sum(np.abs(ridge.coef_) > 0.01)}")
    print(f"Lasso (L1) - Non-zero coefs: {np.sum(np.abs(lasso.coef_) > 0.01)}")

    print(f"\nL2 norm of coefficients:")
    print(f"Linear Regression: {np.linalg.norm(lr.coef_, ord=2):.4f}")
    print(f"Ridge (L2):        {np.linalg.norm(ridge.coef_, ord=2):.4f}")
    print(f"Lasso (L1):        {np.linalg.norm(lasso.coef_, ord=2):.4f}")

    print(f"\nL1 norm of coefficients:")
    print(f"Linear Regression: {np.linalg.norm(lr.coef_, ord=1):.4f}")
    print(f"Ridge (L2):        {np.linalg.norm(ridge.coef_, ord=1):.4f}")
    print(f"Lasso (L1):        {np.linalg.norm(lasso.coef_, ord=1):.4f}")

    print("\nKey insights:")
    print("- L1 (Lasso) produces sparse solutions (many exact zeros)")
    print("- L2 (Ridge) shrinks coefficients but rarely to exactly zero")
    print("- L1 performs feature selection, L2 performs feature shrinkage")

    return X, y, lr, ridge, lasso, true_coef


def visualize_regularization(X, y, lr, ridge, lasso, true_coef):
    """Visualize regularization effects"""
    print("\n" + "=" * 60)
    print("VISUALIZING REGULARIZATION EFFECTS")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Coefficient values
    ax = axes[0, 0]
    x_pos = np.arange(len(true_coef))

    ax.plot(x_pos, true_coef, 'ko-', label='True', linewidth=2, markersize=4)
    ax.plot(x_pos, lr.coef_, 'b.-', label='Linear Reg', alpha=0.7, markersize=3)
    ax.plot(x_pos, ridge.coef_, 'g.-', label='Ridge (L2)', alpha=0.7, markersize=3)
    ax.plot(x_pos, lasso.coef_, 'r.-', label='Lasso (L1)', alpha=0.7, markersize=3)

    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Coefficient Values Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Plot 2: First 10 coefficients (zoomed)
    ax = axes[0, 1]
    n_show = 10
    x_pos_zoom = np.arange(n_show)

    width = 0.2
    ax.bar(x_pos_zoom - 1.5*width, true_coef[:n_show], width, label='True', alpha=0.8)
    ax.bar(x_pos_zoom - 0.5*width, lr.coef_[:n_show], width, label='Linear Reg', alpha=0.8)
    ax.bar(x_pos_zoom + 0.5*width, ridge.coef_[:n_show], width, label='Ridge (L2)', alpha=0.8)
    ax.bar(x_pos_zoom + 1.5*width, lasso.coef_[:n_show], width, label='Lasso (L1)', alpha=0.8)

    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Coefficient Value')
    ax.set_title(f'First {n_show} Coefficients (Zoomed)')
    ax.set_xticks(x_pos_zoom)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Plot 3: Sparsity pattern
    ax = axes[1, 0]

    threshold = 0.01
    sparsity_data = np.array([
        np.sum(np.abs(true_coef) > threshold),
        np.sum(np.abs(lr.coef_) > threshold),
        np.sum(np.abs(ridge.coef_) > threshold),
        np.sum(np.abs(lasso.coef_) > threshold)
    ])

    bars = ax.bar(['True', 'Linear Reg', 'Ridge (L2)', 'Lasso (L1)'],
                   sparsity_data, color=['black', 'blue', 'green', 'red'], alpha=0.7)

    ax.set_ylabel('Number of Non-Zero Coefficients')
    ax.set_title(f'Sparsity Comparison (threshold = {threshold})')
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, value in zip(bars, sparsity_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}', ha='center', va='bottom')

    # Plot 4: Regularization paths
    ax = axes[1, 1]

    # Compute coefficients for different alpha values
    alphas = np.logspace(-3, 2, 50)
    coefs_ridge = []
    coefs_lasso = []

    for alpha in alphas:
        ridge_temp = Ridge(alpha=alpha)
        ridge_temp.fit(X, y)
        coefs_ridge.append(ridge_temp.coef_)

        lasso_temp = Lasso(alpha=alpha)
        lasso_temp.fit(X, y)
        coefs_lasso.append(lasso_temp.coef_)

    coefs_ridge = np.array(coefs_ridge)
    coefs_lasso = np.array(coefs_lasso)

    # Plot first 5 features only
    for i in range(5):
        ax.plot(alphas, coefs_lasso[:, i], '-', linewidth=2, label=f'Feature {i}')

    ax.set_xscale('log')
    ax.set_xlabel('Alpha (Regularization Strength)')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Lasso Regularization Path (First 5 Features)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Math_for_AI/regularization_comparison.png', dpi=150)
    print("Regularization visualization saved to regularization_comparison.png")
    plt.close()


def demonstrate_sparsity():
    """Demonstrate sparsity-inducing property of L1 norm"""
    print("\n" + "=" * 60)
    print("SPARSITY WITH L1 REGULARIZATION")
    print("=" * 60)

    print("\nWhy L1 induces sparsity:")
    print("- L1 penalty: λΣ|w_i| has sharp corners at axes")
    print("- L2 penalty: λΣw_i² is smooth everywhere")
    print("- Optimization with L1 tends to hit corners (exact zeros)")

    # Simple 2D example
    print("\n--- 2D Example ---")

    # Generate data where only x1 is relevant
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 3*x1 + np.random.randn(n) * 0.5  # Only x1 is relevant

    X = np.column_stack([x1, x2])

    # Fit models
    lr = LinearRegression().fit(X, y)
    ridge = Ridge(alpha=1.0).fit(X, y)
    lasso = Lasso(alpha=0.1).fit(X, y)

    print(f"\nTrue relationship: y = 3*x1 + noise")
    print(f"\nLinear Regression coef: {lr.coef_}")
    print(f"Ridge (L2) coef:        {ridge.coef_}")
    print(f"Lasso (L1) coef:        {lasso.coef_}")

    print(f"\nLasso correctly identifies x2 as irrelevant!")
    print(f"x2 coefficient ≈ {lasso.coef_[1]:.6f} (effectively zero)")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_lp_norms()
    visualize_unit_balls()
    distance_metrics()

    X, y, lr, ridge, lasso, true_coef = regularization_comparison()
    visualize_regularization(X, y, lr, ridge, lasso, true_coef)

    demonstrate_sparsity()

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
