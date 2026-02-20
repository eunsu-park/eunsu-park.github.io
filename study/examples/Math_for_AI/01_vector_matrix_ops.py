"""
Vector and Matrix Operations for AI/ML

Demonstrates fundamental linear algebra concepts:
- Vector spaces: basis, span, linear independence
- Matrix operations: multiplication, transpose, inverse
- Rank computation
- ML applications: feature vectors, weight matrices

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def vector_space_basics():
    """Demonstrate vector space concepts: basis, span, linear independence"""
    print("=" * 60)
    print("VECTOR SPACE BASICS")
    print("=" * 60)

    # Standard basis in R^3
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    print("\nStandard basis vectors in R^3:")
    print(f"e1 = {e1}")
    print(f"e2 = {e2}")
    print(f"e3 = {e3}")

    # Any vector can be expressed as linear combination of basis
    v = np.array([3, -2, 5])
    print(f"\nVector v = {v}")
    print(f"v = 3*e1 + (-2)*e2 + 5*e3")
    reconstructed = 3*e1 + (-2)*e2 + 5*e3
    print(f"Reconstructed: {reconstructed}")
    print(f"Match: {np.allclose(v, reconstructed)}")

    # Linear independence check
    print("\n--- Linear Independence ---")
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    v3 = np.array([7, 8, 9])  # v3 is in span of v1, v2

    # Stack vectors as columns and check rank
    A = np.column_stack([v1, v2, v3])
    rank = np.linalg.matrix_rank(A)
    print(f"\nVectors: {v1}, {v2}, {v3}")
    print(f"Matrix A (vectors as columns):\n{A}")
    print(f"Rank(A) = {rank}")
    print(f"Linearly independent: {rank == 3} (expect False)")

    # Linearly independent set
    w1 = np.array([1, 0, 0])
    w2 = np.array([0, 1, 0])
    w3 = np.array([1, 1, 1])
    B = np.column_stack([w1, w2, w3])
    rank_B = np.linalg.matrix_rank(B)
    print(f"\nVectors: {w1}, {w2}, {w3}")
    print(f"Matrix B:\n{B}")
    print(f"Rank(B) = {rank_B}")
    print(f"Linearly independent: {rank_B == 3}")


def matrix_operations():
    """Demonstrate matrix multiplication, transpose, inverse"""
    print("\n" + "=" * 60)
    print("MATRIX OPERATIONS")
    print("=" * 60)

    # Matrix multiplication
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])

    print("\nMatrix A (2x3):")
    print(A)
    print("\nMatrix B (3x2):")
    print(B)

    C = A @ B  # Matrix multiplication
    print("\nC = A @ B (2x2):")
    print(C)
    print(f"Shape: {C.shape}")

    # Transpose
    print("\n--- Transpose ---")
    print(f"A^T (3x2):\n{A.T}")
    print(f"(A^T)^T == A: {np.allclose((A.T).T, A)}")

    # Inverse (square matrix)
    print("\n--- Matrix Inverse ---")
    D = np.array([[4, 7],
                  [2, 6]])
    print(f"Matrix D:\n{D}")

    D_inv = np.linalg.inv(D)
    print(f"\nD_inv:\n{D_inv}")

    # Verify D @ D_inv = I
    identity = D @ D_inv
    print(f"\nD @ D_inv:\n{identity}")
    print(f"Is identity: {np.allclose(identity, np.eye(2))}")

    # Determinant (must be non-zero for invertible)
    det_D = np.linalg.det(D)
    print(f"\ndet(D) = {det_D:.4f}")
    print(f"Invertible: {not np.isclose(det_D, 0)}")


def rank_computation():
    """Demonstrate rank computation and its significance"""
    print("\n" + "=" * 60)
    print("MATRIX RANK")
    print("=" * 60)

    # Full rank matrix
    print("\n--- Full Rank Matrix ---")
    A = np.array([[1, 2, 3],
                  [0, 1, 4],
                  [5, 6, 0]])
    rank_A = np.linalg.matrix_rank(A)
    print(f"Matrix A (3x3):\n{A}")
    print(f"Rank(A) = {rank_A}")
    print(f"Full rank: {rank_A == min(A.shape)}")

    # Rank deficient matrix (linearly dependent rows)
    print("\n--- Rank Deficient Matrix ---")
    B = np.array([[1, 2, 3],
                  [2, 4, 6],  # 2 * row1
                  [4, 5, 6]])
    rank_B = np.linalg.matrix_rank(B)
    print(f"Matrix B (3x3):\n{B}")
    print(f"Rank(B) = {rank_B}")
    print(f"Full rank: {rank_B == min(B.shape)}")
    print("Row 2 = 2 * Row 1, hence rank deficient")

    # Rectangular matrix
    print("\n--- Rectangular Matrix ---")
    C = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
    rank_C = np.linalg.matrix_rank(C)
    print(f"Matrix C (3x4):\n{C}")
    print(f"Rank(C) = {rank_C}")
    print(f"Max possible rank = min(3, 4) = {min(C.shape)}")


def ml_application_example():
    """ML application: feature matrix and weight matrix multiplication"""
    print("\n" + "=" * 60)
    print("ML APPLICATION: LINEAR MODEL")
    print("=" * 60)

    # Feature matrix X: each row is a sample, each column is a feature
    # 5 samples, 3 features
    X = np.array([[1.0, 2.0, 3.0],
                  [1.5, 2.5, 3.5],
                  [2.0, 3.0, 4.0],
                  [2.5, 3.5, 4.5],
                  [3.0, 4.0, 5.0]])

    # Weight matrix W: (3 features, 2 outputs)
    W = np.array([[0.5, -0.3],
                  [0.2, 0.8],
                  [-0.1, 0.4]])

    # Bias vector b: (2 outputs)
    b = np.array([1.0, -0.5])

    print(f"Feature matrix X (5 samples, 3 features):\n{X}")
    print(f"\nWeight matrix W (3 features, 2 outputs):\n{W}")
    print(f"\nBias vector b (2 outputs): {b}")

    # Linear transformation: Y = X @ W + b
    Y = X @ W + b
    print(f"\nOutput Y = X @ W + b (5 samples, 2 outputs):\n{Y}")

    print("\nInterpretation:")
    print("- Each row of X is a feature vector for one sample")
    print("- W maps 3D feature space to 2D output space")
    print("- Each row of Y is the model output for one sample")

    # Compute rank of feature matrix
    rank_X = np.linalg.matrix_rank(X)
    print(f"\nRank(X) = {rank_X}")
    print(f"Max rank = min(5, 3) = {min(X.shape)}")

    if rank_X < min(X.shape):
        print("Warning: Feature matrix is rank deficient!")
        print("Some features may be linearly dependent (redundant)")
    else:
        print("Feature matrix has full rank (good!)")


def visualize_vector_operations():
    """Visualize 2D vector operations"""
    print("\n" + "=" * 60)
    print("VISUALIZING VECTOR OPERATIONS")
    print("=" * 60)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Vector addition
    ax1 = axes[0]
    v1 = np.array([2, 1])
    v2 = np.array([1, 3])
    v_sum = v1 + v2

    origin = np.array([0, 0])
    ax1.quiver(*origin, *v1, angles='xy', scale_units='xy', scale=1,
               color='r', width=0.006, label='v1')
    ax1.quiver(*origin, *v2, angles='xy', scale_units='xy', scale=1,
               color='b', width=0.006, label='v2')
    ax1.quiver(*origin, *v_sum, angles='xy', scale_units='xy', scale=1,
               color='g', width=0.006, label='v1+v2')

    # Parallelogram
    ax1.plot([v1[0], v_sum[0]], [v1[1], v_sum[1]], 'k--', alpha=0.3)
    ax1.plot([v2[0], v_sum[0]], [v2[1], v_sum[1]], 'k--', alpha=0.3)

    ax1.set_xlim(-0.5, 4)
    ax1.set_ylim(-0.5, 5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Vector Addition')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Plot 2: Linear transformation (matrix multiplication)
    ax2 = axes[1]

    # Transformation matrix (rotation + scaling)
    theta = np.pi / 4  # 45 degrees
    A = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]]) * 1.5

    # Original vectors
    vectors = np.array([[1, 0], [0, 1], [1, 1]])

    for i, v in enumerate(vectors):
        v_transformed = A @ v

        # Original vector
        ax2.quiver(*origin, *v, angles='xy', scale_units='xy', scale=1,
                   color='blue', width=0.005, alpha=0.5)

        # Transformed vector
        ax2.quiver(*origin, *v_transformed, angles='xy', scale_units='xy', scale=1,
                   color='red', width=0.005, label='Transformed' if i == 0 else '')

    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(['Original (blue)', 'Transformed (red)'])
    ax2.set_title('Linear Transformation (Rotation + Scale)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Math_for_AI/vector_ops.png', dpi=150)
    print("Visualization saved to vector_ops.png")
    plt.close()


if __name__ == "__main__":
    # Run all demonstrations
    vector_space_basics()
    matrix_operations()
    rank_computation()
    ml_application_example()
    visualize_vector_operations()

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
