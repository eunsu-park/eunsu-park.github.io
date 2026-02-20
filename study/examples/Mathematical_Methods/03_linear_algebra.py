"""
Linear Algebra - Matrices, Eigenvalues, and Decompositions

This script demonstrates:
- Matrix operations (addition, multiplication, transpose)
- Determinants and matrix inverse
- Eigenvalues and eigenvectors
- Matrix diagonalization
- Singular Value Decomposition (SVD)
- Solving linear systems
- Matrix exponential
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def matrix_determinant_2x2(A):
    """Compute determinant of 2x2 matrix manually"""
    if A.shape != (2, 2):
        raise ValueError("Matrix must be 2x2")
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


def matrix_inverse_2x2(A):
    """Compute inverse of 2x2 matrix manually"""
    det = matrix_determinant_2x2(A)
    if abs(det) < 1e-10:
        raise ValueError("Matrix is singular")
    return np.array([[A[1, 1], -A[0, 1]],
                     [-A[1, 0], A[0, 0]]]) / det


def power_iteration(A, num_iterations=100):
    """Find dominant eigenvalue and eigenvector using power iteration"""
    n = A.shape[0]
    b = np.random.rand(n)

    for _ in range(num_iterations):
        b_new = A @ b
        b_new_norm = np.linalg.norm(b_new)
        b = b_new / b_new_norm

    eigenvalue = (b @ (A @ b)) / (b @ b)
    return eigenvalue, b


def gram_schmidt(A):
    """Gram-Schmidt orthogonalization of column vectors"""
    Q = np.zeros_like(A, dtype=float)

    for i in range(A.shape[1]):
        q = A[:, i].astype(float)
        for j in range(i):
            q -= np.dot(Q[:, j], A[:, i]) * Q[:, j]
        Q[:, i] = q / np.linalg.norm(q)

    return Q


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

print("=" * 70)
print("LINEAR ALGEBRA - MATRICES, EIGENVALUES, AND DECOMPOSITIONS")
print("=" * 70)

# Test 1: Matrix operations
print("\n1. MATRIX OPERATIONS")
print("-" * 70)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
print(f"\nA + B:\n{A + B}")
print(f"\nA * B (element-wise):\n{A * B}")
print(f"\nA @ B (matrix multiplication):\n{A @ B}")
print(f"\nA^T (transpose):\n{A.T}")
print(f"\nTrace(A): {np.trace(A)}")

# Test 2: Determinants
print("\n2. DETERMINANTS")
print("-" * 70)
A = np.array([[1, 2], [3, 4]])
det_manual = matrix_determinant_2x2(A)
det_numpy = np.linalg.det(A)
print(f"Matrix A:\n{A}")
print(f"det(A) manual: {det_manual}")
print(f"det(A) numpy: {det_numpy:.6f}")

A3 = np.array([[1, 2, 3], [0, 4, 5], [1, 0, 6]])
print(f"\nMatrix A (3x3):\n{A3}")
print(f"det(A): {np.linalg.det(A3):.6f}")

# Test 3: Matrix inverse
print("\n3. MATRIX INVERSE")
print("-" * 70)
A = np.array([[1, 2], [3, 4]], dtype=float)
A_inv_manual = matrix_inverse_2x2(A)
A_inv_numpy = np.linalg.inv(A)
print(f"Matrix A:\n{A}")
print(f"\nA^(-1) manual:\n{A_inv_manual}")
print(f"\nA^(-1) numpy:\n{A_inv_numpy}")
print(f"\nVerification A @ A^(-1):\n{A @ A_inv_numpy}")

# Test 4: Solving linear systems
print("\n4. SOLVING LINEAR SYSTEMS Ax = b")
print("-" * 70)
A = np.array([[3, 1], [1, 2]], dtype=float)
b = np.array([9, 8], dtype=float)
print(f"Matrix A:\n{A}")
print(f"Vector b: {b}")

# Method 1: Using inverse
x_inv = np.linalg.inv(A) @ b
print(f"\nSolution (using inverse): x = {x_inv}")

# Method 2: Using solve
x_solve = np.linalg.solve(A, b)
print(f"Solution (using solve): x = {x_solve}")

# Verification
print(f"Verification Ax = {A @ x_solve}")
print(f"Expected b = {b}")

# Test 5: Eigenvalues and eigenvectors
print("\n5. EIGENVALUES AND EIGENVECTORS")
print("-" * 70)
A = np.array([[4, 1], [2, 3]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Matrix A:\n{A}")
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nEigenvectors:")
print(eigenvectors)

# Verification: A v = λ v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    Av = A @ v
    lam_v = lam * v
    print(f"\nEigenvalue λ_{i+1} = {lam:.4f}")
    print(f"A v_{i+1} = {Av}")
    print(f"λ_{i+1} v_{i+1} = {lam_v}")

# Test 6: Power iteration
print("\n6. POWER ITERATION FOR DOMINANT EIGENVALUE")
print("-" * 70)
A = np.array([[4, 1], [2, 3]], dtype=float)
lam_power, v_power = power_iteration(A, 50)
print(f"Matrix A:\n{A}")
print(f"\nDominant eigenvalue (power iteration): {lam_power:.6f}")
print(f"Dominant eigenvector: {v_power}")
print(f"\nNumPy eigenvalues: {np.linalg.eigvals(A)}")

# Test 7: Matrix diagonalization
print("\n7. MATRIX DIAGONALIZATION")
print("-" * 70)
A = np.array([[4, 1], [2, 3]], dtype=float)
eigenvalues, P = np.linalg.eig(A)
D = np.diag(eigenvalues)
print(f"Matrix A:\n{A}")
print(f"\nDiagonal matrix D:\n{D}")
print(f"\nEigenvector matrix P:\n{P}")
print(f"\nReconstruction P @ D @ P^(-1):\n{P @ D @ np.linalg.inv(P)}")

# Test 8: Singular Value Decomposition
print("\n8. SINGULAR VALUE DECOMPOSITION (SVD)")
print("-" * 70)
A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
U, s, Vt = np.linalg.svd(A)
print(f"Matrix A ({A.shape[0]}x{A.shape[1]}):")
print(A)
print(f"\nLeft singular vectors U ({U.shape[0]}x{U.shape[1]}):")
print(U)
print(f"\nSingular values: {s}")
print(f"\nRight singular vectors V^T ({Vt.shape[0]}x{Vt.shape[1]}):")
print(Vt)

# Reconstruct A
S = np.zeros((A.shape[0], A.shape[1]))
S[:min(A.shape), :min(A.shape)] = np.diag(s)
A_reconstructed = U @ S @ Vt
print(f"\nReconstructed A = U @ S @ V^T:")
print(A_reconstructed)

# Test 9: Matrix rank and condition number
print("\n9. MATRIX RANK AND CONDITION NUMBER")
print("-" * 70)
A_full = np.array([[1, 2], [3, 4]], dtype=float)
A_singular = np.array([[1, 2], [2, 4]], dtype=float)

print(f"Full rank matrix:\n{A_full}")
print(f"Rank: {np.linalg.matrix_rank(A_full)}")
print(f"Condition number: {np.linalg.cond(A_full):.4f}")

print(f"\nRank-deficient matrix:\n{A_singular}")
print(f"Rank: {np.linalg.matrix_rank(A_singular)}")
print(f"Condition number: {np.linalg.cond(A_singular):.4e}")

# Test 10: Gram-Schmidt orthogonalization
print("\n10. GRAM-SCHMIDT ORTHOGONALIZATION")
print("-" * 70)
A = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
Q = gram_schmidt(A)
print(f"Original matrix A:")
print(A)
print(f"\nOrthogonalized matrix Q:")
print(Q)
print(f"\nQ^T @ Q (should be identity):")
print(Q.T @ Q)

# Test 11: Matrix exponential
print("\n11. MATRIX EXPONENTIAL")
print("-" * 70)
A = np.array([[0, 1], [-1, 0]], dtype=float)
print(f"Matrix A (rotation matrix generator):")
print(A)

# Compute using scipy if available
try:
    from scipy.linalg import expm
    exp_A = expm(A)
    print(f"\nexp(A) using scipy:")
    print(exp_A)
    print(f"\nThis is a rotation by 1 radian:")
    print(f"cos(1) = {np.cos(1):.6f}, sin(1) = {np.sin(1):.6f}")
except ImportError:
    # Manual computation using Taylor series
    exp_A = np.eye(2)
    A_power = np.eye(2)
    for n in range(1, 20):
        A_power = A_power @ A
        exp_A += A_power / np.math.factorial(n)
    print(f"\nexp(A) using Taylor series (20 terms):")
    print(exp_A)

# Visualization
if HAS_MATPLOTLIB:
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Plot 1: Eigenvector visualization
    ax = axes[0, 0]
    A = np.array([[3, 1], [1, 3]], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Draw eigenvectors
    origin = [0, 0]
    for i in range(2):
        v = eigenvectors[:, i]
        ax.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.1,
                fc=f'C{i}', ec=f'C{i}', label=f'v{i+1} (λ={eigenvalues[i]:.1f})')
        # Show transformation
        Av = A @ v
        ax.arrow(0, 0, Av[0], Av[1], head_width=0.1, head_length=0.1,
                fc=f'C{i}', ec=f'C{i}', alpha=0.3, linestyle='--')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Eigenvectors and Av')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_aspect('equal')

    # Plot 2: SVD visualization
    ax = axes[0, 1]
    A = np.array([[3, 1], [1, 2]], dtype=float)
    U, s, Vt = np.linalg.svd(A)

    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    ellipse = A @ circle

    ax.plot(circle[0], circle[1], 'b-', label='Unit circle', linewidth=2)
    ax.plot(ellipse[0], ellipse[1], 'r-', label='Transformed', linewidth=2)

    # Draw singular vectors
    for i in range(2):
        v = Vt[i, :]
        ax.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.1,
                fc='blue', ec='blue', alpha=0.5)
        u = U[:, i] * s[i]
        ax.arrow(0, 0, u[0], u[1], head_width=0.1, head_length=0.1,
                fc='red', ec='red', alpha=0.5)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('SVD: Circle to Ellipse')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 3: Matrix condition number effect
    ax = axes[0, 2]
    cond_numbers = []
    errors = []

    for epsilon in np.logspace(-8, -1, 20):
        A = np.array([[1, 1], [1, 1+epsilon]], dtype=float)
        b = np.array([2, 2+epsilon], dtype=float)
        x = np.linalg.solve(A, b)

        # True solution is [1, 1]
        error = np.linalg.norm(x - np.array([1, 1]))
        cond = np.linalg.cond(A)

        cond_numbers.append(cond)
        errors.append(error)

    ax.loglog(cond_numbers, errors, 'bo-', markersize=4)
    ax.set_xlabel('Condition number')
    ax.set_ylabel('Solution error')
    ax.set_title('Condition Number vs Error')
    ax.grid(True, alpha=0.3, which='both')

    # Plot 4: Power iteration convergence
    ax = axes[1, 0]
    A = np.array([[4, 1], [2, 3]], dtype=float)

    b = np.random.rand(2)
    eigenvalues_history = []

    for i in range(50):
        b = A @ b
        b = b / np.linalg.norm(b)
        lam = (b @ (A @ b)) / (b @ b)
        eigenvalues_history.append(lam)

    true_eigenvalue = max(np.linalg.eigvals(A))

    ax.plot(eigenvalues_history, 'b-', linewidth=2, label='Power iteration')
    ax.axhline(y=true_eigenvalue, color='r', linestyle='--', label=f'True λ = {true_eigenvalue:.4f}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Eigenvalue estimate')
    ax.set_title('Power Iteration Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Gram-Schmidt visualization
    ax = axes[1, 1]
    v1 = np.array([3, 1])
    v2 = np.array([1, 2])

    # Original vectors
    ax.arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.2,
            fc='blue', ec='blue', label='v1', linewidth=2)
    ax.arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.2,
            fc='red', ec='red', label='v2', linewidth=2, alpha=0.5)

    # Orthogonalized vectors
    A = np.column_stack([v1, v2])
    Q = gram_schmidt(A)
    q1, q2 = Q[:, 0], Q[:, 1]

    ax.arrow(0, 0, q1[0], q1[1], head_width=0.2, head_length=0.2,
            fc='blue', ec='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.arrow(0, 0, q2[0], q2[1], head_width=0.2, head_length=0.2,
            fc='green', ec='green', label='q2 (orthogonal)', linewidth=2)

    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Gram-Schmidt Orthogonalization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 6: Matrix exponential for rotation
    ax = axes[1, 2]
    theta_vals = np.linspace(0, 2*np.pi, 8, endpoint=False)

    for theta in theta_vals:
        A = np.array([[0, -theta], [theta, 0]])
        try:
            from scipy.linalg import expm
            R = expm(A)
        except ImportError:
            # Manual exponential
            R = np.eye(2)
            A_power = np.eye(2)
            for n in range(1, 20):
                A_power = A_power @ A
                R += A_power / np.math.factorial(n)

        v = np.array([1, 0])
        v_rotated = R @ v
        ax.arrow(0, 0, v_rotated[0], v_rotated[1], head_width=0.1,
                head_length=0.1, fc='blue', ec='blue', alpha=0.6)

    circle_theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(circle_theta), np.sin(circle_theta), 'r--', alpha=0.3)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Matrix Exponential Rotations')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Mathematical_Methods/03_linear_algebra.png', dpi=150)
    print("Saved visualization: 03_linear_algebra.png")
    plt.close()

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
