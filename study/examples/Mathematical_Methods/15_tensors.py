"""
Tensors - Operations, Einstein Summation, and Coordinate Transformations

This script demonstrates:
- Tensor operations with numpy
- Index notation and Einstein summation (np.einsum)
- Metric tensor
- Christoffel symbols
- Coordinate transformations
- Tensor contraction
- Tensor products
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def kronecker_delta(n):
    """Kronecker delta tensor δ_ij"""
    return np.eye(n)


def levi_civita_3d():
    """
    Levi-Civita (permutation) tensor ε_ijk in 3D
    ε_ijk = +1 for even permutations, -1 for odd, 0 otherwise
    """
    epsilon = np.zeros((3, 3, 3))
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
    return epsilon


def metric_tensor_euclidean_2d():
    """Euclidean metric in 2D: g_ij = δ_ij"""
    return np.eye(2)


def metric_tensor_polar():
    """
    Metric tensor in polar coordinates (r, θ)
    ds² = dr² + r²dθ²
    g_ij = [[1, 0], [0, r²]]
    """
    def g(r):
        return np.array([[1, 0], [0, r**2]])
    return g


def metric_tensor_spherical():
    """
    Metric tensor in spherical coordinates (r, θ, φ)
    ds² = dr² + r²dθ² + r²sin²(θ)dφ²
    """
    def g(r, theta):
        return np.array([
            [1, 0, 0],
            [0, r**2, 0],
            [0, 0, r**2 * np.sin(theta)**2]
        ])
    return g


def christoffel_symbols_polar(r):
    """
    Compute Christoffel symbols Γ^k_ij for polar coordinates
    Γ^r_θθ = -r
    Γ^θ_rθ = Γ^θ_θr = 1/r
    """
    Gamma = np.zeros((2, 2, 2))

    # Γ^r_θθ = -r (component [0, 1, 1])
    Gamma[0, 1, 1] = -r

    # Γ^θ_rθ = Γ^θ_θr = 1/r (components [1, 0, 1] and [1, 1, 0])
    Gamma[1, 0, 1] = 1 / r
    Gamma[1, 1, 0] = 1 / r

    return Gamma


def christoffel_symbols_spherical(r, theta):
    """
    Compute Christoffel symbols Γ^k_ij for spherical coordinates
    """
    Gamma = np.zeros((3, 3, 3))

    # Non-zero components
    # Γ^r_θθ = -r
    Gamma[0, 1, 1] = -r

    # Γ^r_φφ = -r sin²(θ)
    Gamma[0, 2, 2] = -r * np.sin(theta)**2

    # Γ^θ_rθ = Γ^θ_θr = 1/r
    Gamma[1, 0, 1] = 1 / r
    Gamma[1, 1, 0] = 1 / r

    # Γ^θ_φφ = -sin(θ)cos(θ)
    Gamma[1, 2, 2] = -np.sin(theta) * np.cos(theta)

    # Γ^φ_rφ = Γ^φ_φr = 1/r
    Gamma[2, 0, 2] = 1 / r
    Gamma[2, 2, 0] = 1 / r

    # Γ^φ_θφ = Γ^φ_φθ = cot(θ)
    if np.sin(theta) != 0:
        Gamma[2, 1, 2] = np.cos(theta) / np.sin(theta)
        Gamma[2, 2, 1] = np.cos(theta) / np.sin(theta)

    return Gamma


def coordinate_transform_cartesian_to_polar(x, y):
    """Transform from Cartesian to polar coordinates"""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def coordinate_transform_polar_to_cartesian(r, theta):
    """Transform from polar to Cartesian coordinates"""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def jacobian_cartesian_to_polar(x, y):
    """
    Jacobian matrix for Cartesian to polar transformation
    J_ij = ∂x^i_new / ∂x^j_old
    """
    r = np.sqrt(x**2 + y**2)
    if r == 0:
        return np.eye(2)

    # dr/dx, dr/dy
    # dθ/dx, dθ/dy
    J = np.array([
        [x/r, y/r],
        [-y/(r**2), x/(r**2)]
    ])
    return J


def transform_vector_contravariant(v, J):
    """
    Transform contravariant vector: v'^i = J^i_j v^j
    """
    return J @ v


def transform_covector_covariant(w, J):
    """
    Transform covariant vector (covector): w'_i = (J^T)^(-1)_ij w_j
    """
    return np.linalg.inv(J.T) @ w


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

print("=" * 70)
print("TENSORS - OPERATIONS, EINSTEIN SUMMATION, TRANSFORMATIONS")
print("=" * 70)

# Test 1: Basic tensor operations
print("\n1. BASIC TENSOR OPERATIONS")
print("-" * 70)

# Rank-2 tensors (matrices)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Tensor A:")
print(A)
print("\nTensor B:")
print(B)

# Tensor addition
print("\nA + B:")
print(A + B)

# Tensor product (outer product)
C = np.outer(A.flatten(), B.flatten()).reshape(2, 2, 2, 2)
print(f"\nTensor product A ⊗ B shape: {C.shape}")

# Trace (contraction)
trace_A = np.trace(A)
print(f"\nTrace of A (contraction): {trace_A}")

# Test 2: Einstein summation
print("\n2. EINSTEIN SUMMATION CONVENTION (np.einsum)")
print("-" * 70)

# Matrix multiplication: C_ij = A_ik B_kj
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C_matmul = np.einsum('ik,kj->ij', A, B)
C_direct = A @ B

print("Matrix multiplication: C = A @ B")
print("Using einsum:")
print(C_matmul)
print("Direct computation:")
print(C_direct)
print(f"Match: {np.allclose(C_matmul, C_direct)}")

# Trace: tr(A) = A_ii
trace_einsum = np.einsum('ii->', A)
print(f"\nTrace using einsum: {trace_einsum}")
print(f"Trace using np.trace: {np.trace(A)}")

# Vector dot product: c = a_i b_i
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot_einsum = np.einsum('i,i->', a, b)
dot_direct = np.dot(a, b)

print(f"\nDot product a·b:")
print(f"  Using einsum: {dot_einsum}")
print(f"  Using np.dot:  {dot_direct}")

# Outer product: C_ij = a_i b_j
outer_einsum = np.einsum('i,j->ij', a, b)
outer_direct = np.outer(a, b)

print(f"\nOuter product a ⊗ b:")
print("Using einsum:")
print(outer_einsum)
print("Using np.outer:")
print(outer_direct)

# Batch matrix multiplication: D_ijk = A_ij B_jk for multiple matrices
n_batch = 3
A_batch = np.random.rand(n_batch, 2, 3)
B_batch = np.random.rand(n_batch, 3, 2)

D_batch = np.einsum('nij,njk->nik', A_batch, B_batch)
print(f"\nBatch matrix multiplication shape: {D_batch.shape}")

# Test 3: Metric tensor
print("\n3. METRIC TENSOR")
print("-" * 70)

# Euclidean 2D
g_euclidean = metric_tensor_euclidean_2d()
print("Euclidean metric (Cartesian):")
print(g_euclidean)

# Polar coordinates
r = 2.0
g_polar_func = metric_tensor_polar()
g_polar = g_polar_func(r)
print(f"\nPolar metric at r={r}:")
print(g_polar)

# Line element ds²
dr = 0.1
dtheta = 0.05
ds_squared = g_polar[0, 0] * dr**2 + g_polar[1, 1] * dtheta**2
print(f"\nLine element ds² = dr² + r²dθ²")
print(f"For dr={dr}, dθ={dtheta}, r={r}:")
print(f"ds² = {ds_squared:.6f}")
print(f"ds = {np.sqrt(ds_squared):.6f}")

# Test 4: Christoffel symbols
print("\n4. CHRISTOFFEL SYMBOLS")
print("-" * 70)

r = 2.0
Gamma_polar = christoffel_symbols_polar(r)

print(f"Polar coordinates at r={r}:")
print(f"Γ^r_θθ = {Gamma_polar[0, 1, 1]:.4f} (expected: {-r})")
print(f"Γ^θ_rθ = {Gamma_polar[1, 0, 1]:.4f} (expected: {1/r:.4f})")
print(f"Γ^θ_θr = {Gamma_polar[1, 1, 0]:.4f} (expected: {1/r:.4f})")

# Spherical coordinates
r, theta = 3.0, np.pi / 4
Gamma_spherical = christoffel_symbols_spherical(r, theta)

print(f"\nSpherical coordinates at r={r}, θ={np.degrees(theta):.1f}°:")
print(f"Γ^r_θθ = {Gamma_spherical[0, 1, 1]:.4f} (expected: {-r})")
print(f"Γ^r_φφ = {Gamma_spherical[0, 2, 2]:.4f} (expected: {-r*np.sin(theta)**2:.4f})")
print(f"Γ^θ_rθ = {Gamma_spherical[1, 0, 1]:.4f} (expected: {1/r:.4f})")

# Test 5: Coordinate transformations
print("\n5. COORDINATE TRANSFORMATIONS")
print("-" * 70)

# Cartesian to polar
x, y = 3.0, 4.0
r, theta = coordinate_transform_cartesian_to_polar(x, y)

print(f"Cartesian: (x, y) = ({x}, {y})")
print(f"Polar: (r, θ) = ({r:.4f}, {np.degrees(theta):.2f}°)")

# Back to Cartesian
x_back, y_back = coordinate_transform_polar_to_cartesian(r, theta)
print(f"Back to Cartesian: ({x_back:.4f}, {y_back:.4f})")

# Jacobian
J = jacobian_cartesian_to_polar(x, y)
print(f"\nJacobian matrix (∂polar/∂cartesian):")
print(J)

# Verify det(J) = 1/r for area element transformation
det_J = np.linalg.det(J)
print(f"det(J) = {det_J:.6f}")
print(f"Expected 1/r = {1/r:.6f}")

# Test 6: Vector transformation
print("\n6. VECTOR TRANSFORMATION")
print("-" * 70)

# Velocity vector in Cartesian coordinates
v_cartesian = np.array([1.0, 2.0])
print(f"Velocity in Cartesian: v = {v_cartesian}")

# Transform to polar
x, y = 3.0, 4.0
J = jacobian_cartesian_to_polar(x, y)
v_polar = transform_vector_contravariant(v_cartesian, J)

print(f"Velocity in polar: v = {v_polar}")
print(f"  v^r = {v_polar[0]:.4f}")
print(f"  v^θ = {v_polar[1]:.4f}")

# Transform back
J_inverse = np.linalg.inv(J)
v_back = transform_vector_contravariant(v_polar, J_inverse)
print(f"Back to Cartesian: v = {v_back}")
print(f"Match: {np.allclose(v_cartesian, v_back)}")

# Test 7: Levi-Civita tensor and cross product
print("\n7. LEVI-CIVITA TENSOR AND CROSS PRODUCT")
print("-" * 70)

epsilon = levi_civita_3d()
print("Levi-Civita tensor ε_ijk (3D)")

# Cross product using Einstein summation
# (a × b)_i = ε_ijk a_j b_k
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])

cross_einsum = np.einsum('ijk,j,k->i', epsilon, a, b)
cross_direct = np.cross(a, b)

print(f"\na = {a}")
print(f"b = {b}")
print(f"a × b (einsum):  {cross_einsum}")
print(f"a × b (np.cross): {cross_direct}")

# Verify ε_ijk ε_imn = δ_jm δ_kn - δ_jn δ_km
product = np.einsum('ijk,imn->jkmn', epsilon, epsilon)
delta = kronecker_delta(3)

expected = (np.einsum('jm,kn->jkmn', delta, delta) -
            np.einsum('jn,km->jkmn', delta, delta))

print(f"\nVerify ε_ijk ε_imn = δ_jm δ_kn - δ_jn δ_km:")
print(f"Match: {np.allclose(product, expected)}")

# Test 8: Raising and lowering indices
print("\n8. RAISING AND LOWERING INDICES")
print("-" * 70)

# In Euclidean space, g_ij = g^ij = δ_ij
g = metric_tensor_euclidean_2d()
g_inv = np.linalg.inv(g)

print("Metric tensor g_ij:")
print(g)
print("\nInverse metric g^ij:")
print(g_inv)

# Covariant vector (covector)
w_down = np.array([1, 2])
print(f"\nCovariant vector w_i = {w_down}")

# Raise index: w^i = g^ij w_j
w_up = g_inv @ w_down
print(f"Contravariant vector w^i = g^ij w_j = {w_up}")

# In Euclidean space, they're the same
print(f"Match (Euclidean): {np.allclose(w_down, w_up)}")

# In polar coordinates
r = 2.0
g_polar_func = metric_tensor_polar()
g_polar = g_polar_func(r)
g_polar_inv = np.linalg.inv(g_polar)

w_down_polar = np.array([1, 2])
w_up_polar = g_polar_inv @ w_down_polar

print(f"\nIn polar coordinates at r={r}:")
print(f"Covariant w_i = {w_down_polar}")
print(f"Contravariant w^i = {w_up_polar}")

# Visualization
if HAS_MATPLOTLIB:
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Coordinate grid transformation
    ax1 = plt.subplot(2, 3, 1)

    # Cartesian grid
    x = np.linspace(-3, 3, 7)
    y = np.linspace(-3, 3, 7)

    for xi in x:
        ax1.plot([xi, xi], [-3, 3], 'b-', alpha=0.5, linewidth=0.8)
    for yi in y:
        ax1.plot([-3, 3], [yi, yi], 'r-', alpha=0.5, linewidth=0.8)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Cartesian Grid')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Polar grid
    ax2 = plt.subplot(2, 3, 2)

    r_vals = np.linspace(0.5, 3, 6)
    theta_vals = np.linspace(0, 2*np.pi, 13)

    # Radial lines
    for theta in theta_vals:
        r_line = np.linspace(0, 3, 50)
        x_line = r_line * np.cos(theta)
        y_line = r_line * np.sin(theta)
        ax2.plot(x_line, y_line, 'b-', alpha=0.5, linewidth=0.8)

    # Circular lines
    for r in r_vals:
        theta_circle = np.linspace(0, 2*np.pi, 100)
        x_circle = r * np.cos(theta_circle)
        y_circle = r * np.sin(theta_circle)
        ax2.plot(x_circle, y_circle, 'r-', alpha=0.5, linewidth=0.8)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Polar Grid in Cartesian Space')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Vector transformation
    ax3 = plt.subplot(2, 3, 3)

    x0, y0 = 2.0, 1.5
    v_cart = np.array([1.0, 0.5])

    # Original vector
    ax3.arrow(x0, y0, v_cart[0], v_cart[1], head_width=0.15, head_length=0.1,
             fc='blue', ec='blue', linewidth=2, label='Cartesian')

    # Transform to polar basis
    J = jacobian_cartesian_to_polar(x0, y0)
    v_polar = transform_vector_contravariant(v_cart, J)

    # Polar basis vectors at (x0, y0)
    r0, theta0 = coordinate_transform_cartesian_to_polar(x0, y0)

    # e_r in Cartesian coordinates
    e_r = np.array([np.cos(theta0), np.sin(theta0)])
    # e_θ in Cartesian coordinates
    e_theta = np.array([-np.sin(theta0), np.cos(theta0)])

    # Reconstruct vector in Cartesian using polar components
    v_reconstructed = v_polar[0] * e_r + v_polar[1] * e_theta

    ax3.arrow(x0, y0, v_reconstructed[0], v_reconstructed[1],
             head_width=0.15, head_length=0.1, fc='red', ec='red',
             linewidth=2, alpha=0.5, linestyle='--', label='From polar')

    # Show basis vectors
    ax3.arrow(x0, y0, e_r[0], e_r[1], head_width=0.1, head_length=0.05,
             fc='green', ec='green', alpha=0.5, label='e_r')
    ax3.arrow(x0, y0, e_theta[0], e_theta[1], head_width=0.1, head_length=0.05,
             fc='orange', ec='orange', alpha=0.5, label='e_θ')

    ax3.plot(x0, y0, 'ko', markersize=8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Vector Transformation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.set_xlim(0, 5)
    ax3.set_ylim(0, 4)

    # Plot 4: Metric tensor components in polar
    ax4 = plt.subplot(2, 3, 4)

    r_range = np.linspace(0.5, 5, 100)
    g_rr = np.ones_like(r_range)
    g_theta_theta = r_range**2

    ax4.plot(r_range, g_rr, 'b-', linewidth=2, label='g_rr = 1')
    ax4.plot(r_range, g_theta_theta, 'r-', linewidth=2, label='g_θθ = r²')
    ax4.set_xlabel('r')
    ax4.set_ylabel('Metric component')
    ax4.set_title('Polar Metric Tensor Components')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Christoffel symbols
    ax5 = plt.subplot(2, 3, 5)

    r_range = np.linspace(0.5, 5, 100)
    Gamma_r_theta_theta = -r_range
    Gamma_theta_r_theta = 1 / r_range

    ax5.plot(r_range, Gamma_r_theta_theta, 'b-', linewidth=2, label='Γ^r_θθ = -r')
    ax5.plot(r_range, Gamma_theta_r_theta, 'r-', linewidth=2, label='Γ^θ_rθ = 1/r')
    ax5.set_xlabel('r')
    ax5.set_ylabel('Christoffel symbol')
    ax5.set_title('Polar Christoffel Symbols')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Cross product with Levi-Civita
    ax6 = plt.subplot(2, 3, 6, projection='3d')

    # Vectors
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    c = np.cross(a, b)

    origin = np.array([0, 0, 0])

    ax6.quiver(origin[0], origin[1], origin[2], a[0], a[1], a[2],
              color='red', arrow_length_ratio=0.2, linewidth=2, label='a')
    ax6.quiver(origin[0], origin[1], origin[2], b[0], b[1], b[2],
              color='blue', arrow_length_ratio=0.2, linewidth=2, label='b')
    ax6.quiver(origin[0], origin[1], origin[2], c[0], c[1], c[2],
              color='green', arrow_length_ratio=0.2, linewidth=2, label='a×b')

    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_zlabel('z')
    ax6.set_title('Cross Product: a × b')
    ax6.legend()
    ax6.set_xlim(-0.5, 1.5)
    ax6.set_ylim(-0.5, 1.5)
    ax6.set_zlim(-0.5, 1.5)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Mathematical_Methods/15_tensors.png', dpi=150)
    print("Saved visualization: 15_tensors.png")
    plt.close()

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
