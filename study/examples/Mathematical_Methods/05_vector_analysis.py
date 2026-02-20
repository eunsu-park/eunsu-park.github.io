"""
Vector Analysis - Gradient, Divergence, Curl, and Integral Theorems

This script demonstrates:
- Gradient of scalar fields
- Divergence of vector fields
- Curl of vector fields
- Line integrals
- Surface integrals
- Green's theorem
- Stokes' theorem
- Divergence theorem verification
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def gradient_2d(f, x, y, h=1e-5):
    """Compute gradient of scalar field f using finite differences"""
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])


def gradient_3d(f, x, y, z, h=1e-5):
    """Compute gradient of scalar field f(x,y,z) using finite differences"""
    df_dx = (f(x + h, y, z) - f(x - h, y, z)) / (2 * h)
    df_dy = (f(x, y + h, z) - f(x, y - h, z)) / (2 * h)
    df_dz = (f(x, y, z + h) - f(x, y, z - h)) / (2 * h)
    return np.array([df_dx, df_dy, df_dz])


def divergence_2d(Fx, Fy, x, y, h=1e-5):
    """Compute divergence of 2D vector field F = (Fx, Fy)"""
    dFx_dx = (Fx(x + h, y) - Fx(x - h, y)) / (2 * h)
    dFy_dy = (Fy(x, y + h) - Fy(x, y - h)) / (2 * h)
    return dFx_dx + dFy_dy


def divergence_3d(Fx, Fy, Fz, x, y, z, h=1e-5):
    """Compute divergence of 3D vector field F = (Fx, Fy, Fz)"""
    dFx_dx = (Fx(x + h, y, z) - Fx(x - h, y, z)) / (2 * h)
    dFy_dy = (Fy(x, y + h, z) - Fy(x, y - h, z)) / (2 * h)
    dFz_dz = (Fz(x, y, z + h) - Fz(x, y, z - h)) / (2 * h)
    return dFx_dx + dFy_dy + dFz_dz


def curl_3d(Fx, Fy, Fz, x, y, z, h=1e-5):
    """Compute curl of 3D vector field F = (Fx, Fy, Fz)"""
    dFz_dy = (Fz(x, y + h, z) - Fz(x, y - h, z)) / (2 * h)
    dFy_dz = (Fy(x, y, z + h) - Fy(x, y, z - h)) / (2 * h)

    dFx_dz = (Fx(x, y, z + h) - Fx(x, y, z - h)) / (2 * h)
    dFz_dx = (Fz(x + h, y, z) - Fz(x - h, y, z)) / (2 * h)

    dFy_dx = (Fy(x + h, y, z) - Fy(x - h, y, z)) / (2 * h)
    dFx_dy = (Fx(x, y + h, z) - Fx(x, y - h, z)) / (2 * h)

    curl_x = dFz_dy - dFy_dz
    curl_y = dFx_dz - dFz_dx
    curl_z = dFy_dx - dFx_dy

    return np.array([curl_x, curl_y, curl_z])


def line_integral_2d(Fx, Fy, curve_x, curve_y, t_vals):
    """Compute line integral of F·dr along parametric curve"""
    integral = 0
    for i in range(len(t_vals) - 1):
        t = t_vals[i]
        dt = t_vals[i + 1] - t_vals[i]

        # Midpoint
        t_mid = t + dt / 2
        x_mid = curve_x(t_mid)
        y_mid = curve_y(t_mid)

        # Tangent vector dr/dt
        dx_dt = (curve_x(t_mid + 1e-5) - curve_x(t_mid - 1e-5)) / (2e-5)
        dy_dt = (curve_y(t_mid + 1e-5) - curve_y(t_mid - 1e-5)) / (2e-5)

        # F dot dr
        F_dot_dr = Fx(x_mid, y_mid) * dx_dt + Fy(x_mid, y_mid) * dy_dt
        integral += F_dot_dr * dt

    return integral


def surface_integral_divergence(Fx, Fy, Fz, x_range, y_range, z_val, n_points=20):
    """
    Compute surface integral of F·n over a horizontal surface z=z_val
    where n = (0, 0, 1) is the upward normal
    """
    x_vals = np.linspace(x_range[0], x_range[1], n_points)
    y_vals = np.linspace(y_range[0], y_range[1], n_points)

    integral = 0
    dx = (x_range[1] - x_range[0]) / (n_points - 1)
    dy = (y_range[1] - y_range[0]) / (n_points - 1)

    for x in x_vals:
        for y in y_vals:
            # F·n = Fz for upward normal (0,0,1)
            integral += Fz(x, y, z_val) * dx * dy

    return integral


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

print("=" * 70)
print("VECTOR ANALYSIS - GRADIENT, DIVERGENCE, CURL, AND THEOREMS")
print("=" * 70)

# Test 1: Gradient of scalar field
print("\n1. GRADIENT OF SCALAR FIELD")
print("-" * 70)
# f(x,y) = x^2 + y^2
f = lambda x, y: x**2 + y**2
x, y = 1.0, 2.0
grad_f = gradient_2d(f, x, y)
print(f"f(x,y) = x² + y²")
print(f"∇f at ({x}, {y}) = {grad_f}")
print(f"Expected: [2x, 2y] = [2, 4]")

# 3D example: f(x,y,z) = x^2 + y^2 + z^2
f_3d = lambda x, y, z: x**2 + y**2 + z**2
x, y, z = 1.0, 1.0, 1.0
grad_f_3d = gradient_3d(f_3d, x, y, z)
print(f"\nf(x,y,z) = x² + y² + z²")
print(f"∇f at ({x}, {y}, {z}) = {grad_f_3d}")
print(f"Expected: [2x, 2y, 2z] = [2, 2, 2]")

# Test 2: Divergence of vector field
print("\n2. DIVERGENCE OF VECTOR FIELD")
print("-" * 70)
# F = (x, y)
Fx = lambda x, y: x
Fy = lambda x, y: y
x, y = 1.0, 2.0
div_F = divergence_2d(Fx, Fy, x, y)
print(f"F(x,y) = (x, y)")
print(f"∇·F at ({x}, {y}) = {div_F:.6f}")
print(f"Expected: ∂x/∂x + ∂y/∂y = 2")

# 3D example: F = (x, y, z)
Fx_3d = lambda x, y, z: x
Fy_3d = lambda x, y, z: y
Fz_3d = lambda x, y, z: z
x, y, z = 1.0, 1.0, 1.0
div_F_3d = divergence_3d(Fx_3d, Fy_3d, Fz_3d, x, y, z)
print(f"\nF(x,y,z) = (x, y, z)")
print(f"∇·F at ({x}, {y}, {z}) = {div_F_3d:.6f}")
print(f"Expected: 3")

# Test 3: Curl of vector field
print("\n3. CURL OF VECTOR FIELD")
print("-" * 70)
# F = (-y, x, 0) - rotation field
Fx_rot = lambda x, y, z: -y
Fy_rot = lambda x, y, z: x
Fz_rot = lambda x, y, z: 0
x, y, z = 1.0, 0.0, 0.0
curl_F = curl_3d(Fx_rot, Fy_rot, Fz_rot, x, y, z)
print(f"F(x,y,z) = (-y, x, 0)")
print(f"∇×F at ({x}, {y}, {z}) = {curl_F}")
print(f"Expected: (0, 0, 2)")

# Conservative field: F = ∇(xy)
Fx_cons = lambda x, y, z: y
Fy_cons = lambda x, y, z: x
Fz_cons = lambda x, y, z: 0
curl_cons = curl_3d(Fx_cons, Fy_cons, Fz_cons, 1.0, 1.0, 0.0)
print(f"\nConservative field F = ∇(xy) = (y, x, 0)")
print(f"∇×F = {curl_cons}")
print(f"Expected: (0, 0, 0) for conservative field")

# Test 4: Line integral
print("\n4. LINE INTEGRAL")
print("-" * 70)
# Integrate F = (y, x) along circle x=cos(t), y=sin(t)
Fx_field = lambda x, y: y
Fy_field = lambda x, y: x
curve_x = lambda t: np.cos(t)
curve_y = lambda t: np.sin(t)
t_vals = np.linspace(0, 2*np.pi, 100)

line_int = line_integral_2d(Fx_field, Fy_field, curve_x, curve_y, t_vals)
print(f"F(x,y) = (y, x)")
print(f"Curve: circle x=cos(t), y=sin(t), t∈[0,2π]")
print(f"∮ F·dr = {line_int:.6f}")
print(f"Expected: 2π = {2*np.pi:.6f}")

# Test 5: Green's theorem
print("\n5. GREEN'S THEOREM")
print("-" * 70)
# ∮ P dx + Q dy = ∬ (∂Q/∂x - ∂P/∂y) dA
# F = (-y, x), over unit circle
P = lambda x, y: -y
Q = lambda x, y: x

# Line integral (already computed above)
print(f"Field: F = (-y, x)")
print(f"Region: unit circle")

# Compute ∂Q/∂x - ∂P/∂y
dQ_dx = lambda x, y: 0  # ∂x/∂x = 1, but we need derivative of Q w.r.t. x
dP_dy = lambda x, y: 0  # ∂(-y)/∂y = -1

# Actually: ∂Q/∂x - ∂P/∂y = ∂(x)/∂x - ∂(-y)/∂y = 1 - (-1) = 2
integrand_value = 2
area = np.pi * 1**2  # unit circle
double_integral = integrand_value * area

print(f"∮ F·dr (line integral) = {-line_int:.6f}")
print(f"∬ (∂Q/∂x - ∂P/∂y) dA = 2 × π = {double_integral:.6f}")
print(f"Green's theorem verified: {np.abs(-line_int - double_integral) < 0.1}")

# Test 6: Divergence theorem
print("\n6. DIVERGENCE THEOREM")
print("-" * 70)
# F = (x, y, z), over cube [0,1]^3
print(f"Field: F = (x, y, z)")
print(f"Region: unit cube [0,1]³")

# Surface integral over 6 faces
flux = 0

# Face x=1 (outward normal = (1,0,0))
y_vals = np.linspace(0, 1, 10)
z_vals = np.linspace(0, 1, 10)
for y in y_vals:
    for z in z_vals:
        flux += Fx_3d(1, y, z) * (1/10) * (1/10)

# Face x=0 (outward normal = (-1,0,0))
for y in y_vals:
    for z in z_vals:
        flux -= Fx_3d(0, y, z) * (1/10) * (1/10)

# Face y=1
for x in np.linspace(0, 1, 10):
    for z in z_vals:
        flux += Fy_3d(x, 1, z) * (1/10) * (1/10)

# Face y=0
for x in np.linspace(0, 1, 10):
    for z in z_vals:
        flux -= Fy_3d(x, 0, z) * (1/10) * (1/10)

# Face z=1
for x in np.linspace(0, 1, 10):
    for y in y_vals:
        flux += Fz_3d(x, y, 1) * (1/10) * (1/10)

# Face z=0
for x in np.linspace(0, 1, 10):
    for y in y_vals:
        flux -= Fz_3d(x, y, 0) * (1/10) * (1/10)

print(f"Surface integral ∬ F·n dS = {flux:.6f}")

# Volume integral of divergence
# div(F) = 3, volume = 1
volume_integral = 3 * 1
print(f"Volume integral ∭ ∇·F dV = {volume_integral:.6f}")
print(f"Divergence theorem verified: {np.abs(flux - volume_integral) < 0.2}")

# Test 7: Stokes' theorem
print("\n7. STOKES' THEOREM")
print("-" * 70)
# F = (-y, x, 0), over unit disk in xy-plane
print(f"Field: F = (-y, x, 0)")
print(f"Surface: unit disk in xy-plane")
print(f"Boundary: unit circle")

# Line integral around boundary (circle)
Fx_stokes = lambda x, y: -y
Fy_stokes = lambda x, y: x
line_int_stokes = line_integral_2d(Fx_stokes, Fy_stokes, curve_x, curve_y, t_vals)
print(f"∮ F·dr (line integral) = {line_int_stokes:.6f}")

# Surface integral of curl·n
# curl(F) = (0, 0, 2), n = (0, 0, 1)
# curl·n = 2
surface_int_stokes = 2 * np.pi * 1**2
print(f"∬ (∇×F)·n dS = 2 × π = {surface_int_stokes:.6f}")
print(f"Stokes' theorem verified: {np.abs(line_int_stokes - surface_int_stokes) < 0.1}")

# Visualization
if HAS_MATPLOTLIB:
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Gradient field
    ax1 = plt.subplot(2, 3, 1)
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)

    f_scalar = X**2 + Y**2
    U = 2*X  # ∂f/∂x
    V = 2*Y  # ∂f/∂y

    contour = ax1.contour(X, Y, f_scalar, levels=10, cmap='viridis', alpha=0.3)
    ax1.quiver(X, Y, U, V, color='red', alpha=0.6)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Gradient Field ∇f (f=x²+y²)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Divergence visualization
    ax2 = plt.subplot(2, 3, 2)
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    U = X
    V = Y

    div = np.ones_like(X) * 2  # div(F) = 2
    im = ax2.contourf(X, Y, div, levels=10, cmap='RdBu_r', alpha=0.6)
    ax2.quiver(X, Y, U, V, color='black', alpha=0.6)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Divergence: F=(x,y), ∇·F=2')
    ax2.set_aspect('equal')
    plt.colorbar(im, ax=ax2)

    # Plot 3: Curl visualization
    ax3 = plt.subplot(2, 3, 3)
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    U = -Y
    V = X

    ax3.quiver(X, Y, U, V, color='blue', alpha=0.6)

    # Add circular streamlines
    theta = np.linspace(0, 2*np.pi, 100)
    for r in [0.5, 1.0, 1.5]:
        ax3.plot(r*np.cos(theta), r*np.sin(theta), 'r--', alpha=0.3, linewidth=0.5)

    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Curl: F=(-y,x,0), ∇×F=(0,0,2)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Line integral path
    ax4 = plt.subplot(2, 3, 4)
    t = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(t)
    y_circle = np.sin(t)

    # Draw vector field
    x_grid = np.linspace(-1.5, 1.5, 15)
    y_grid = np.linspace(-1.5, 1.5, 15)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    U_grid = Y_grid
    V_grid = X_grid

    ax4.quiver(X_grid, Y_grid, U_grid, V_grid, alpha=0.4)
    ax4.plot(x_circle, y_circle, 'r-', linewidth=3, label='Integration path')
    ax4.arrow(1, 0, 0, 0.3, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Line Integral: F=(y,x)')
    ax4.legend()
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Green's theorem
    ax5 = plt.subplot(2, 3, 5)
    x = np.linspace(-1.5, 1.5, 20)
    y = np.linspace(-1.5, 1.5, 20)
    X, Y = np.meshgrid(x, y)

    # ∂Q/∂x - ∂P/∂y = 2 for F=(-y,x)
    integrand = np.ones_like(X) * 2

    # Mask to show only inside circle
    mask = X**2 + Y**2 <= 1
    integrand_masked = np.where(mask, integrand, np.nan)

    im = ax5.contourf(X, Y, integrand_masked, levels=10, cmap='YlOrRd', alpha=0.7)
    ax5.plot(x_circle, y_circle, 'b-', linewidth=3, label='∂D')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title("Green's Theorem: ∂Q/∂x - ∂P/∂y = 2")
    ax5.legend()
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Conservative vs non-conservative field
    ax6 = plt.subplot(2, 3, 6)
    x = np.linspace(-2, 2, 15)
    y = np.linspace(-2, 2, 15)
    X, Y = np.meshgrid(x, y)

    # Conservative: F = ∇(xy) = (y, x)
    U_cons = Y
    V_cons = X

    ax6.quiver(X, Y, U_cons, V_cons, color='green', alpha=0.6, label='Conservative')

    # Draw potential lines
    for c in [-2, -1, 0, 1, 2]:
        x_pot = np.linspace(-2, 2, 100)
        y_pot = c / x_pot
        y_pot = np.clip(y_pot, -2, 2)
        mask = np.abs(x_pot) > 0.1
        ax6.plot(x_pot[mask], y_pot[mask], 'g--', alpha=0.3, linewidth=0.5)

    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('Conservative Field: F=∇(xy)=(y,x)')
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Mathematical_Methods/05_vector_analysis.png', dpi=150)
    print("Saved visualization: 05_vector_analysis.png")
    plt.close()

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
