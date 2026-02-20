"""
Calculus of Variations - Euler-Lagrange Equation and Applications

This script demonstrates:
- Euler-Lagrange equation
- Brachistochrone problem
- Catenary curve
- Geodesics on surfaces
- Lagrangian mechanics (pendulum, spring)
- Minimal surface of revolution
- Isoperimetric problems
"""

import numpy as np

try:
    from scipy.optimize import minimize, fsolve
    from scipy.integrate import odeint, solve_ivp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using limited implementations")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def euler_lagrange_geodesic_plane(x0, y0, x1, y1, n_points=100):
    """
    Solve for geodesic (straight line) on a plane
    Minimize: ∫ √(1 + y'²) dx
    Euler-Lagrange: d/dx(y'/√(1+y'²)) = 0 → y'' = 0
    """
    x = np.linspace(x0, x1, n_points)
    # Straight line
    y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return x, y


def brachistochrone_curve(x0, y0, x1, y1, n_points=100):
    """
    Brachistochrone problem: fastest descent curve under gravity
    Solution is a cycloid: x = r(θ - sin(θ)), y = r(1 - cos(θ))
    """
    # Find r and theta_max that satisfy boundary conditions
    # This is a transcendental equation, solve numerically

    def equations(params):
        r, theta_max = params
        x_end = r * (theta_max - np.sin(theta_max))
        y_end = r * (1 - np.cos(theta_max))
        return [x_end - (x1 - x0), y_end - (y1 - y0)]

    if HAS_SCIPY:
        r_init = (x1 - x0) / np.pi
        theta_init = np.pi
        r, theta_max = fsolve(equations, [r_init, theta_init])

        theta = np.linspace(0, theta_max, n_points)
        x = x0 + r * (theta - np.sin(theta))
        y = y0 + r * (1 - np.cos(theta))
    else:
        # Approximate with straight line
        x = np.linspace(x0, x1, n_points)
        y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    return x, y


def catenary_curve(x0, x1, length, n_points=100):
    """
    Catenary: shape of hanging chain
    Minimize: ∫ y√(1 + y'²) dx subject to fixed arc length
    Solution: y = a*cosh(x/a) + b
    """
    # Find parameter 'a' such that arc length matches
    def arc_length(a):
        x = np.linspace(x0, x1, 1000)
        y = a * np.cosh(x / a)
        dy_dx = np.sinh(x / a)
        ds = np.sqrt(1 + dy_dx**2)
        return np.trapz(ds, x)

    if HAS_SCIPY:
        # Find 'a' that gives correct length
        result = minimize(lambda a: (arc_length(a[0]) - length)**2, [1.0], bounds=[(0.1, 10)])
        a = result.x[0]
    else:
        a = 1.0

    x = np.linspace(x0, x1, n_points)
    y = a * np.cosh(x / a)

    return x, y


def lagrangian_simple_pendulum(t, state, g=9.8, L=1.0):
    """
    Simple pendulum using Lagrangian mechanics
    L = T - V = (1/2)mL²θ'² - mgL(1-cos(θ))
    Euler-Lagrange: d/dt(∂L/∂θ') - ∂L/∂θ = 0
    → θ'' + (g/L)sin(θ) = 0
    """
    theta, theta_dot = state
    theta_ddot = -(g / L) * np.sin(theta)
    return [theta_dot, theta_ddot]


def lagrangian_spring_mass(t, state, k=1.0, m=1.0):
    """
    Spring-mass system using Lagrangian mechanics
    L = T - V = (1/2)mx'² - (1/2)kx²
    Euler-Lagrange: mx'' + kx = 0
    """
    x, x_dot = state
    x_ddot = -(k / m) * x
    return [x_dot, x_ddot]


def minimal_surface_revolution(y0, y1, x_range, n_points=100):
    """
    Minimal surface of revolution
    Minimize: ∫ y√(1 + y'²) dx
    Euler-Lagrange: y'' = (1 + y'²)/y
    Solution: catenoid y = c*cosh(x/c)
    """
    x0, x1 = x_range
    x = np.linspace(x0, x1, n_points)

    # For simplicity, use catenary with c=1
    c = 1.0
    y = c * np.cosh((x - (x0 + x1)/2) / c)

    # Scale to match boundary conditions
    y_scaled = y0 + (y1 - y0) * (y - y[0]) / (y[-1] - y[0])

    return x, y_scaled


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

print("=" * 70)
print("CALCULUS OF VARIATIONS - EULER-LAGRANGE EQUATION")
print("=" * 70)

# Test 1: Geodesic on plane (straight line)
print("\n1. GEODESIC ON PLANE - SHORTEST PATH")
print("-" * 70)

x0, y0 = 0, 0
x1, y1 = 3, 2

x_geo, y_geo = euler_lagrange_geodesic_plane(x0, y0, x1, y1)

length_geo = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

print(f"From ({x0}, {y0}) to ({x1}, {y1})")
print(f"Geodesic length: {length_geo:.6f}")
print(f"Expected (Euclidean): {length_geo:.6f}")

# Compute actual path length
dx = x_geo[1] - x_geo[0]
dy_dx = np.gradient(y_geo, dx)
ds = np.sqrt(1 + dy_dx**2)
computed_length = np.trapz(ds, x_geo)
print(f"Numerical verification: {computed_length:.6f}")

# Test 2: Brachistochrone problem
print("\n2. BRACHISTOCHRONE - FASTEST DESCENT")
print("-" * 70)

x0, y0 = 0, 0
x1, y1 = 2, -1

x_brach, y_brach = brachistochrone_curve(x0, y0, x1, y1)

print(f"From ({x0}, {y0}) to ({x1}, {y1})")
print("Solution is a cycloid curve")

# Compute descent time (T = ∫ ds/v where v = √(2g|y|))
g = 9.8
dx = x_brach[1] - x_brach[0]
dy_dx = np.gradient(y_brach, dx)
ds = np.sqrt(dx**2 + (dy_dx * dx)**2)

# Velocity v = √(2g|y - y0|)
v = np.sqrt(2 * g * np.abs(y_brach - y0) + 1e-10)
dt = ds / v
time_brach = np.sum(dt)

print(f"Descent time (brachistochrone): {time_brach:.6f} s")

# Compare with straight line
x_straight = np.linspace(x0, x1, len(x_brach))
y_straight = y0 + (y1 - y0) * (x_straight - x0) / (x1 - x0)
dy_dx_straight = np.gradient(y_straight, dx)
ds_straight = np.sqrt(dx**2 + (dy_dx_straight * dx)**2)
v_straight = np.sqrt(2 * g * np.abs(y_straight - y0) + 1e-10)
dt_straight = ds_straight / v_straight
time_straight = np.sum(dt_straight)

print(f"Descent time (straight line):   {time_straight:.6f} s")
print(f"Time saved: {(time_straight - time_brach):.6f} s ({(time_straight/time_brach - 1)*100:.1f}%)")

# Test 3: Catenary
print("\n3. CATENARY - HANGING CHAIN")
print("-" * 70)

x0, x1 = -1, 1
chain_length = 3.0

x_cat, y_cat = catenary_curve(x0, x1, chain_length)

print(f"Chain from x={x0} to x={x1}")
print(f"Total length: {chain_length}")

# Verify length
dx = x_cat[1] - x_cat[0]
dy_dx = np.gradient(y_cat, dx)
ds = np.sqrt(1 + dy_dx**2)
computed_length = np.trapz(ds, x_cat)

print(f"Computed length: {computed_length:.6f}")
print(f"Shape: y = a·cosh(x/a)")

# Potential energy (proportional to ∫ y ds)
potential_energy = np.trapz(y_cat * ds, x_cat)
print(f"Potential energy (normalized): {potential_energy:.6f}")

# Test 4: Lagrangian mechanics - Simple pendulum
print("\n4. LAGRANGIAN MECHANICS - SIMPLE PENDULUM")
print("-" * 70)

g, L = 9.8, 1.0
theta0 = np.pi / 4  # 45 degrees
theta_dot0 = 0

print(f"Pendulum length L = {L} m")
print(f"Initial angle θ₀ = {np.degrees(theta0):.1f}°")
print(f"Initial velocity θ'₀ = {theta_dot0} rad/s")

if HAS_SCIPY:
    t_span = (0, 5)
    t_eval = np.linspace(0, 5, 500)

    sol = solve_ivp(lagrangian_simple_pendulum, t_span, [theta0, theta_dot0],
                    t_eval=t_eval, args=(g, L))
    t_pend = sol.t
    theta = sol.y[0]
    theta_dot = sol.y[1]

    # Energy conservation check
    T = 0.5 * L**2 * theta_dot**2  # Kinetic (with m=1)
    V = g * L * (1 - np.cos(theta))  # Potential
    E = T + V

    print(f"\nEnergy conservation:")
    print(f"  E(t=0) = {E[0]:.6f} J")
    print(f"  E(t={t_eval[-1]}) = {E[-1]:.6f} J")
    print(f"  Relative change: {abs(E[-1] - E[0])/E[0] * 100:.2f}%")

    # Small angle approximation period
    period_small_angle = 2 * np.pi * np.sqrt(L / g)
    print(f"\nSmall angle period: {period_small_angle:.4f} s")

# Test 5: Lagrangian mechanics - Spring-mass
print("\n5. LAGRANGIAN MECHANICS - SPRING-MASS SYSTEM")
print("-" * 70)

k, m = 4.0, 1.0
x0 = 1.0
v0 = 0.0

omega = np.sqrt(k / m)
period = 2 * np.pi / omega

print(f"Spring constant k = {k} N/m")
print(f"Mass m = {m} kg")
print(f"Natural frequency ω = {omega:.4f} rad/s")
print(f"Period T = {period:.4f} s")

if HAS_SCIPY:
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 500)

    sol = solve_ivp(lagrangian_spring_mass, t_span, [x0, v0],
                    t_eval=t_eval, args=(k, m))
    t_spring = sol.t
    x = sol.y[0]
    x_dot = sol.y[1]

    # Energy conservation
    T = 0.5 * m * x_dot**2
    V = 0.5 * k * x**2
    E = T + V

    print(f"\nEnergy conservation:")
    print(f"  E(t=0) = {E[0]:.6f} J")
    print(f"  E(t={t_eval[-1]}) = {E[-1]:.6f} J")
    print(f"  Relative change: {abs(E[-1] - E[0])/E[0] * 100:.2f}%")

# Test 6: Minimal surface of revolution
print("\n6. MINIMAL SURFACE OF REVOLUTION - CATENOID")
print("-" * 70)

y0, y1 = 1.0, 1.0
x_range = (-1, 1)

x_surf, y_surf = minimal_surface_revolution(y0, y1, x_range)

print(f"Boundary conditions: y({x_range[0]}) = {y0}, y({x_range[1]}) = {y1}")
print("Rotating curve around x-axis")
print("Solution: catenoid (minimal surface)")

# Surface area = 2π ∫ y√(1 + y'²) dx
dx = x_surf[1] - x_surf[0]
dy_dx = np.gradient(y_surf, dx)
integrand = y_surf * np.sqrt(1 + dy_dx**2)
surface_area = 2 * np.pi * np.trapz(integrand, x_surf)

print(f"Surface area: {surface_area:.6f}")

# Visualization
if HAS_MATPLOTLIB:
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Geodesic
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(x_geo, y_geo, 'b-', linewidth=2, label='Geodesic')
    ax1.plot([x0, x1], [y0, y1], 'ro', markersize=8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Geodesic on Plane')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: Brachistochrone
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(x_brach, y_brach, 'b-', linewidth=2, label='Brachistochrone')
    ax2.plot(x_straight, y_straight, 'r--', linewidth=2, label='Straight line')
    ax2.plot([x0, x1], [y0, y1], 'ko', markersize=8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Brachistochrone: Fastest Descent')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Catenary
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(x_cat, y_cat, 'b-', linewidth=2, label='Catenary')
    ax3.plot([x0, x1], [y_cat[0], y_cat[-1]], 'ro', markersize=8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Catenary: Hanging Chain')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()

    # Plot 4: Pendulum motion
    if HAS_SCIPY:
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(t_pend, np.degrees(theta), 'b-', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Angle (degrees)')
        ax4.set_title('Simple Pendulum: θ(t)')
        ax4.grid(True, alpha=0.3)

        # Plot 5: Pendulum phase portrait
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(np.degrees(theta), np.degrees(theta_dot), 'b-', linewidth=1.5)
        ax5.plot(np.degrees(theta[0]), np.degrees(theta_dot[0]), 'go', markersize=8)
        ax5.set_xlabel('θ (degrees)')
        ax5.set_ylabel("θ' (degrees/s)")
        ax5.set_title('Pendulum Phase Portrait')
        ax5.grid(True, alpha=0.3)

        # Plot 6: Spring-mass energy
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(t_spring, T, 'b-', linewidth=2, label='Kinetic')
        ax6.plot(t_spring, V, 'r-', linewidth=2, label='Potential')
        ax6.plot(t_spring, E, 'k--', linewidth=2, label='Total')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Energy (J)')
        ax6.set_title('Spring-Mass Energy Conservation')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Mathematical_Methods/14_calculus_of_variations.png', dpi=150)
    print("Saved visualization: 14_calculus_of_variations.png")
    plt.close()

    # Additional 3D plot: Catenoid surface
    if HAS_SCIPY:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create surface of revolution
        theta_surf = np.linspace(0, 2*np.pi, 50)
        X_surf, Theta_surf = np.meshgrid(x_surf, theta_surf)

        Y_surf_2d = np.tile(y_surf, (len(theta_surf), 1))
        Y_surf_3d = Y_surf_2d * np.cos(Theta_surf)
        Z_surf_3d = Y_surf_2d * np.sin(Theta_surf)

        ax.plot_surface(X_surf, Y_surf_3d, Z_surf_3d, cmap='viridis', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Catenoid: Minimal Surface of Revolution')

        plt.savefig('/opt/projects/01_Personal/03_Study/examples/Mathematical_Methods/14_catenoid_3d.png', dpi=150)
        print("Saved visualization: 14_catenoid_3d.png")
        plt.close()

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
