"""
Partial Differential Equations (PDEs) - Numerical Solutions

This script demonstrates:
- Heat equation (parabolic): ∂u/∂t = α ∂²u/∂x²
- Wave equation (hyperbolic): ∂²u/∂t² = c² ∂²u/∂x²
- Laplace equation (elliptic): ∂²u/∂x² + ∂²u/∂y² = 0
- Finite difference methods
- Relaxation method for Laplace equation
- Time evolution visualization
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def solve_heat_equation_1d(alpha, L, T, nx, nt, initial_condition, boundary_conditions):
    """
    Solve 1D heat equation: ∂u/∂t = α ∂²u/∂x²
    Using explicit finite difference (FTCS scheme)

    boundary_conditions: tuple (left_type, left_val, right_type, right_val)
    where type is 'dirichlet' or 'neumann'
    """
    dx = L / (nx - 1)
    dt = T / nt

    # Stability criterion: α*dt/dx² <= 0.5
    r = alpha * dt / dx**2
    if r > 0.5:
        print(f"Warning: Stability criterion violated! r={r:.3f} > 0.5")

    x = np.linspace(0, L, nx)
    u = np.zeros((nt, nx))

    # Initial condition
    u[0, :] = initial_condition(x)

    # Time stepping
    for n in range(nt - 1):
        for i in range(1, nx - 1):
            u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])

        # Boundary conditions
        left_type, left_val, right_type, right_val = boundary_conditions

        if left_type == 'dirichlet':
            u[n+1, 0] = left_val
        elif left_type == 'neumann':
            u[n+1, 0] = u[n+1, 1] - left_val * dx

        if right_type == 'dirichlet':
            u[n+1, -1] = right_val
        elif right_type == 'neumann':
            u[n+1, -1] = u[n+1, -2] + right_val * dx

    return x, u


def solve_wave_equation_1d(c, L, T, nx, nt, initial_u, initial_v, boundary_conditions):
    """
    Solve 1D wave equation: ∂²u/∂t² = c² ∂²u/∂x²
    Using explicit finite difference

    initial_u: initial displacement
    initial_v: initial velocity
    """
    dx = L / (nx - 1)
    dt = T / nt

    # Stability criterion: c*dt/dx <= 1 (CFL condition)
    cfl = c * dt / dx
    if cfl > 1:
        print(f"Warning: CFL condition violated! CFL={cfl:.3f} > 1")

    x = np.linspace(0, L, nx)
    u = np.zeros((nt, nx))

    # Initial conditions
    u[0, :] = initial_u(x)

    # First time step using initial velocity
    r = (c * dt / dx)**2
    for i in range(1, nx - 1):
        u[1, i] = u[0, i] + dt * initial_v(x[i]) + 0.5 * r * (u[0, i+1] - 2*u[0, i] + u[0, i-1])

    # Apply boundary conditions to first step
    left_type, left_val, right_type, right_val = boundary_conditions
    if left_type == 'dirichlet':
        u[1, 0] = left_val
    if right_type == 'dirichlet':
        u[1, -1] = right_val

    # Time stepping
    for n in range(1, nt - 1):
        for i in range(1, nx - 1):
            u[n+1, i] = 2*u[n, i] - u[n-1, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])

        # Boundary conditions
        if left_type == 'dirichlet':
            u[n+1, 0] = left_val
        if right_type == 'dirichlet':
            u[n+1, -1] = right_val

    return x, u


def solve_laplace_2d(nx, ny, max_iter=1000, tol=1e-5, boundary_func=None):
    """
    Solve 2D Laplace equation: ∂²u/∂x² + ∂²u/∂y² = 0
    Using Jacobi relaxation method
    """
    u = np.zeros((nx, ny))

    # Apply boundary conditions
    if boundary_func is not None:
        # Left and right boundaries
        for j in range(ny):
            y = j / (ny - 1)
            u[0, j] = boundary_func('left', 0, y)
            u[-1, j] = boundary_func('right', 1, y)

        # Top and bottom boundaries
        for i in range(nx):
            x = i / (nx - 1)
            u[i, 0] = boundary_func('bottom', x, 0)
            u[i, -1] = boundary_func('top', x, 1)

    # Jacobi iteration
    for iteration in range(max_iter):
        u_old = u.copy()

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u[i, j] = 0.25 * (u_old[i+1, j] + u_old[i-1, j] +
                                  u_old[i, j+1] + u_old[i, j-1])

        # Check convergence
        error = np.max(np.abs(u - u_old))
        if error < tol:
            print(f"Converged in {iteration} iterations, error={error:.2e}")
            break

    return u


def solve_poisson_2d(nx, ny, source_func, max_iter=1000, tol=1e-5):
    """
    Solve 2D Poisson equation: ∂²u/∂x² + ∂²u/∂y² = f(x,y)
    Using Jacobi relaxation
    """
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    u = np.zeros((nx, ny))
    f = np.zeros((nx, ny))

    # Compute source term
    for i in range(nx):
        for j in range(ny):
            x = i * dx
            y = j * dy
            f[i, j] = source_func(x, y)

    # Jacobi iteration
    for iteration in range(max_iter):
        u_old = u.copy()

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u[i, j] = 0.25 * (u_old[i+1, j] + u_old[i-1, j] +
                                  u_old[i, j+1] + u_old[i, j-1] -
                                  dx**2 * f[i, j])

        # Check convergence
        error = np.max(np.abs(u - u_old))
        if error < tol:
            print(f"Converged in {iteration} iterations")
            break

    return u


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

print("=" * 70)
print("PARTIAL DIFFERENTIAL EQUATIONS - NUMERICAL SOLUTIONS")
print("=" * 70)

# Test 1: Heat equation - temperature diffusion
print("\n1. HEAT EQUATION - 1D DIFFUSION")
print("-" * 70)
alpha = 0.01  # Thermal diffusivity
L = 1.0       # Length
T = 10.0      # Total time
nx = 50       # Spatial points
nt = 500      # Time steps

# Initial condition: Gaussian pulse
initial_temp = lambda x: np.exp(-50 * (x - 0.5)**2)

# Boundary conditions: Dirichlet (fixed temperature at ends)
bc = ('dirichlet', 0.0, 'dirichlet', 0.0)

x, u_heat = solve_heat_equation_1d(alpha, L, T, nx, nt, initial_temp, bc)

print(f"Domain: [0, {L}], Time: [0, {T}]")
print(f"Grid: {nx} × {nt}")
print(f"Thermal diffusivity α = {alpha}")

dt = T / nt
dx = L / (nx - 1)
r = alpha * dt / dx**2
print(f"Stability parameter r = α*dt/dx² = {r:.4f} (must be ≤ 0.5)")

# Check heat conservation (integral should decrease due to boundary)
heat_initial = np.trapz(u_heat[0, :], x)
heat_final = np.trapz(u_heat[-1, :], x)
print(f"\nHeat content:")
print(f"  Initial: {heat_initial:.6f}")
print(f"  Final:   {heat_final:.6f}")

# Test 2: Wave equation - vibrating string
print("\n2. WAVE EQUATION - VIBRATING STRING")
print("-" * 70)
c = 1.0       # Wave speed
L = 1.0
T = 2.0
nx = 100
nt = 400

# Initial displacement: plucked string
initial_disp = lambda x: np.where(x < 0.5, 2*x, 2*(1-x))
initial_vel = lambda x: 0.0  # Released from rest

# Boundary conditions: fixed ends
bc = ('dirichlet', 0.0, 'dirichlet', 0.0)

x, u_wave = solve_wave_equation_1d(c, L, T, nx, nt, initial_disp, initial_vel, bc)

dt = T / nt
dx = L / (nx - 1)
cfl = c * dt / dx
print(f"Domain: [0, {L}], Time: [0, {T}]")
print(f"Grid: {nx} × {nt}")
print(f"Wave speed c = {c}")
print(f"CFL number = c*dt/dx = {cfl:.4f} (must be ≤ 1)")

# Energy conservation check
def wave_energy(u_current, u_prev, dx, dt, c):
    """Compute total energy: kinetic + potential"""
    # Kinetic energy: (1/2) ∫ (∂u/∂t)² dx
    u_t = (u_current - u_prev) / dt
    kinetic = 0.5 * np.trapz(u_t**2, dx=dx)

    # Potential energy: (1/2) c² ∫ (∂u/∂x)² dx
    u_x = np.gradient(u_current, dx)
    potential = 0.5 * c**2 * np.trapz(u_x**2, dx=dx)

    return kinetic + potential

energy_initial = wave_energy(u_wave[1, :], u_wave[0, :], dx, dt, c)
energy_middle = wave_energy(u_wave[nt//2, :], u_wave[nt//2-1, :], dx, dt, c)
energy_final = wave_energy(u_wave[-1, :], u_wave[-2, :], dx, dt, c)

print(f"\nTotal energy (should be conserved):")
print(f"  t={0:.2f}: E = {energy_initial:.6f}")
print(f"  t={T/2:.2f}: E = {energy_middle:.6f}")
print(f"  t={T:.2f}: E = {energy_final:.6f}")

# Test 3: Laplace equation - steady-state temperature
print("\n3. LAPLACE EQUATION - 2D STEADY-STATE")
print("-" * 70)
nx, ny = 50, 50

# Boundary conditions: heated on top, cold on bottom, insulated sides
def boundary_laplace(side, x, y):
    if side == 'top':
        return 100.0  # Hot
    elif side == 'bottom':
        return 0.0    # Cold
    elif side == 'left':
        return 50.0 * y  # Linear gradient
    elif side == 'right':
        return 50.0 * y
    return 0.0

u_laplace = solve_laplace_2d(nx, ny, max_iter=5000, tol=1e-6,
                             boundary_func=boundary_laplace)

print(f"Grid: {nx} × {ny}")
print(f"Boundary conditions:")
print(f"  Top:    T = 100°C")
print(f"  Bottom: T = 0°C")
print(f"  Sides:  T = 50y°C")
print(f"\nSolution statistics:")
print(f"  Min temperature: {np.min(u_laplace):.2f}°C")
print(f"  Max temperature: {np.max(u_laplace):.2f}°C")
print(f"  Center temperature: {u_laplace[nx//2, ny//2]:.2f}°C")

# Test 4: Poisson equation with source
print("\n4. POISSON EQUATION - WITH SOURCE TERM")
print("-" * 70)
nx, ny = 50, 50

# Source term: point source at center
def source_poisson(x, y):
    r_squared = (x - 0.5)**2 + (y - 0.5)**2
    return 1000 * np.exp(-100 * r_squared)

u_poisson = solve_poisson_2d(nx, ny, source_poisson, max_iter=5000, tol=1e-6)

print(f"Grid: {nx} × {ny}")
print(f"Source: Gaussian centered at (0.5, 0.5)")
print(f"\nSolution statistics:")
print(f"  Min: {np.min(u_poisson):.4f}")
print(f"  Max: {np.max(u_poisson):.4f}")
print(f"  Center: {u_poisson[nx//2, ny//2]:.4f}")

# Visualization
if HAS_MATPLOTLIB:
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Heat equation evolution
    ax1 = plt.subplot(2, 3, 1)
    time_indices = [0, nt//4, nt//2, -1]
    times = [0, T/4, T/2, T]

    for idx, t in zip(time_indices, times):
        ax1.plot(x, u_heat[idx, :], linewidth=2, label=f't={t:.2f}')

    ax1.set_xlabel('x')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Heat Equation: 1D Diffusion')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Heat equation space-time plot
    ax2 = plt.subplot(2, 3, 2)
    t_vals = np.linspace(0, T, nt)
    X, T_mesh = np.meshgrid(x, t_vals)
    im = ax2.contourf(X, T_mesh, u_heat, levels=20, cmap='hot')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('Heat Equation: Space-Time')
    plt.colorbar(im, ax=ax2)

    # Plot 3: Wave equation snapshots
    ax3 = plt.subplot(2, 3, 3)
    time_indices = [0, nt//8, nt//4, nt//2]
    times = [0, T/8, T/4, T/2]

    for idx, t in zip(time_indices, times):
        ax3.plot(x, u_wave[idx, :], linewidth=2, label=f't={t:.2f}')

    ax3.set_xlabel('x')
    ax3.set_ylabel('Displacement')
    ax3.set_title('Wave Equation: Vibrating String')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.2, 1.2)

    # Plot 4: Wave equation space-time
    ax4 = plt.subplot(2, 3, 4)
    t_vals = np.linspace(0, T, nt)
    X, T_mesh = np.meshgrid(x, t_vals)
    im = ax4.contourf(X, T_mesh, u_wave, levels=20, cmap='RdBu_r')
    ax4.set_xlabel('x')
    ax4.set_ylabel('t')
    ax4.set_title('Wave Equation: Space-Time')
    plt.colorbar(im, ax=ax4)

    # Plot 5: Laplace equation solution
    ax5 = plt.subplot(2, 3, 5)
    x_grid = np.linspace(0, 1, nx)
    y_grid = np.linspace(0, 1, ny)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    im = ax5.contourf(X_grid, Y_grid, u_laplace.T, levels=20, cmap='coolwarm')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title('Laplace Equation: Steady-State')
    plt.colorbar(im, ax=ax5)
    ax5.set_aspect('equal')

    # Plot 6: Poisson equation solution
    ax6 = plt.subplot(2, 3, 6)
    im = ax6.contourf(X_grid, Y_grid, u_poisson.T, levels=20, cmap='viridis')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('Poisson Equation: With Source')
    plt.colorbar(im, ax=ax6)
    ax6.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Mathematical_Methods/10_pde.png', dpi=150)
    print("Saved visualization: 10_pde.png")
    plt.close()

    # Additional plot: 3D surface for Laplace equation
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X_grid, Y_grid, u_laplace.T, cmap='coolwarm', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Temperature')
    ax1.set_title('Laplace Equation: 3D View')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X_grid, Y_grid, u_poisson.T, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')
    ax2.set_title('Poisson Equation: 3D View')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Mathematical_Methods/10_pde_3d.png', dpi=150)
    print("Saved visualization: 10_pde_3d.png")
    plt.close()

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
