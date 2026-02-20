"""
Ordinary Differential Equations (ODEs) - Numerical Methods and Analysis

This script demonstrates:
- Euler's method
- Runge-Kutta 4th order (RK4)
- scipy.integrate.solve_ivp
- Harmonic oscillator (second-order ODE)
- Damped oscillator
- Lorenz system (chaotic system)
- Phase portraits
"""

import numpy as np

try:
    from scipy.integrate import solve_ivp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def euler_method(f, y0, t_span, dt):
    """
    Solve ODE dy/dt = f(t, y) using Euler's method
    y_{n+1} = y_n + dt * f(t_n, y_n)
    """
    t0, tf = t_span
    t = np.arange(t0, tf + dt, dt)
    y = np.zeros((len(t), len(np.atleast_1d(y0))))
    y[0] = y0

    for i in range(len(t) - 1):
        y[i + 1] = y[i] + dt * f(t[i], y[i])

    return t, y


def rk4_method(f, y0, t_span, dt):
    """
    Solve ODE dy/dt = f(t, y) using Runge-Kutta 4th order
    """
    t0, tf = t_span
    t = np.arange(t0, tf + dt, dt)
    y = np.zeros((len(t), len(np.atleast_1d(y0))))
    y[0] = y0

    for i in range(len(t) - 1):
        k1 = dt * f(t[i], y[i])
        k2 = dt * f(t[i] + dt/2, y[i] + k1/2)
        k3 = dt * f(t[i] + dt/2, y[i] + k2/2)
        k4 = dt * f(t[i] + dt, y[i] + k3)

        y[i + 1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t, y


def harmonic_oscillator(t, y, omega=1.0):
    """
    Simple harmonic oscillator: y'' + ω²y = 0
    State: y = [y, y']
    """
    return np.array([y[1], -omega**2 * y[0]])


def damped_oscillator(t, y, gamma=0.5, omega=1.0):
    """
    Damped harmonic oscillator: y'' + 2γy' + ω²y = 0
    State: y = [y, y']
    """
    return np.array([y[1], -2*gamma*y[1] - omega**2 * y[0]])


def lorenz_system(t, y, sigma=10, rho=28, beta=8/3):
    """
    Lorenz system (chaotic):
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    """
    x, y_val, z = y
    return np.array([
        sigma * (y_val - x),
        x * (rho - z) - y_val,
        x * y_val - beta * z
    ])


def van_der_pol(t, y, mu=1.0):
    """
    Van der Pol oscillator (limit cycle):
    y'' - μ(1 - y²)y' + y = 0
    """
    return np.array([y[1], mu * (1 - y[0]**2) * y[1] - y[0]])


def pendulum(t, y, g=9.8, L=1.0, damping=0.0):
    """
    Nonlinear pendulum: θ'' + (g/L)sin(θ) + damping*θ' = 0
    State: y = [θ, θ']
    """
    return np.array([y[1], -(g/L) * np.sin(y[0]) - damping * y[1]])


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

print("=" * 70)
print("ORDINARY DIFFERENTIAL EQUATIONS - NUMERICAL METHODS")
print("=" * 70)

# Test 1: Exponential decay - Euler vs RK4
print("\n1. EXPONENTIAL DECAY: dy/dt = -y, y(0) = 1")
print("-" * 70)

# Exact solution: y(t) = e^(-t)
f_decay = lambda t, y: -y
y0 = np.array([1.0])
t_span = (0, 5)
dt = 0.5

t_euler, y_euler = euler_method(f_decay, y0, t_span, dt)
t_rk4, y_rk4 = rk4_method(f_decay, y0, t_span, dt)
y_exact = np.exp(-t_euler)

print(f"Time step dt = {dt}")
print(f"\n{'t':>6s} {'Exact':>10s} {'Euler':>10s} {'RK4':>10s} {'Euler Err':>12s} {'RK4 Err':>12s}")
print("-" * 70)
for i in range(0, len(t_euler), 2):
    t = t_euler[i]
    exact = y_exact[i]
    euler_val = y_euler[i, 0]
    rk4_val = y_rk4[i, 0]
    euler_err = abs(exact - euler_val)
    rk4_err = abs(exact - rk4_val)
    print(f"{t:6.1f} {exact:10.6f} {euler_val:10.6f} {rk4_val:10.6f} "
          f"{euler_err:12.2e} {rk4_err:12.2e}")

# Test 2: Harmonic oscillator
print("\n2. HARMONIC OSCILLATOR: y'' + ω²y = 0")
print("-" * 70)
omega = 2.0
y0 = np.array([1.0, 0.0])  # y(0)=1, y'(0)=0
t_span = (0, 10)
dt = 0.05

t_rk4, y_rk4 = rk4_method(lambda t, y: harmonic_oscillator(t, y, omega),
                          y0, t_span, dt)

# Exact solution: y(t) = cos(ωt)
y_exact = np.cos(omega * t_rk4)

print(f"ω = {omega}, y(0) = {y0[0]}, y'(0) = {y0[1]}")
print(f"\nEnergy conservation check (E = 0.5(y'² + ω²y²)):")
for i in [0, len(t_rk4)//4, len(t_rk4)//2, -1]:
    t = t_rk4[i]
    y_val = y_rk4[i, 0]
    y_dot = y_rk4[i, 1]
    energy = 0.5 * (y_dot**2 + omega**2 * y_val**2)
    print(f"  t={t:5.2f}: E = {energy:.6f}")

# Test 3: Damped oscillator
print("\n3. DAMPED HARMONIC OSCILLATOR")
print("-" * 70)
gamma_vals = [0.1, 0.5, 1.0, 2.0]
omega = 1.0
y0 = np.array([1.0, 0.0])
t_span = (0, 20)
dt = 0.05

print("Damping regimes:")
for gamma in gamma_vals:
    discriminant = gamma**2 - omega**2
    if discriminant < 0:
        regime = "Underdamped"
    elif discriminant == 0:
        regime = "Critically damped"
    else:
        regime = "Overdamped"

    t_rk4, y_rk4 = rk4_method(lambda t, y: damped_oscillator(t, y, gamma, omega),
                              y0, t_span, dt)

    # Find amplitude decay
    max_val_initial = np.max(np.abs(y_rk4[:100, 0]))
    max_val_final = np.max(np.abs(y_rk4[-100:, 0]))

    print(f"  γ={gamma:.1f}: {regime:18s}, "
          f"initial amp={max_val_initial:.4f}, "
          f"final amp={max_val_final:.4f}")

# Test 4: Lorenz system
print("\n4. LORENZ SYSTEM (CHAOTIC ATTRACTOR)")
print("-" * 70)
sigma, rho, beta = 10, 28, 8/3
y0 = np.array([1.0, 1.0, 1.0])
t_span = (0, 50)
dt = 0.01

if HAS_SCIPY:
    sol = solve_ivp(lambda t, y: lorenz_system(t, y, sigma, rho, beta),
                    t_span, y0, dense_output=True, max_step=dt)
    t_lorenz = sol.t
    y_lorenz = sol.y.T
else:
    t_lorenz, y_lorenz = rk4_method(lambda t, y: lorenz_system(t, y, sigma, rho, beta),
                                     y0, t_span, dt)

print(f"σ={sigma}, ρ={rho}, β={beta:.3f}")
print(f"Initial conditions: x={y0[0]}, y={y0[1]}, z={y0[2]}")
print(f"\nTrajectory statistics:")
print(f"  x: min={np.min(y_lorenz[:, 0]):7.3f}, max={np.max(y_lorenz[:, 0]):7.3f}")
print(f"  y: min={np.min(y_lorenz[:, 1]):7.3f}, max={np.max(y_lorenz[:, 1]):7.3f}")
print(f"  z: min={np.min(y_lorenz[:, 2]):7.3f}, max={np.max(y_lorenz[:, 2]):7.3f}")

# Test sensitivity to initial conditions
y0_perturb = y0 + np.array([0.001, 0, 0])
if HAS_SCIPY:
    sol_perturb = solve_ivp(lambda t, y: lorenz_system(t, y, sigma, rho, beta),
                            t_span, y0_perturb, dense_output=True, max_step=dt)
    y_perturb = sol_perturb.y.T
else:
    _, y_perturb = rk4_method(lambda t, y: lorenz_system(t, y, sigma, rho, beta),
                              y0_perturb, t_span, dt)

divergence = np.linalg.norm(y_lorenz - y_perturb, axis=1)
print(f"\nSensitivity to initial conditions (Δx₀ = 0.001):")
print(f"  t={t_lorenz[len(t_lorenz)//4]:.1f}: distance = {divergence[len(divergence)//4]:.4f}")
print(f"  t={t_lorenz[len(t_lorenz)//2]:.1f}: distance = {divergence[len(divergence)//2]:.4f}")

# Test 5: Van der Pol oscillator
print("\n5. VAN DER POL OSCILLATOR (LIMIT CYCLE)")
print("-" * 70)
mu_vals = [0.1, 1.0, 5.0]
t_span = (0, 50)
dt = 0.05

print("Nonlinearity parameter μ:")
for mu in mu_vals:
    y0 = np.array([2.0, 0.0])
    t_vdp, y_vdp = rk4_method(lambda t, y: van_der_pol(t, y, mu),
                              y0, t_span, dt)

    # Check limit cycle amplitude (look at later times)
    y_steady = y_vdp[len(y_vdp)//2:, 0]
    amplitude = (np.max(y_steady) - np.min(y_steady)) / 2

    print(f"  μ={mu:.1f}: limit cycle amplitude ≈ {amplitude:.3f}")

# Test 6: Nonlinear pendulum
print("\n6. NONLINEAR PENDULUM")
print("-" * 70)
g, L = 9.8, 1.0
t_span = (0, 10)
dt = 0.01

print("Initial angle θ₀:")
for theta0 in [0.1, 1.0, 3.0]:  # Small, medium, large angle
    y0 = np.array([theta0, 0.0])
    t_pend, y_pend = rk4_method(lambda t, y: pendulum(t, y, g, L),
                                y0, t_span, dt)

    # Compute period (time between zero crossings from same direction)
    crossings = []
    for i in range(1, len(y_pend)):
        if y_pend[i-1, 0] < 0 and y_pend[i, 0] >= 0 and y_pend[i, 1] > 0:
            crossings.append(t_pend[i])

    if len(crossings) >= 2:
        period = np.mean(np.diff(crossings))
    else:
        period = np.nan

    # Small angle approximation: T = 2π√(L/g)
    period_approx = 2 * np.pi * np.sqrt(L / g)

    print(f"  θ₀={theta0:.1f} rad: T={period:.4f}s "
          f"(small angle: {period_approx:.4f}s)")

# Visualization
if HAS_MATPLOTLIB:
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Euler vs RK4 comparison
    ax1 = plt.subplot(2, 3, 1)
    f_decay = lambda t, y: -y
    y0 = np.array([1.0])
    t_span = (0, 5)
    dt = 0.5

    t_euler, y_euler = euler_method(f_decay, y0, t_span, dt)
    t_rk4, y_rk4 = rk4_method(f_decay, y0, t_span, dt)
    t_exact = np.linspace(0, 5, 100)
    y_exact = np.exp(-t_exact)

    ax1.plot(t_exact, y_exact, 'k-', linewidth=2, label='Exact')
    ax1.plot(t_euler, y_euler[:, 0], 'ro--', markersize=6, label='Euler')
    ax1.plot(t_rk4, y_rk4[:, 0], 'bs--', markersize=6, label='RK4')
    ax1.set_xlabel('t')
    ax1.set_ylabel('y')
    ax1.set_title('Exponential Decay: dy/dt = -y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Harmonic oscillator
    ax2 = plt.subplot(2, 3, 2)
    omega = 2.0
    y0 = np.array([1.0, 0.0])
    t_span = (0, 10)
    dt = 0.05

    t_rk4, y_rk4 = rk4_method(lambda t, y: harmonic_oscillator(t, y, omega),
                              y0, t_span, dt)
    y_exact = np.cos(omega * t_rk4)

    ax2.plot(t_rk4, y_exact, 'k--', linewidth=2, alpha=0.5, label='Exact')
    ax2.plot(t_rk4, y_rk4[:, 0], 'b-', linewidth=1.5, label='RK4')
    ax2.set_xlabel('t')
    ax2.set_ylabel('y')
    ax2.set_title(f'Harmonic Oscillator (ω={omega})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Phase portrait - harmonic oscillator
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(y_rk4[:, 0], y_rk4[:, 1], 'b-', linewidth=1.5)
    ax3.plot(y_rk4[0, 0], y_rk4[0, 1], 'go', markersize=8, label='Start')
    ax3.set_xlabel('y')
    ax3.set_ylabel("y'")
    ax3.set_title('Phase Portrait: Harmonic Oscillator')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # Plot 4: Damped oscillators
    ax4 = plt.subplot(2, 3, 4)
    for gamma in [0.1, 0.5, 1.0, 2.0]:
        t_rk4, y_rk4 = rk4_method(lambda t, y: damped_oscillator(t, y, gamma, 1.0),
                                  np.array([1.0, 0.0]), (0, 20), 0.05)
        ax4.plot(t_rk4, y_rk4[:, 0], linewidth=1.5, label=f'γ={gamma}')

    ax4.set_xlabel('t')
    ax4.set_ylabel('y')
    ax4.set_title('Damped Oscillators')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Lorenz attractor (3D)
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax5.plot(y_lorenz[:, 0], y_lorenz[:, 1], y_lorenz[:, 2],
            'b-', linewidth=0.5, alpha=0.7)
    ax5.plot([y_lorenz[0, 0]], [y_lorenz[0, 1]], [y_lorenz[0, 2]],
            'go', markersize=6)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_zlabel('z')
    ax5.set_title('Lorenz Attractor')

    # Plot 6: Van der Pol limit cycle
    ax6 = plt.subplot(2, 3, 6)
    for mu in [0.5, 2.0]:
        t_vdp, y_vdp = rk4_method(lambda t, y: van_der_pol(t, y, mu),
                                  np.array([2.0, 0.0]), (0, 50), 0.05)
        # Plot only steady-state portion
        ax6.plot(y_vdp[1000:, 0], y_vdp[1000:, 1],
                linewidth=1.5, label=f'μ={mu}')

    ax6.set_xlabel('y')
    ax6.set_ylabel("y'")
    ax6.set_title('Van der Pol Limit Cycles')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Mathematical_Methods/07_ode.png', dpi=150)
    print("Saved visualization: 07_ode.png")
    plt.close()

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
