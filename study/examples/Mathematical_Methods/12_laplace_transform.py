"""
Laplace Transform - Transform Pairs, Inverse Transform, and ODE Solutions

This script demonstrates:
- Laplace transform pairs
- Inverse Laplace transform (numerical)
- Solving ODEs using Laplace transform
- Transfer functions
- Step response
- Frequency response
- Convolution theorem
"""

import numpy as np

try:
    from scipy.integrate import quad
    from scipy.signal import lti, step, impulse, bode
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using limited implementations")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def laplace_transform(f, s, t_max=10, n_points=1000):
    """
    Compute Laplace transform: F(s) = ∫₀^∞ f(t)e^(-st) dt
    Numerically integrate from 0 to t_max
    """
    if HAS_SCIPY:
        integrand = lambda t: f(t) * np.exp(-s * t)
        result, error = quad(integrand, 0, t_max)
        return result
    else:
        t = np.linspace(0, t_max, n_points)
        integrand = f(t) * np.exp(-s * t)
        result = np.trapz(integrand, t)
        return result


def inverse_laplace_bromwich(F, t, gamma=0.5, n_points=100):
    """
    Inverse Laplace transform using Bromwich integral (simplified)
    f(t) = (1/2πi) ∫_{γ-i∞}^{γ+i∞} F(s)e^(st) ds

    Use finite limits for practical computation
    """
    omega = np.linspace(-50, 50, n_points)
    s = gamma + 1j * omega
    integrand = F(s) * np.exp(s * t)

    # Trapezoidal integration
    result = np.trapz(integrand, omega) / (2 * np.pi)
    return result.real


class LaplacePairs:
    """Common Laplace transform pairs"""

    @staticmethod
    def unit_step():
        """u(t) ↔ 1/s"""
        f = lambda t: 1.0
        F = lambda s: 1 / s
        return f, F

    @staticmethod
    def exponential(a):
        """e^(at) ↔ 1/(s-a)"""
        f = lambda t: np.exp(a * t)
        F = lambda s: 1 / (s - a)
        return f, F

    @staticmethod
    def sine(omega):
        """sin(ωt) ↔ ω/(s²+ω²)"""
        f = lambda t: np.sin(omega * t)
        F = lambda s: omega / (s**2 + omega**2)
        return f, F

    @staticmethod
    def cosine(omega):
        """cos(ωt) ↔ s/(s²+ω²)"""
        f = lambda t: np.cos(omega * t)
        F = lambda s: s / (s**2 + omega**2)
        return f, F

    @staticmethod
    def damped_sine(a, omega):
        """e^(-at)sin(ωt) ↔ ω/((s+a)²+ω²)"""
        f = lambda t: np.exp(-a * t) * np.sin(omega * t)
        F = lambda s: omega / ((s + a)**2 + omega**2)
        return f, F

    @staticmethod
    def power(n):
        """t^n ↔ n!/s^(n+1)"""
        f = lambda t: t**n
        F = lambda s: np.math.factorial(n) / s**(n+1)
        return f, F


def solve_ode_laplace_first_order(a, b, y0):
    """
    Solve first-order ODE: y' + ay = b using Laplace transform
    Returns solution function y(t)
    """
    # Laplace transform: sY(s) - y(0) + aY(s) = b/s
    # Y(s) = (y0 + b/s) / (s + a)

    def y(t):
        # Inverse transform: y(t) = y0*e^(-at) + (b/a)(1 - e^(-at))
        return y0 * np.exp(-a * t) + (b / a) * (1 - np.exp(-a * t))

    return y


def solve_ode_laplace_second_order(a, b, c, y0, yp0):
    """
    Solve second-order ODE: y'' + ay' + by = c using Laplace transform
    Returns solution function y(t)
    """
    # Laplace: s²Y(s) - sy(0) - y'(0) + a(sY(s) - y(0)) + bY(s) = c/s
    # Y(s) = [sy(0) + y'(0) + ay(0) + c/s] / (s² + as + b)

    # Characteristic equation: s² + as + b = 0
    discriminant = a**2 - 4*b

    if discriminant > 0:
        # Overdamped
        r1 = (-a + np.sqrt(discriminant)) / 2
        r2 = (-a - np.sqrt(discriminant)) / 2

        def y(t):
            # Particular solution: y_p = c/b
            y_p = c / b
            # Homogeneous solution
            A = (yp0 - r2*(y0 - y_p)) / (r1 - r2)
            B = (r1*(y0 - y_p) - yp0) / (r1 - r2)
            return A * np.exp(r1 * t) + B * np.exp(r2 * t) + y_p

    elif discriminant == 0:
        # Critically damped
        r = -a / 2

        def y(t):
            y_p = c / b
            A = y0 - y_p
            B = yp0 - r * A
            return (A + B * t) * np.exp(r * t) + y_p

    else:
        # Underdamped
        sigma = -a / 2
        omega = np.sqrt(-discriminant) / 2

        def y(t):
            y_p = c / b
            A = y0 - y_p
            B = (yp0 - sigma * A) / omega
            return np.exp(sigma * t) * (A * np.cos(omega * t) + B * np.sin(omega * t)) + y_p

    return y


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

print("=" * 70)
print("LAPLACE TRANSFORM - PAIRS, INVERSE, AND ODE SOLUTIONS")
print("=" * 70)

# Test 1: Verify transform pairs
print("\n1. LAPLACE TRANSFORM PAIRS")
print("-" * 70)

pairs = [
    ("Unit step", LaplacePairs.unit_step()),
    ("e^(-2t)", LaplacePairs.exponential(-2)),
    ("sin(3t)", LaplacePairs.sine(3)),
    ("cos(2t)", LaplacePairs.cosine(2)),
    ("t²", LaplacePairs.power(2)),
]

s_test = 2.0
print(f"Testing at s = {s_test}")
print(f"\n{'Function':20s} {'F(s) formula':20s} {'Numerical':15s}")
print("-" * 70)

for name, (f, F) in pairs:
    F_formula = F(s_test)
    F_numerical = laplace_transform(f, s_test, t_max=10)
    print(f"{name:20s} {F_formula:20.6f} {F_numerical:15.6f}")

# Test 2: Inverse Laplace transform
print("\n2. INVERSE LAPLACE TRANSFORM")
print("-" * 70)

# F(s) = 1/(s+1) → f(t) = e^(-t)
F_exp = lambda s: 1 / (s + 1)
t_test = 1.0
f_inverse = inverse_laplace_bromwich(F_exp, t_test, gamma=0.5)
f_exact = np.exp(-t_test)

print(f"F(s) = 1/(s+1), expected: f(t) = e^(-t)")
print(f"At t = {t_test}:")
print(f"  Inverse transform: {f_inverse:.6f}")
print(f"  Exact:             {f_exact:.6f}")
print(f"  Error:             {abs(f_inverse - f_exact):.2e}")

# F(s) = ω/(s²+ω²) → f(t) = sin(ωt)
omega = 2.0
F_sin = lambda s: omega / (s**2 + omega**2)
t_test = np.pi / (2 * omega)  # Peak of sine
f_inverse = inverse_laplace_bromwich(F_sin, t_test, gamma=0.5)
f_exact = np.sin(omega * t_test)

print(f"\nF(s) = {omega}/(s²+{omega**2}), expected: f(t) = sin({omega}t)")
print(f"At t = {t_test:.4f}:")
print(f"  Inverse transform: {f_inverse:.6f}")
print(f"  Exact:             {f_exact:.6f}")

# Test 3: Solving first-order ODE
print("\n3. SOLVING FIRST-ORDER ODE: y' + 2y = 4, y(0) = 0")
print("-" * 70)

a, b, y0 = 2.0, 4.0, 0.0
y_solution = solve_ode_laplace_first_order(a, b, y0)

print(f"ODE: y' + {a}y = {b}")
print(f"Initial condition: y(0) = {y0}")
print(f"\nSolution: y(t) = 2(1 - e^(-2t))")

t_vals = [0, 0.5, 1.0, 2.0, 5.0]
print(f"\n{'t':>6s} {'y(t)':>12s}")
print("-" * 20)
for t in t_vals:
    print(f"{t:6.1f} {y_solution(t):12.6f}")

print(f"\nSteady-state (t→∞): y = {b/a:.6f}")

# Test 4: Solving second-order ODE
print("\n4. SOLVING SECOND-ORDER ODE: y'' + 3y' + 2y = 4, y(0)=0, y'(0)=0")
print("-" * 70)

a, b, c = 3.0, 2.0, 4.0
y0, yp0 = 0.0, 0.0
y_solution = solve_ode_laplace_second_order(a, b, c, y0, yp0)

print(f"ODE: y'' + {a}y' + {b}y = {c}")
print(f"Initial conditions: y(0) = {y0}, y'(0) = {yp0}")

# Characteristic roots
discriminant = a**2 - 4*b
r1 = (-a + np.sqrt(discriminant)) / 2
r2 = (-a - np.sqrt(discriminant)) / 2
print(f"Characteristic roots: r₁ = {r1:.4f}, r₂ = {r2:.4f}")
print(f"System is overdamped")

t_vals = [0, 0.5, 1.0, 2.0, 5.0]
print(f"\n{'t':>6s} {'y(t)':>12s}")
print("-" * 20)
for t in t_vals:
    print(f"{t:6.1f} {y_solution(t):12.6f}")

print(f"\nSteady-state (t→∞): y = {c/b:.6f}")

# Test 5: Underdamped oscillator
print("\n5. UNDERDAMPED OSCILLATOR: y'' + 0.4y' + 4y = 0, y(0)=1, y'(0)=0")
print("-" * 70)

a, b, c = 0.4, 4.0, 0.0
y0, yp0 = 1.0, 0.0
y_solution = solve_ode_laplace_second_order(a, b, c, y0, yp0)

discriminant = a**2 - 4*b
print(f"Discriminant: {discriminant:.4f} < 0 (underdamped)")

sigma = -a / 2
omega_d = np.sqrt(-discriminant) / 2
print(f"Damping coefficient: σ = {sigma:.4f}")
print(f"Damped frequency: ωd = {omega_d:.4f}")

t_vals = [0, 0.5, 1.0, 2.0, 5.0]
print(f"\n{'t':>6s} {'y(t)':>12s}")
print("-" * 20)
for t in t_vals:
    print(f"{t:6.1f} {y_solution(t):12.6f}")

# Test 6: Transfer function (if scipy available)
if HAS_SCIPY:
    print("\n6. TRANSFER FUNCTION AND FREQUENCY RESPONSE")
    print("-" * 70)

    # Second-order system: H(s) = ω_n² / (s² + 2ζω_n s + ω_n²)
    omega_n = 2.0  # Natural frequency
    zeta = 0.3     # Damping ratio

    num = [omega_n**2]
    den = [1, 2*zeta*omega_n, omega_n**2]

    print(f"Transfer function: H(s) = {omega_n**2}/(s² + {2*zeta*omega_n}s + {omega_n**2})")
    print(f"Natural frequency ωn = {omega_n} rad/s")
    print(f"Damping ratio ζ = {zeta}")

    # Create system
    system = lti(num, den)

    # Peak frequency
    if zeta < 1/np.sqrt(2):
        omega_peak = omega_n * np.sqrt(1 - 2*zeta**2)
        print(f"Resonant peak at ω = {omega_peak:.4f} rad/s")

# Test 7: Convolution theorem
print("\n7. CONVOLUTION THEOREM")
print("-" * 70)

print("Theorem: L{f*g} = F(s)·G(s)")
print("Example: f(t) = e^(-t), g(t) = e^(-2t)")

# Functions
f = lambda t: np.exp(-t)
g = lambda t: np.exp(-2*t)

# Laplace transforms
F = lambda s: 1 / (s + 1)
G = lambda s: 1 / (s + 2)

# Product in s-domain
FG = lambda s: F(s) * G(s)

# Analytical convolution: (f*g)(t) = ∫₀^t e^(-τ)e^(-2(t-τ))dτ = e^(-t) - e^(-2t)
convolution_exact = lambda t: np.exp(-t) - np.exp(-2*t)

# Verify at s=3
s_test = 3.0
FG_value = FG(s_test)
L_convolution = laplace_transform(convolution_exact, s_test, t_max=10)

print(f"\nAt s = {s_test}:")
print(f"  F(s)·G(s) = {FG_value:.6f}")
print(f"  L{{f*g}} = {L_convolution:.6f}")
print(f"  Error: {abs(FG_value - L_convolution):.2e}")

# Visualization
if HAS_MATPLOTLIB:
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Common transform pairs
    ax1 = plt.subplot(2, 3, 1)
    t = np.linspace(0, 5, 500)

    functions = [
        (lambda t: np.exp(-t), 'e^(-t)'),
        (lambda t: np.exp(-2*t), 'e^(-2t)'),
        (lambda t: t * np.exp(-t), 'te^(-t)'),
    ]

    for f, label in functions:
        ax1.plot(t, f(t), linewidth=2, label=label)

    ax1.set_xlabel('t')
    ax1.set_ylabel('f(t)')
    ax1.set_title('Time Domain Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Step response of first-order system
    ax2 = plt.subplot(2, 3, 2)
    t = np.linspace(0, 5, 500)

    for a in [0.5, 1.0, 2.0]:
        y_sol = solve_ode_laplace_first_order(a, 1.0, 0.0)
        y = np.array([y_sol(ti) for ti in t])
        ax2.plot(t, y, linewidth=2, label=f'a={a}')

    ax2.set_xlabel('t')
    ax2.set_ylabel('y(t)')
    ax2.set_title("First-Order Step Response: y' + ay = 1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Second-order responses
    ax3 = plt.subplot(2, 3, 3)
    t = np.linspace(0, 10, 500)

    # Different damping ratios
    configs = [
        (0.1, 'Underdamped ζ=0.05'),
        (0.4, 'Underdamped ζ=0.1'),
        (4.0, 'Critically damped'),
    ]

    for a, label in configs:
        y_sol = solve_ode_laplace_second_order(a, 4.0, 4.0, 0.0, 0.0)
        y = np.array([y_sol(ti) for ti in t])
        ax3.plot(t, y, linewidth=2, label=label)

    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('t')
    ax3.set_ylabel('y(t)')
    ax3.set_title("Second-Order Step Response")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Damped oscillation
    ax4 = plt.subplot(2, 3, 4)
    t = np.linspace(0, 15, 500)

    y_sol = solve_ode_laplace_second_order(0.4, 4.0, 0.0, 1.0, 0.0)
    y = np.array([y_sol(ti) for ti in t])

    # Envelope
    sigma = -0.4 / 2
    envelope_pos = np.exp(sigma * t)
    envelope_neg = -np.exp(sigma * t)

    ax4.plot(t, y, 'b-', linewidth=2, label='y(t)')
    ax4.plot(t, envelope_pos, 'r--', alpha=0.5, label='Envelope')
    ax4.plot(t, envelope_neg, 'r--', alpha=0.5)
    ax4.set_xlabel('t')
    ax4.set_ylabel('y(t)')
    ax4.set_title('Underdamped Oscillator')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Frequency response (if scipy available)
    if HAS_SCIPY:
        ax5 = plt.subplot(2, 3, 5)

        omega_n = 2.0
        zeta = 0.3
        num = [omega_n**2]
        den = [1, 2*zeta*omega_n, omega_n**2]
        system = lti(num, den)

        w, mag, phase = bode(system, np.logspace(-1, 1, 100))

        ax5.semilogx(w, mag, 'b-', linewidth=2)
        ax5.set_xlabel('Frequency (rad/s)')
        ax5.set_ylabel('Magnitude (dB)')
        ax5.set_title(f'Bode Plot: ζ={zeta}, ωn={omega_n}')
        ax5.grid(True, alpha=0.3, which='both')

        # Plot 6: Phase response
        ax6 = plt.subplot(2, 3, 6)
        ax6.semilogx(w, phase, 'r-', linewidth=2)
        ax6.set_xlabel('Frequency (rad/s)')
        ax6.set_ylabel('Phase (deg)')
        ax6.set_title('Phase Response')
        ax6.grid(True, alpha=0.3, which='both')
    else:
        # Alternative plots
        ax5 = plt.subplot(2, 3, 5)
        t = np.linspace(0, 5, 500)
        f1 = np.exp(-t)
        f2 = np.exp(-2*t)

        # Numerical convolution
        dt = t[1] - t[0]
        conv = np.convolve(f1, f2, mode='same') * dt

        ax5.plot(t, conv, 'b-', linewidth=2, label='f*g (numerical)')
        ax5.plot(t, np.exp(-t) - np.exp(-2*t), 'r--', linewidth=2, label='Analytical')
        ax5.set_xlabel('t')
        ax5.set_ylabel('(f*g)(t)')
        ax5.set_title('Convolution: e^(-t) * e^(-2t)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Mathematical_Methods/12_laplace_transform.png', dpi=150)
    print("Saved visualization: 12_laplace_transform.png")
    plt.close()

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
