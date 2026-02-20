"""
Special Functions - Bessel, Legendre, Hermite, Laguerre, Spherical Harmonics

This script demonstrates:
- Bessel functions J_n(x) and Y_n(x)
- Legendre polynomials P_n(x)
- Hermite polynomials H_n(x)
- Laguerre polynomials L_n(x)
- Spherical harmonics Y_l^m(θ,φ)
- Gamma function Γ(x)
- Orthogonality properties
"""

import numpy as np

try:
    from scipy.special import (jv, yn, eval_legendre, eval_hermite,
                               eval_laguerre, sph_harm, gamma, factorial)
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using limited implementations")
    # Provide basic gamma function
    gamma = lambda x: np.exp(np.sum(np.log(np.arange(1, int(x)))))

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def legendre_poly_manual(n, x):
    """Compute Legendre polynomial P_n(x) using recurrence relation"""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x

    P_prev2 = np.ones_like(x)
    P_prev1 = x

    for k in range(2, n + 1):
        P_n = ((2*k - 1) * x * P_prev1 - (k - 1) * P_prev2) / k
        P_prev2 = P_prev1
        P_prev1 = P_n

    return P_n


def hermite_poly_manual(n, x):
    """Compute Hermite polynomial H_n(x) using recurrence relation"""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x

    H_prev2 = np.ones_like(x)
    H_prev1 = 2 * x

    for k in range(2, n + 1):
        H_n = 2 * x * H_prev1 - 2 * (k - 1) * H_prev2
        H_prev2 = H_prev1
        H_prev1 = H_n

    return H_n


def laguerre_poly_manual(n, x):
    """Compute Laguerre polynomial L_n(x) using recurrence relation"""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 1 - x

    L_prev2 = np.ones_like(x)
    L_prev1 = 1 - x

    for k in range(2, n + 1):
        L_n = ((2*k - 1 - x) * L_prev1 - (k - 1) * L_prev2) / k
        L_prev2 = L_prev1
        L_prev1 = L_n

    return L_n


def check_orthogonality_legendre(n, m, num_points=1000):
    """
    Check orthogonality of Legendre polynomials:
    ∫_{-1}^{1} P_n(x) P_m(x) dx = 0 if n≠m, = 2/(2n+1) if n=m
    """
    x = np.linspace(-1, 1, num_points)
    if HAS_SCIPY:
        P_n = eval_legendre(n, x)
        P_m = eval_legendre(m, x)
    else:
        P_n = legendre_poly_manual(n, x)
        P_m = legendre_poly_manual(m, x)

    integrand = P_n * P_m
    integral = np.trapz(integrand, x)

    return integral


def check_orthogonality_hermite(n, m, num_points=500):
    """
    Check orthogonality of Hermite polynomials:
    ∫_{-∞}^{∞} H_n(x) H_m(x) e^(-x²) dx = 0 if n≠m
    """
    x = np.linspace(-5, 5, num_points)
    if HAS_SCIPY:
        H_n = eval_hermite(n, x)
        H_m = eval_hermite(m, x)
    else:
        H_n = hermite_poly_manual(n, x)
        H_m = hermite_poly_manual(m, x)

    weight = np.exp(-x**2)
    integrand = H_n * H_m * weight
    integral = np.trapz(integrand, x)

    return integral


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

print("=" * 70)
print("SPECIAL FUNCTIONS - BESSEL, LEGENDRE, HERMITE, LAGUERRE")
print("=" * 70)

# Test 1: Bessel functions
if HAS_SCIPY:
    print("\n1. BESSEL FUNCTIONS J_n(x)")
    print("-" * 70)
    x_vals = [1.0, 5.0, 10.0]
    n_vals = [0, 1, 2]

    print(f"{'x':>6s}", end='')
    for n in n_vals:
        print(f"{'J_'+str(n)+'(x)':>12s}", end='')
    print()
    print("-" * 70)

    for x in x_vals:
        print(f"{x:6.1f}", end='')
        for n in n_vals:
            J_n = jv(n, x)
            print(f"{J_n:12.6f}", end='')
        print()

    # Bessel function zeros (important for boundary value problems)
    print("\nFirst 3 zeros of J_0(x):")
    x_range = np.linspace(0.1, 15, 500)
    J_0 = jv(0, x_range)

    # Find approximate zeros
    zeros = []
    for i in range(1, len(x_range)):
        if J_0[i-1] * J_0[i] < 0:
            zeros.append(x_range[i])
            if len(zeros) >= 3:
                break

    for i, zero in enumerate(zeros):
        print(f"  x_{i+1} ≈ {zero:.4f}")

    # Test 2: Legendre polynomials
    print("\n2. LEGENDRE POLYNOMIALS P_n(x)")
    print("-" * 70)
    x_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    n_vals = [0, 1, 2, 3]

    print(f"{'x':>6s}", end='')
    for n in n_vals:
        print(f"{'P_'+str(n)+'(x)':>10s}", end='')
    print()
    print("-" * 70)

    for x in x_vals:
        print(f"{x:6.1f}", end='')
        for n in n_vals:
            P_n = eval_legendre(n, x)
            print(f"{P_n:10.4f}", end='')
        print()

    # Orthogonality check
    print("\nOrthogonality check: ∫₋₁¹ P_n(x)P_m(x)dx")
    print("Expected: 0 if n≠m, 2/(2n+1) if n=m")
    for n in range(3):
        for m in range(3):
            integral = check_orthogonality_legendre(n, m)
            expected = 2/(2*n+1) if n == m else 0
            print(f"  ∫P_{n}P_{m} = {integral:8.4f} (expected: {expected:6.4f})")

else:
    print("\n1-2. LEGENDRE POLYNOMIALS (manual implementation)")
    print("-" * 70)
    x_vals = [0.0, 0.5, 1.0]
    for x in x_vals:
        print(f"\nx = {x}:")
        for n in range(4):
            P_n = legendre_poly_manual(n, np.array([x]))[0]
            print(f"  P_{n}({x}) = {P_n:.4f}")

# Test 3: Hermite polynomials
if HAS_SCIPY:
    print("\n3. HERMITE POLYNOMIALS H_n(x)")
    print("-" * 70)
    x_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
    n_vals = [0, 1, 2, 3]

    print(f"{'x':>6s}", end='')
    for n in n_vals:
        print(f"{'H_'+str(n)+'(x)':>10s}", end='')
    print()
    print("-" * 70)

    for x in x_vals:
        print(f"{x:6.1f}", end='')
        for n in n_vals:
            H_n = eval_hermite(n, x)
            print(f"{H_n:10.2f}", end='')
        print()

    # Connection to quantum harmonic oscillator
    print("\nQuantum harmonic oscillator wavefunctions:")
    print("ψ_n(x) ∝ H_n(x) exp(-x²/2)")
    x = 0.0
    for n in range(4):
        H_n = eval_hermite(n, x)
        psi_n = H_n * np.exp(-x**2 / 2)
        print(f"  ψ_{n}(0) ∝ {psi_n:.4f}")

# Test 4: Laguerre polynomials
if HAS_SCIPY:
    print("\n4. LAGUERRE POLYNOMIALS L_n(x)")
    print("-" * 70)
    x_vals = [0.0, 1.0, 2.0, 3.0]
    n_vals = [0, 1, 2, 3]

    print(f"{'x':>6s}", end='')
    for n in n_vals:
        print(f"{'L_'+str(n)+'(x)':>10s}", end='')
    print()
    print("-" * 70)

    for x in x_vals:
        print(f"{x:6.1f}", end='')
        for n in n_vals:
            L_n = eval_laguerre(n, x)
            print(f"{L_n:10.4f}", end='')
        print()

    # Connection to hydrogen atom
    print("\nHydrogen atom radial wavefunctions involve Laguerre polynomials")
    print("R_nl(r) ∝ r^l L_{n-l-1}^{2l+1}(2r/na₀) exp(-r/na₀)")

# Test 5: Spherical harmonics
if HAS_SCIPY:
    print("\n5. SPHERICAL HARMONICS Y_l^m(θ,φ)")
    print("-" * 70)
    theta_vals = [0, np.pi/4, np.pi/2]
    phi = 0

    print("At φ=0:")
    print(f"{'θ':>10s} {'Y_0^0':>15s} {'Y_1^0':>15s} {'Y_2^0':>15s}")
    print("-" * 70)

    for theta in theta_vals:
        print(f"{theta:10.4f}", end='')
        for l in [0, 1, 2]:
            m = 0
            Y_lm = sph_harm(m, l, phi, theta)
            # Real part (imaginary part is zero for m=0)
            print(f"{Y_lm.real:15.6f}", end='')
        print()

    # Check normalization
    print("\nNormalization check (numerical integration):")
    n_theta = 50
    n_phi = 100
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    Theta, Phi = np.meshgrid(theta, phi)

    for l in [0, 1, 2]:
        for m in range(-l, l+1):
            Y_lm = sph_harm(m, l, Phi, Theta)
            integrand = np.abs(Y_lm)**2 * np.sin(Theta)

            dtheta = np.pi / (n_theta - 1)
            dphi = 2*np.pi / (n_phi - 1)
            integral = np.sum(integrand) * dtheta * dphi

            print(f"  ∫|Y_{l}^{m:2d}|² dΩ = {integral:.4f} (expected: 1.0)")

# Test 6: Gamma function
print("\n6. GAMMA FUNCTION Γ(x)")
print("-" * 70)
print("Γ(n+1) = n! for integer n")
print(f"{'n':>4s} {'n!':>12s} {'Γ(n+1)':>12s}")
print("-" * 70)

for n in range(1, 8):
    factorial_n = np.math.factorial(n)
    gamma_n_plus_1 = gamma(n + 1)
    print(f"{n:4d} {factorial_n:12.0f} {gamma_n_plus_1:12.6f}")

print("\nΓ(x) for half-integer values:")
print(f"Γ(1/2) = √π = {gamma(0.5):.6f} (expected: {np.sqrt(np.pi):.6f})")
print(f"Γ(3/2) = √π/2 = {gamma(1.5):.6f} (expected: {np.sqrt(np.pi)/2:.6f})")
print(f"Γ(5/2) = 3√π/4 = {gamma(2.5):.6f} (expected: {3*np.sqrt(np.pi)/4:.6f})")

# Visualization
if HAS_MATPLOTLIB and HAS_SCIPY:
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    fig = plt.figure(figsize=(15, 12))

    # Plot 1: Bessel functions J_n
    ax1 = plt.subplot(3, 3, 1)
    x = np.linspace(0, 15, 500)
    for n in range(4):
        J_n = jv(n, x)
        ax1.plot(x, J_n, linewidth=2, label=f'$J_{n}(x)$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('$J_n(x)$')
    ax1.set_title('Bessel Functions of First Kind')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 1.0)

    # Plot 2: Bessel functions Y_n
    ax2 = plt.subplot(3, 3, 2)
    x = np.linspace(0.1, 15, 500)
    for n in range(3):
        Y_n = yn(n, x)
        ax2.plot(x, Y_n, linewidth=2, label=f'$Y_{n}(x)$')
    ax2.set_xlabel('x')
    ax2.set_ylabel('$Y_n(x)$')
    ax2.set_title('Bessel Functions of Second Kind')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-2, 1)

    # Plot 3: Legendre polynomials
    ax3 = plt.subplot(3, 3, 3)
    x = np.linspace(-1, 1, 500)
    for n in range(6):
        P_n = eval_legendre(n, x)
        ax3.plot(x, P_n, linewidth=2, label=f'$P_{n}(x)$')
    ax3.set_xlabel('x')
    ax3.set_ylabel('$P_n(x)$')
    ax3.set_title('Legendre Polynomials')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.2, 1.2)

    # Plot 4: Hermite polynomials
    ax4 = plt.subplot(3, 3, 4)
    x = np.linspace(-3, 3, 500)
    for n in range(5):
        H_n = eval_hermite(n, x)
        ax4.plot(x, H_n, linewidth=2, label=f'$H_{n}(x)$')
    ax4.set_xlabel('x')
    ax4.set_ylabel('$H_n(x)$')
    ax4.set_title('Hermite Polynomials')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-50, 50)

    # Plot 5: Quantum harmonic oscillator wavefunctions
    ax5 = plt.subplot(3, 3, 5)
    x = np.linspace(-4, 4, 500)
    for n in range(4):
        H_n = eval_hermite(n, x)
        # Normalized wavefunction
        norm = 1 / np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi))
        psi_n = norm * H_n * np.exp(-x**2 / 2)
        ax5.plot(x, psi_n + n, linewidth=2, label=f'$\\psi_{n}(x)$')

    ax5.set_xlabel('x')
    ax5.set_ylabel('$\\psi_n(x)$ + offset')
    ax5.set_title('Quantum Harmonic Oscillator')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Laguerre polynomials
    ax6 = plt.subplot(3, 3, 6)
    x = np.linspace(0, 10, 500)
    for n in range(5):
        L_n = eval_laguerre(n, x)
        ax6.plot(x, L_n, linewidth=2, label=f'$L_{n}(x)$')
    ax6.set_xlabel('x')
    ax6.set_ylabel('$L_n(x)$')
    ax6.set_title('Laguerre Polynomials')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-5, 5)

    # Plot 7: Spherical harmonic |Y_2^0|
    ax7 = fig.add_subplot(3, 3, 7, projection='3d')
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 100)
    Theta, Phi = np.meshgrid(theta, phi)

    l, m = 2, 0
    Y_lm = sph_harm(m, l, Phi, Theta)
    r = np.abs(Y_lm)

    x_sph = r * np.sin(Theta) * np.cos(Phi)
    y_sph = r * np.sin(Theta) * np.sin(Phi)
    z_sph = r * np.cos(Theta)

    ax7.plot_surface(x_sph, y_sph, z_sph, cmap='viridis', alpha=0.8)
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    ax7.set_zlabel('z')
    ax7.set_title('$|Y_2^0(\\theta,\\phi)|$')

    # Plot 8: Gamma function
    ax8 = plt.subplot(3, 3, 8)
    x_pos = np.linspace(0.1, 5, 500)
    y_gamma = gamma(x_pos)

    ax8.plot(x_pos, y_gamma, 'b-', linewidth=2)
    ax8.plot([1, 2, 3, 4], [gamma(1), gamma(2), gamma(3), gamma(4)],
            'ro', markersize=8, label='Integer points')
    ax8.set_xlabel('x')
    ax8.set_ylabel('$\\Gamma(x)$')
    ax8.set_title('Gamma Function')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 25)

    # Plot 9: Associated Legendre functions (used in spherical harmonics)
    ax9 = plt.subplot(3, 3, 9)
    x = np.linspace(-1, 1, 500)
    from scipy.special import lpmv

    for m in range(3):
        P_2m = lpmv(m, 2, x)
        ax9.plot(x, P_2m, linewidth=2, label=f'$P_2^{m}(x)$')

    ax9.set_xlabel('x')
    ax9.set_ylabel('$P_l^m(x)$')
    ax9.set_title('Associated Legendre Functions (l=2)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Mathematical_Methods/08_special_functions.png', dpi=150)
    print("Saved visualization: 08_special_functions.png")
    plt.close()

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
