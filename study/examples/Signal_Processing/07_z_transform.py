#!/usr/bin/env python3
"""
Z-Transform Analysis
====================

This script demonstrates the Z-transform and its applications in discrete-time
signal processing, including:

- Common Z-transform pairs and their verification
- Pole-zero diagrams with unit circle and stability regions
- System stability analysis from pole locations
- Inverse Z-transform via partial fraction expansion
- 2nd-order IIR system analysis (poles, zeros, frequency response)

Key Concepts:
    X(z) = sum_{n=-inf}^{inf} x[n] * z^{-n}
    - Poles inside the unit circle → stable causal system
    - Poles outside the unit circle → unstable causal system
    - Poles on the unit circle → marginally stable

Author: Educational example for Signal Processing
License: MIT
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# ============================================================================
# Z-TRANSFORM UTILITIES
# ============================================================================

def compute_z_transform_samples(x, z_values):
    """
    Numerically evaluate X(z) = sum_{n=0}^{N-1} x[n] * z^{-n}
    for a finite-length sequence x[n].

    Args:
        x (ndarray): Input sequence (assumed causal, starting at n=0)
        z_values (ndarray): Complex z values at which to evaluate X(z)

    Returns:
        ndarray: X(z) evaluated at each z value
    """
    N = len(x)
    n = np.arange(N)
    # X(z) = sum x[n] * z^{-n}  — broadcast over z_values
    # Shape: (len(z_values), N), then sum along axis=1
    z_inv_n = z_values[:, np.newaxis] ** (-n[np.newaxis, :])
    return (x[np.newaxis, :] * z_inv_n).sum(axis=1)


def partial_fraction_inverse(b, a, n_samples=20):
    """
    Compute inverse Z-transform via partial fraction expansion.

    Uses scipy.signal.residuez to decompose H(z) = B(z)/A(z) into:
        H(z) = sum_k  r_k / (1 - p_k * z^{-1})  + direct terms

    The inverse Z-transform of r_k / (1 - p_k * z^{-1}) is r_k * p_k^n * u[n].

    Args:
        b (array-like): Numerator polynomial coefficients (descending z powers)
        a (array-like): Denominator polynomial coefficients
        n_samples (int): Number of samples for h[n]

    Returns:
        tuple: (h, residues, poles, direct_terms)
    """
    # residuez works with z^{-1} polynomials (signal convention)
    r, p, k = signal.residuez(b, a)

    # Reconstruct h[n] = sum_i r[i] * p[i]^n  for n >= 0
    # The imaginary parts of complex-conjugate pairs cancel exactly;
    # taking np.real() discards the numerical rounding residual.
    n = np.arange(n_samples)
    h = np.zeros(n_samples, dtype=complex)
    for ri, pi in zip(r, p):
        h += ri * pi ** n
    h = np.real(h)   # imaginary part is ~machine-epsilon for real-valued systems

    return h, r, p, k


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

def demo_common_z_transforms():
    """Verify common Z-transform pairs numerically."""
    print("=" * 65)
    print("1. COMMON Z-TRANSFORM PAIRS — NUMERICAL VERIFICATION")
    print("=" * 65)

    # Evaluate on a circle of radius r = 1.2 (inside ROC for stable signals)
    theta = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    r = 1.2
    z = r * np.exp(1j * theta)

    # --- Pair 1: Unit step u[n]  →  z / (z - 1),  |z| > 1
    N = 200  # approximate infinite sum with long finite sequence
    x_step = np.ones(N)
    X_numerical = compute_z_transform_samples(x_step, z)
    X_analytical = z / (z - 1)
    err_step = np.max(np.abs(X_numerical - X_analytical)) / np.max(np.abs(X_analytical))
    print(f"\nUnit step  u[n]  →  z/(z-1)")
    print(f"  Max relative error (N={N} terms): {err_step:.4e}")

    # --- Pair 2: Exponential a^n * u[n]  →  z / (z - a),  |z| > |a|
    a = 0.8
    x_exp = a ** np.arange(N)
    X_numerical = compute_z_transform_samples(x_exp, z)
    X_analytical = z / (z - a)
    err_exp = np.max(np.abs(X_numerical - X_analytical)) / np.max(np.abs(X_analytical))
    print(f"\nExponential  a^n·u[n]  (a={a})  →  z/(z-a)")
    print(f"  Max relative error (N={N} terms): {err_exp:.4e}")

    # --- Pair 3: Unit impulse δ[n]  →  1  (all z)
    x_imp = np.zeros(50); x_imp[0] = 1.0
    X_numerical = compute_z_transform_samples(x_imp, z)
    X_analytical = np.ones_like(z)
    err_imp = np.max(np.abs(X_numerical - X_analytical))
    print(f"\nUnit impulse  δ[n]  →  1")
    print(f"  Max absolute error: {err_imp:.4e}")

    # --- Pair 4: Delayed impulse δ[n-k]  →  z^{-k}
    k_delay = 3
    x_del = np.zeros(50); x_del[k_delay] = 1.0
    X_numerical = compute_z_transform_samples(x_del, z)
    X_analytical = z ** (-k_delay)
    err_del = np.max(np.abs(X_numerical - X_analytical))
    print(f"\nDelayed impulse  δ[n-{k_delay}]  →  z^{{-{k_delay}}}")
    print(f"  Max absolute error: {err_del:.4e}")


def demo_pole_zero_and_stability():
    """Plot pole-zero diagrams and assess stability."""
    print("\n" + "=" * 65)
    print("2. POLE-ZERO DIAGRAMS AND STABILITY ANALYSIS")
    print("=" * 65)

    # System 1: Stable — poles inside unit circle
    b1 = [1, -0.5]               # zeros of H1(z)
    a1 = [1, -0.9, 0.2]          # poles from denominator
    z1, p1, k1 = signal.tf2zpk(b1, a1)
    stable1 = np.all(np.abs(p1) < 1.0)
    print(f"\nSystem 1  H(z) = (z - 0.5) / (z^2 - 0.9z + 0.2)")
    print(f"  Zeros : {z1}")
    print(f"  Poles : {p1}")
    print(f"  |Poles|: {np.abs(p1)}")
    print(f"  Stable? {stable1}  (all poles inside unit circle)")

    # System 2: Unstable — one pole outside unit circle
    b2 = [1, 0]
    a2 = [1, -1.5, 0.56]         # roots at 0.8 and 0.7 — stable
    # Let's make one pole outside: a2 roots at 1.2 and 0.5
    a2_unstable = np.poly([1.2, 0.5])
    z2, p2, k2 = signal.tf2zpk(b2, a2_unstable)
    stable2 = np.all(np.abs(p2) < 1.0)
    print(f"\nSystem 2  (one pole outside unit circle)")
    print(f"  Poles : {p2}")
    print(f"  |Poles|: {np.abs(p2)}")
    print(f"  Stable? {stable2}")

    # System 3: Marginally stable — complex poles ON unit circle
    omega0 = np.pi / 4            # oscillation frequency
    p3 = np.array([np.exp(1j * omega0), np.exp(-1j * omega0)])
    b3, a3 = signal.zpk2tf([], p3, 1.0)
    z3, p3_check, _ = signal.tf2zpk(b3, a3)
    print(f"\nSystem 3  (complex poles on unit circle, ω₀=π/4)")
    print(f"  Poles : {np.round(p3_check, 4)}")
    print(f"  |Poles|: {np.abs(p3_check)}")
    print(f"  Marginally stable (oscillator, never decays or grows)")

    return (b1, a1, z1, p1), (b2, a2_unstable, z2, p2), (b3, a3, z3, p3_check)


def demo_2nd_order_iir():
    """Full analysis of a 2nd-order IIR system via Z-transform."""
    print("\n" + "=" * 65)
    print("3. 2nd-ORDER IIR SYSTEM ANALYSIS")
    print("=" * 65)

    # H(z) = (1 - 0.5 z^{-1}) / (1 - 0.6 z^{-1} + 0.5 z^{-2})
    # Written in positive powers: H(z) = z(z - 0.5) / (z^2 - 0.6z + 0.5)
    b = [1, -0.5, 0]             # numerator coefficients (z^2, z^1, z^0)
    a = [1, -0.6, 0.5]           # denominator coefficients

    zeros, poles, gain = signal.tf2zpk(b, a)
    print(f"\nH(z) = (z² - 0.5z) / (z² - 0.6z + 0.5)")
    print(f"  Zeros  : {np.round(zeros, 4)}")
    print(f"  Poles  : {np.round(poles, 4)}")
    print(f"  |Poles|: {np.round(np.abs(poles), 4)}  → stable: {np.all(np.abs(poles) < 1)}")

    # Inverse Z-transform via partial fractions
    h_pf, residues, pf_poles, direct = partial_fraction_inverse(b, a, n_samples=30)
    print(f"\nPartial Fraction Expansion:")
    for i, (ri, pi) in enumerate(zip(residues, pf_poles)):
        print(f"  r_{i} = {ri:.4f},  p_{i} = {pi:.4f}  → term: {ri:.4f} * ({pi:.4f})^n")

    # Compare with direct impulse response from lfilter
    impulse = np.zeros(30); impulse[0] = 1.0
    h_direct = signal.lfilter(b, a, impulse)
    err = np.max(np.abs(h_pf - h_direct))
    print(f"\nImpulse response comparison (PF vs lfilter): max error = {err:.2e}")

    # Frequency response
    w, H = signal.freqz(b, a, worN=512)
    print(f"\nFrequency response (first 5 points):")
    for i in range(5):
        print(f"  ω = {w[i]:.3f} rad/sample  |H| = {np.abs(H[i]):.4f}")

    return b, a, zeros, poles, w, H, h_direct


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_pole_zero(ax, zeros, poles, title):
    """Draw a pole-zero diagram on the given axes."""
    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=0.8, alpha=0.5, label='Unit circle')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    # Zeros (circles) and poles (crosses)
    if len(zeros) > 0:
        ax.scatter(np.real(zeros), np.imag(zeros), marker='o', s=80,
                   facecolors='none', edgecolors='blue', linewidths=2, zorder=5, label='Zeros')
    if len(poles) > 0:
        ax.scatter(np.real(poles), np.imag(poles), marker='x', s=80,
                   color='red', linewidths=2, zorder=5, label='Poles')

    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    lim = 1.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


if __name__ == "__main__":
    print("Z-TRANSFORM ANALYSIS")
    print("=" * 65)

    # --- Numerical demonstrations ---
    demo_common_z_transforms()
    sys1, sys2, sys3 = demo_pole_zero_and_stability()
    b, a, zeros, poles, w, H, h_direct = demo_2nd_order_iir()

    # --- Visualizations ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Z-Transform Analysis', fontsize=14, fontweight='bold')

    # Row 0: Pole-zero diagrams for the three systems
    plot_pole_zero(axes[0, 0], sys1[2], sys1[3], 'Stable System\n(poles inside unit circle)')
    plot_pole_zero(axes[0, 1], sys2[2], sys2[3], 'Unstable System\n(pole outside unit circle)')
    plot_pole_zero(axes[0, 2], sys3[2], sys3[3], 'Marginally Stable\n(poles on unit circle)')

    # Row 1, col 0: Pole-zero of 2nd-order IIR system
    plot_pole_zero(axes[1, 0], zeros, poles, '2nd-Order IIR\nPole-Zero Diagram')

    # Row 1, col 1: Impulse response (inverse Z-transform)
    n_plot = np.arange(30)
    axes[1, 1].stem(n_plot, h_direct, basefmt='k-', linefmt='C0-', markerfmt='C0o')
    axes[1, 1].set_xlabel('Sample n')
    axes[1, 1].set_ylabel('h[n]')
    axes[1, 1].set_title('Impulse Response h[n]\n(Inverse Z-Transform via PFE)')
    axes[1, 1].grid(True, alpha=0.3)

    # Row 1, col 2: Magnitude and phase frequency response
    axes[1, 2].plot(w / np.pi, 20 * np.log10(np.maximum(np.abs(H), 1e-10)),
                    'b-', linewidth=1.8)
    axes[1, 2].set_xlabel('Normalized Frequency (×π rad/sample)')
    axes[1, 2].set_ylabel('Magnitude (dB)')
    axes[1, 2].set_title('2nd-Order IIR Frequency Response\n(from Z-Transform)')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = '/opt/projects/01_Personal/03_Study/examples/Signal_Processing/07_z_transform.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved visualization: {out_path}")
    plt.close()

    print("\n" + "=" * 65)
    print("Key Takeaways:")
    print("  - ROC determines whether the Z-transform exists")
    print("  - Poles inside unit circle ↔ stable causal system")
    print("  - Partial fraction expansion gives the inverse Z-transform")
    print("  - Z-transform on the unit circle equals the DTFT")
    print("=" * 65)
