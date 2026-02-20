#!/usr/bin/env python3
"""
Signals and Systems - Basic Signal Types and Classification

Demonstrates fundamental signal concepts:
- Basic signal types: sinusoidal, exponential, unit step, unit impulse,
  rectangular pulse
- Signal classification: continuous vs discrete, periodic vs aperiodic,
  deterministic vs random
- Even/odd decomposition of a signal
- Signal energy and power
- Time-shifting, scaling, and reversal operations

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_basic_signals():
    """Generate and plot the five fundamental signal types."""
    print("=" * 60)
    print("BASIC SIGNAL TYPES")
    print("=" * 60)

    t = np.linspace(-2, 4, 1000)   # Continuous-time axis
    n = np.arange(-10, 30)          # Discrete-time axis

    # --- 1. Sinusoidal signal: x(t) = A * cos(2*pi*f*t + phi) ---
    A, f, phi = 1.5, 1.0, np.pi / 4
    x_sin = A * np.cos(2 * np.pi * f * t + phi)
    print(f"Sinusoidal: A={A}, f={f} Hz, phi={phi:.2f} rad")
    print(f"  Period T = 1/f = {1/f:.2f} s")

    # --- 2. Decaying exponential: x(t) = e^(-alpha * t) * u(t) ---
    alpha = 1.2
    x_exp = np.exp(-alpha * t) * (t >= 0)
    print(f"\nDecaying exponential: alpha={alpha}")
    print(f"  Time constant tau = 1/alpha = {1/alpha:.2f} s")

    # --- 3. Unit step function: u(t) = 1 if t >= 0, else 0 ---
    x_step = (t >= 0).astype(float)
    print("\nUnit step: u(t) = 1 for t >= 0, 0 otherwise")

    # --- 4. Unit impulse (approximated as narrow pulse): delta(t) ---
    # The true Dirac delta is the limit of a rectangular pulse of width epsilon
    # and height 1/epsilon as epsilon -> 0. Here we approximate it.
    dt = t[1] - t[0]
    x_impulse = np.zeros_like(t)
    idx = np.argmin(np.abs(t))     # index closest to t=0
    x_impulse[idx] = 1.0 / dt     # area = 1 (sifting property)
    print("\nUnit impulse (approximated): delta(t), area = 1")

    # --- 5. Rectangular pulse: rect(t) = 1 for |t| < T/2, else 0 ---
    T_rect = 1.0                   # pulse width
    x_rect = (np.abs(t) < T_rect / 2).astype(float)
    print(f"\nRectangular pulse: width={T_rect} s, centered at t=0")

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Basic Signal Types", fontsize=14, fontweight='bold')

    axes[0, 0].plot(t, x_sin, 'b')
    axes[0, 0].set_title(f"Sinusoidal: {A}·cos(2π·{f}·t + π/4)")
    axes[0, 0].set_xlabel("t (s)")
    axes[0, 0].axhline(0, color='k', linewidth=0.5)

    axes[0, 1].plot(t, x_exp, 'r')
    axes[0, 1].set_title(f"Decaying Exponential: e^(-{alpha}t)·u(t)")
    axes[0, 1].set_xlabel("t (s)")

    axes[1, 0].plot(t, x_step, 'g')
    axes[1, 0].set_title("Unit Step: u(t)")
    axes[1, 0].set_xlabel("t (s)")
    axes[1, 0].set_ylim(-0.2, 1.5)

    axes[1, 1].plot(t, x_impulse, 'm')
    axes[1, 1].set_title("Unit Impulse: δ(t) [approximated]")
    axes[1, 1].set_xlabel("t (s)")
    axes[1, 1].set_xlim(-0.5, 0.5)

    axes[2, 0].plot(t, x_rect, 'orange')
    axes[2, 0].set_title(f"Rectangular Pulse: width={T_rect} s")
    axes[2, 0].set_xlabel("t (s)")
    axes[2, 0].set_ylim(-0.2, 1.5)

    # Discrete-time sinusoid alongside
    x_disc = np.cos(2 * np.pi * 0.1 * n)
    axes[2, 1].stem(n, x_disc, basefmt='k-', linefmt='C0-', markerfmt='C0o')
    axes[2, 1].set_title("Discrete-Time Sinusoid: cos(0.2π·n)")
    axes[2, 1].set_xlabel("n (samples)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


def classify_signals():
    """Illustrate signal classification categories."""
    print("\n" + "=" * 60)
    print("SIGNAL CLASSIFICATION")
    print("=" * 60)

    t = np.linspace(0, 4, 1000)
    n = np.arange(0, 40)

    # Periodic vs Aperiodic
    # Periodic: x(t) = cos(2*pi*t), period T=1
    x_periodic = np.cos(2 * np.pi * t)
    # Aperiodic: decaying exponential has no repeating pattern
    x_aperiodic = np.exp(-t)

    # Deterministic vs Random
    x_deterministic = np.sin(2 * np.pi * t)
    rng = np.random.default_rng(42)
    x_random = rng.standard_normal(len(t))   # White Gaussian noise

    # Check periodicity numerically
    T_candidate = 1.0          # Expected period
    N_check = int(T_candidate / (t[1] - t[0]))
    error_periodic = np.mean(np.abs(x_periodic[N_check:] - x_periodic[:-N_check]))
    print(f"Periodic signal x(t)=cos(2πt):  period-shift error = {error_periodic:.6f}")
    print(f"  -> {'Periodic' if error_periodic < 1e-6 else 'Aperiodic'}")

    print("\nSignal classification summary:")
    print("  Continuous-time: defined for all real t  (e.g. cos(2πt))")
    print("  Discrete-time:   defined only at integer n  (e.g. cos(0.2πn))")
    print("  Periodic:        x(t+T) = x(t) for all t")
    print("  Aperiodic:       no such T exists")
    print("  Deterministic:   future values are exactly predictable")
    print("  Random:          future values described by probability")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Signal Classification", fontsize=14, fontweight='bold')

    axes[0, 0].plot(t, x_periodic, 'b')
    axes[0, 0].set_title("Periodic: cos(2πt), T=1 s")
    axes[0, 0].axvline(1, color='r', linestyle='--', alpha=0.5, label='T=1')
    axes[0, 0].axvline(2, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].legend()

    axes[0, 1].plot(t, x_aperiodic, 'r')
    axes[0, 1].set_title("Aperiodic: e^(-t)")

    axes[1, 0].plot(t, x_deterministic, 'g')
    axes[1, 0].set_title("Deterministic: sin(2πt)")

    axes[1, 1].plot(t, x_random, 'm', linewidth=0.5)
    axes[1, 1].set_title("Random: White Gaussian Noise")

    for ax in axes.flat:
        ax.set_xlabel("t (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def even_odd_decomposition():
    """
    Decompose a signal into its even and odd components.

    Any signal x(t) can be written as:
        x(t) = x_e(t) + x_o(t)
    where:
        x_e(t) = [x(t) + x(-t)] / 2    (even part)
        x_o(t) = [x(t) - x(-t)] / 2    (odd part)
    """
    print("\n" + "=" * 60)
    print("EVEN/ODD DECOMPOSITION")
    print("=" * 60)

    t = np.linspace(-3, 3, 1000)
    # Example signal: x(t) = e^t * u(t)  (one-sided exponential)
    x = np.exp(t) * (t >= 0)

    x_even = (x + x[::-1]) / 2   # x[::-1] is x(-t) for symmetric t axis
    x_odd  = (x - x[::-1]) / 2

    # Verify decomposition
    error = np.max(np.abs(x - (x_even + x_odd)))
    print(f"x(t) = e^t * u(t)")
    print(f"Reconstruction error ||x - (x_e + x_o)||_inf = {error:.2e}")
    print(f"Even part is symmetric:  x_e(t) = x_e(-t)")
    print(f"Odd part is antisymmetric: x_o(t) = -x_o(-t)")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Even/Odd Decomposition of x(t) = e^t·u(t)", fontsize=13)

    axes[0].plot(t, x, 'b', linewidth=2)
    axes[0].set_title("Original: x(t)")

    axes[1].plot(t, x_even, 'r', linewidth=2)
    axes[1].set_title("Even Part: x_e(t) = [x(t)+x(-t)]/2")
    axes[1].axhline(0, color='k', linewidth=0.5)

    axes[2].plot(t, x_odd, 'g', linewidth=2)
    axes[2].set_title("Odd Part: x_o(t) = [x(t)-x(-t)]/2")
    axes[2].axhline(0, color='k', linewidth=0.5)

    for ax in axes:
        ax.set_xlabel("t")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='k', linewidth=0.5)

    plt.tight_layout()
    plt.show()


def energy_and_power():
    """
    Compute signal energy and power.

    Energy:  E = integral |x(t)|^2 dt  (finite for energy signals)
    Power:   P = lim(T->inf) (1/T) * integral_{-T/2}^{T/2} |x(t)|^2 dt
               (finite for power signals)

    - Energy signal: E < inf, P = 0  (e.g., decaying exponential)
    - Power signal:  P < inf, E = inf (e.g., sinusoidal)
    """
    print("\n" + "=" * 60)
    print("SIGNAL ENERGY AND POWER")
    print("=" * 60)

    dt = 0.001
    t_long = np.arange(-50, 50, dt)

    # Energy signal: x(t) = e^(-t) * u(t)
    alpha = 1.0
    x_energy = np.exp(-alpha * np.abs(t_long)) * (t_long >= 0)
    E_numerical = np.sum(x_energy ** 2) * dt
    E_analytical = 1 / (2 * alpha)   # integral of e^(-2*alpha*t), t>=0
    print(f"Energy signal: x(t) = e^(-{alpha}t)·u(t)")
    print(f"  Numerical energy  E = {E_numerical:.4f}")
    print(f"  Analytical energy E = 1/(2α) = {E_analytical:.4f}")

    # Power signal: x(t) = A * cos(2*pi*f*t)
    A_pow, f_pow = 2.0, 1.0
    x_power = A_pow * np.cos(2 * np.pi * f_pow * t_long)
    T_obs = t_long[-1] - t_long[0]
    P_numerical = np.sum(x_power ** 2) * dt / T_obs
    P_analytical = A_pow ** 2 / 2     # average power of sinusoid
    print(f"\nPower signal:  x(t) = {A_pow}·cos(2π·{f_pow}·t)")
    print(f"  Numerical power   P = {P_numerical:.4f}")
    print(f"  Analytical power  P = A²/2 = {P_analytical:.4f}")
    print(f"  Energy of sinusoid is infinite (power signal)")


def time_operations():
    """
    Demonstrate time-domain signal transformations:
      - Time shifting:  x(t - t0)
      - Time scaling:   x(a*t)    -- compression (a>1) or expansion (a<1)
      - Time reversal:  x(-t)
    """
    print("\n" + "=" * 60)
    print("TIME-DOMAIN OPERATIONS")
    print("=" * 60)

    t = np.linspace(-3, 5, 1000)
    # Base signal: rectangular pulse centered at t=0, width=1
    x = (np.abs(t) < 0.5).astype(float)

    t0 = 1.5        # shift amount
    a_compress = 2  # compression factor
    a_expand = 0.5  # expansion factor

    x_shifted  = (np.abs(t - t0) < 0.5).astype(float)    # delay by t0
    x_compress = (np.abs(a_compress * t) < 0.5).astype(float)  # x(2t): half width
    x_expand   = (np.abs(a_expand * t) < 0.5).astype(float)    # x(t/2): double width
    x_reversed = (np.abs(-t) < 0.5).astype(float)              # x(-t)

    print(f"Base: rect(t), width=1")
    print(f"Time shift by {t0}:       x(t-{t0}) -> pulse centered at t={t0}")
    print(f"Time compression (a={a_compress}): x({a_compress}t) -> pulse width = 1/{a_compress} = 0.5")
    print(f"Time expansion   (a={a_expand}): x({a_expand}t) -> pulse width = 1/{a_expand} = 2.0")
    print(f"Time reversal:            x(-t) -> pulse reflected about t=0")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Time-Domain Signal Transformations", fontsize=14, fontweight='bold')

    plots = [
        (x,          "Original: x(t)"),
        (x_shifted,  f"Time Shift: x(t-{t0})"),
        (x_compress, f"Compression: x({a_compress}t)"),
        (x_expand,   f"Expansion: x(t/{1//a_expand:.0f})"),
        (x_reversed, "Reversal: x(-t)"),
    ]

    for idx, (sig, title) in enumerate(plots):
        ax = axes[idx // 3][idx % 3]
        ax.plot(t, sig, 'b', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-0.2, 1.5)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    axes[1][2].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generate_basic_signals()
    classify_signals()
    even_odd_decomposition()
    energy_and_power()
    time_operations()

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
