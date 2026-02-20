#!/usr/bin/env python3
"""
Continuous Fourier Transform (CTFT) - Numerical Approximation
=============================================================

The Continuous-Time Fourier Transform (CTFT) of a signal x(t) is defined as:

    X(f) = integral_{-inf}^{inf} x(t) * exp(-j*2*pi*f*t) dt

We numerically approximate this integral using the trapezoidal rule over a
finite time window, sampled at a high rate.

Topics Covered:
    1. CTFT of common signals: rectangular pulse, Gaussian, exponential decay
    2. Fourier transform properties: linearity, time shift, frequency shift
    3. Parseval's theorem: energy conservation in time and frequency domains
    4. Duality: rect(t) <--> sinc(f) pair
    5. Bandwidth and energy spectral density

Author: Educational example for Signal Processing
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gausspulse


# =============================================================================
# 1. Numerical CTFT computation
# =============================================================================

def ctft(x, t):
    """
    Numerically approximate the Continuous-Time Fourier Transform.

    Uses the trapezoidal rule:  X(f) ≈ sum( x(t) * exp(-j2πft) ) * dt

    Args:
        x (ndarray): Signal samples
        t (ndarray): Uniformly spaced time array

    Returns:
        tuple: (f, X) frequency axis and complex spectrum
    """
    dt = t[1] - t[0]
    N = len(t)
    # Frequency axis: centered at 0, resolution = 1/(N*dt)
    f = np.fft.fftfreq(N, d=dt)
    # FFT approximates the CTFT integral when multiplied by dt
    X = np.fft.fft(x) * dt
    # Shift zero-frequency component to center for display
    f = np.fft.fftshift(f)
    X = np.fft.fftshift(X)
    return f, X


# =============================================================================
# 2. Common signal definitions
# =============================================================================

def rectangular_pulse(t, width=1.0, center=0.0):
    """
    Rectangular pulse of given width centered at 'center'.

    rect(t/T) = 1  if |t - center| <= T/2
                0  otherwise

    CTFT: T * sinc(f*T)  where sinc(x) = sin(pi*x)/(pi*x)
    """
    return np.where(np.abs(t - center) <= width / 2, 1.0, 0.0)


def gaussian_signal(t, sigma=0.5, center=0.0):
    """
    Gaussian pulse: x(t) = exp(-t^2 / (2*sigma^2))

    The Fourier transform of a Gaussian is also a Gaussian:
    CTFT: sigma*sqrt(2*pi) * exp(-2*pi^2*sigma^2*f^2)

    Key property: time-bandwidth product is constant (sigma * sigma_f = 1/(2*pi))
    """
    return np.exp(-((t - center) ** 2) / (2 * sigma ** 2))


def exponential_decay(t, alpha=2.0):
    """
    One-sided exponential decay: x(t) = exp(-alpha*t) * u(t)

    CTFT: 1 / (alpha + j*2*pi*f)
    Magnitude: 1 / sqrt(alpha^2 + (2*pi*f)^2)
    """
    return np.where(t >= 0, np.exp(-alpha * t), 0.0)


# =============================================================================
# 3. Main demonstration functions
# =============================================================================

def demo_common_signals():
    """Compute and plot CTFT for three common signals."""
    print("=" * 60)
    print("Demo 1: CTFT of Common Signals")
    print("=" * 60)

    # Time axis: long window, high sampling rate for good approximation
    dt = 0.005
    t = np.arange(-10, 10, dt)

    # --- Rectangular pulse (width = 1 s) ---
    T_rect = 1.0
    x_rect = rectangular_pulse(t, width=T_rect)
    f_rect, X_rect = ctft(x_rect, t)
    # Analytical: T * sinc(f*T)
    with np.errstate(divide='ignore', invalid='ignore'):
        X_rect_analytical = T_rect * np.sinc(f_rect * T_rect)

    # --- Gaussian (sigma = 0.4 s) ---
    sigma = 0.4
    x_gauss = gaussian_signal(t, sigma=sigma)
    f_gauss, X_gauss = ctft(x_gauss, t)
    # Analytical: sigma*sqrt(2*pi)*exp(-2*pi^2*sigma^2*f^2)
    X_gauss_analytical = sigma * np.sqrt(2 * np.pi) * np.exp(
        -2 * np.pi ** 2 * sigma ** 2 * f_gauss ** 2
    )

    # --- Exponential decay (alpha = 3) ---
    alpha = 3.0
    x_exp = exponential_decay(t, alpha=alpha)
    f_exp, X_exp = ctft(x_exp, t)
    # Analytical magnitude: 1/sqrt(alpha^2 + (2*pi*f)^2)
    X_exp_analytical = 1.0 / np.sqrt(alpha ** 2 + (2 * np.pi * f_exp) ** 2)

    # Report bandwidths (3 dB point)
    fmax = 5.0  # display limit
    mask = np.abs(f_rect) <= fmax

    print(f"  Rect pulse (T={T_rect}s): first null at f = {1/T_rect:.2f} Hz")
    print(f"  Gaussian (sigma={sigma}s): spectral sigma = {1/(2*np.pi*sigma):.3f} Hz")
    print(f"  Exp decay (alpha={alpha}): half-power bandwidth = {alpha/(2*np.pi):.3f} Hz")

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("CTFT of Common Signals", fontsize=14, fontweight='bold')

    signals = [
        ("Rectangular Pulse", t, x_rect, f_rect, X_rect, X_rect_analytical, "b"),
        ("Gaussian Pulse", t, x_gauss, f_gauss, X_gauss, X_gauss_analytical, "g"),
        ("Exponential Decay", t, x_exp, f_exp, X_exp, X_exp_analytical, "r"),
    ]

    for i, (name, ti, xi, fi, Xi, Xi_anal, color) in enumerate(signals):
        mask = np.abs(fi) <= fmax
        # Time domain
        axes[i, 0].plot(ti, xi, color=color)
        axes[i, 0].set_xlim(-3, 5)
        axes[i, 0].set_title(f"{name}: Time Domain")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True, alpha=0.3)
        # Frequency domain: magnitude
        axes[i, 1].plot(fi[mask], np.abs(Xi[mask]), color=color, label='Numerical', lw=2)
        axes[i, 1].plot(fi[mask], np.abs(Xi_anal[mask]), 'k--', label='Analytical', lw=1.5)
        axes[i, 1].set_title(f"{name}: |X(f)|")
        axes[i, 1].set_xlabel("Frequency (Hz)")
        axes[i, 1].set_ylabel("|X(f)|")
        axes[i, 1].legend(fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/04_ctft_signals.png', dpi=150)
    print("  Saved plot to /tmp/04_ctft_signals.png")
    plt.show()


def demo_fourier_properties():
    """
    Illustrate key Fourier transform properties:
      - Linearity:    a*x1(t) + b*x2(t)  <-->  a*X1(f) + b*X2(f)
      - Time shift:   x(t - t0)           <-->  X(f)*exp(-j2*pi*f*t0)
      - Freq shift:   x(t)*exp(j2*pi*f0*t)<-->  X(f - f0)   (modulation)
    """
    print("\n" + "=" * 60)
    print("Demo 2: Fourier Transform Properties")
    print("=" * 60)

    dt = 0.005
    t = np.arange(-10, 10, dt)
    fmax = 6.0

    # Base signal: Gaussian
    sigma = 0.3
    x1 = gaussian_signal(t, sigma=sigma, center=0.0)
    x2 = rectangular_pulse(t, width=0.8, center=0.0)

    f, X1 = ctft(x1, t)
    _, X2 = ctft(x2, t)

    # --- Linearity ---
    a, b = 0.7, 0.5
    x_linear = a * x1 + b * x2
    _, X_linear = ctft(x_linear, t)
    X_linear_theory = a * X1 + b * X2

    # --- Time shift: shift x1 by t0 seconds ---
    t0 = 1.5
    x_shifted = gaussian_signal(t, sigma=sigma, center=t0)
    _, X_shifted = ctft(x_shifted, t)
    # Theory: X1(f) * exp(-j2*pi*f*t0)
    X_shifted_theory = X1 * np.exp(-1j * 2 * np.pi * f * t0)

    # --- Frequency shift (modulation): multiply x1 by cos(2*pi*f0*t) ---
    f0 = 3.0
    x_modulated = x1 * np.cos(2 * np.pi * f0 * t)
    _, X_modulated = ctft(x_modulated, t)
    # Theory: 0.5*(X1(f-f0) + X1(f+f0)), approximated by shifting X1
    print(f"  Linearity error:  {np.max(np.abs(X_linear - X_linear_theory)):.4e}")
    print(f"  Time-shift error: {np.max(np.abs(X_shifted - X_shifted_theory)):.4e}")

    mask = np.abs(f) <= fmax
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle("Fourier Transform Properties", fontsize=14, fontweight='bold')

    # Linearity
    axes[0, 0].plot(t, x_linear)
    axes[0, 0].set_xlim(-3, 3)
    axes[0, 0].set_title(f"Linearity: {a}·x₁(t) + {b}·x₂(t)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(f[mask], np.abs(X_linear[mask]), label='Numerical', lw=2)
    axes[0, 1].plot(f[mask], np.abs(X_linear_theory[mask]), 'r--', label='a·X₁+b·X₂', lw=1.5)
    axes[0, 1].set_title("Spectrum: Linearity verified")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Time shift
    axes[1, 0].plot(t, x1, label='x₁(t)', alpha=0.7)
    axes[1, 0].plot(t, x_shifted, label=f'x₁(t-{t0})', alpha=0.7)
    axes[1, 0].set_xlim(-3, 5)
    axes[1, 0].set_title(f"Time Shift: t₀ = {t0} s")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    # Phase of X_shifted should be linear: -2*pi*f*t0
    phase_numerical = np.angle(X_shifted[mask])
    phase_theory = np.angle(X_shifted_theory[mask])
    axes[1, 1].plot(f[mask], phase_numerical, label='Numerical phase')
    axes[1, 1].plot(f[mask], phase_theory, 'r--', label='-2πf·t₀')
    axes[1, 1].set_title("Phase Shift from Time Delay")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Phase (rad)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # Modulation
    axes[2, 0].plot(t, x_modulated)
    axes[2, 0].set_xlim(-3, 3)
    axes[2, 0].set_title(f"Modulation: x₁(t)·cos(2π·{f0}·t)")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 1].plot(f[mask], np.abs(X_modulated[mask]), label='Modulated', lw=2)
    axes[2, 1].plot(f[mask], np.abs(X1[mask]) * 0.5, 'r--', label='0.5·|X₁(f)|', lw=1.5)
    axes[2, 1].axvline(f0, color='gray', linestyle=':', alpha=0.6, label=f'f₀={f0} Hz')
    axes[2, 1].axvline(-f0, color='gray', linestyle=':', alpha=0.6)
    axes[2, 1].set_title("Spectrum shifts to ±f₀ (Freq. Shift)")
    axes[2, 1].set_xlabel("Frequency (Hz)")
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/04_fourier_properties.png', dpi=150)
    print("  Saved plot to /tmp/04_fourier_properties.png")
    plt.show()


def demo_parseval_and_duality():
    """
    Demonstrate Parseval's theorem and the rect <--> sinc duality.

    Parseval's theorem:
        integral |x(t)|^2 dt  =  integral |X(f)|^2 df
    (energy is conserved between time and frequency domains)

    Duality:
        If x(t) <--> X(f), then X(t) <--> x(-f)
        For real symmetric signals: rect(t/T) <--> T*sinc(f*T)
        and by duality:             sinc(t*W) <--> (1/W)*rect(f/W)
    """
    print("\n" + "=" * 60)
    print("Demo 3: Parseval's Theorem and Rect-Sinc Duality")
    print("=" * 60)

    dt = 0.002
    t = np.arange(-15, 15, dt)

    # --- Parseval's Theorem ---
    T_rect = 1.0
    x = rectangular_pulse(t, width=T_rect)
    f, X = ctft(x, t)
    df = f[1] - f[0]

    energy_time = np.trapz(np.abs(x) ** 2, t)
    energy_freq = np.trapz(np.abs(X) ** 2, f)

    print(f"  Parseval's Theorem (rect pulse, T={T_rect}):")
    print(f"    Time-domain energy:  {energy_time:.6f}")
    print(f"    Freq-domain energy:  {energy_freq:.6f}")
    print(f"    Relative error:      {abs(energy_time - energy_freq)/energy_time:.2e}")

    # --- Duality: sinc in time → rect in frequency ---
    # x(t) = sinc(W*t) has Fourier transform (1/W)*rect(f/W)
    W = 2.0  # bandwidth in Hz
    x_sinc = np.sinc(W * t)   # np.sinc(x) = sin(pi*x)/(pi*x)
    f_sinc, X_sinc = ctft(x_sinc, t)

    fmax = 6.0
    mask = np.abs(f_sinc) <= fmax

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Parseval's Theorem and Rect-Sinc Duality", fontsize=14, fontweight='bold')

    # Energy spectral density |X(f)|^2
    axes[0, 0].plot(t, x, 'b')
    axes[0, 0].set_xlim(-3, 3)
    axes[0, 0].set_title(f"rect(t/{T_rect}): Energy = {energy_time:.3f} J")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(f[mask], np.abs(X[mask]) ** 2, 'b')
    axes[0, 1].fill_between(f[mask], np.abs(X[mask]) ** 2, alpha=0.3)
    axes[0, 1].set_title(f"Energy Spectral Density |X(f)|² (∫ = {energy_freq:.3f} J)")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("|X(f)|²")
    axes[0, 1].grid(True, alpha=0.3)

    # Duality: sinc(Wt) <--> (1/W)*rect(f/W)
    axes[1, 0].plot(t, x_sinc, 'g')
    axes[1, 0].set_xlim(-4, 4)
    axes[1, 0].set_title(f"sinc({W}t) in time domain")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(f_sinc[mask], np.abs(X_sinc[mask]), 'g', label='Numerical', lw=2)
    # Analytical: (1/W)*rect(f/W)
    X_sinc_analytical = (1.0 / W) * rectangular_pulse(f_sinc, width=W)
    axes[1, 1].plot(f_sinc[mask], X_sinc_analytical[mask], 'r--', label=f'(1/{W})·rect(f/{W})', lw=1.5)
    axes[1, 1].set_title("Duality: sinc(Wt) ↔ (1/W)·rect(f/W)")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("|X(f)|")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/04_parseval_duality.png', dpi=150)
    print("  Saved plot to /tmp/04_parseval_duality.png")
    plt.show()


if __name__ == "__main__":
    print("Continuous Fourier Transform (CTFT) - Numerical Approximation")
    print("=" * 60)
    print("The CTFT integral is approximated numerically using FFT * dt.")
    print("A fine time grid (high sampling rate) over a long window")
    print("gives good agreement with analytical formulas.\n")

    demo_common_signals()
    demo_fourier_properties()
    demo_parseval_and_duality()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - CTFT ≈ FFT * dt  (for uniformly sampled, windowed signals)")
    print("  - Linearity: superposition holds in both domains")
    print("  - Time shift introduces linear phase: exp(-j2πft₀)")
    print("  - Modulation shifts spectrum to carrier frequency ±f₀")
    print("  - Parseval: total energy is conserved across domains")
    print("  - Duality: rect ↔ sinc (fundamental transform pair)")
    print("=" * 60)
