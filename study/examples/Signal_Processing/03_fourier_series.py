#!/usr/bin/env python3
"""
Fourier Series

Demonstrates Fourier series analysis and synthesis:
- Fourier coefficients for square wave, sawtooth, and triangular wave
- Signal reconstruction using partial sums (Gibbs phenomenon)
- Magnitude and phase spectra of the coefficients
- Parseval's theorem: energy in time domain equals energy in frequency domain
- Convergence behaviour with increasing number of harmonics

The Fourier series of a periodic signal x(t) with period T is:
    x(t) = sum_{n=-inf}^{inf} c_n * exp(j * 2*pi*n*t / T)

The complex Fourier coefficients are:
    c_n = (1/T) * integral_{-T/2}^{T/2} x(t) * exp(-j * 2*pi*n*t / T) dt

Dependencies: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Fourier coefficient computation
# ---------------------------------------------------------------------------

def fourier_coefficients_numerical(x_func, T: float, N_terms: int) -> np.ndarray:
    """
    Numerically compute complex Fourier coefficients c_n for n = -N..N.

    c_n = (1/T) * integral_{0}^{T} x(t) * exp(-j * 2*pi*n*t / T) dt

    Parameters
    ----------
    x_func  : callable, periodic signal x(t)
    T       : period
    N_terms : number of harmonics on each side (output has 2*N_terms+1 coeffs)

    Returns
    -------
    c : complex array, shape (2*N_terms+1,), indices correspond to n = -N..N
    """
    c = np.zeros(2 * N_terms + 1, dtype=complex)
    for idx, n in enumerate(range(-N_terms, N_terms + 1)):
        def integrand_real(t, n=n):
            return x_func(t) * np.cos(2 * np.pi * n * t / T)

        def integrand_imag(t, n=n):
            return -x_func(t) * np.sin(2 * np.pi * n * t / T)

        real_part, _ = quad(integrand_real, 0, T)
        imag_part, _ = quad(integrand_imag, 0, T)
        c[idx] = (real_part + 1j * imag_part) / T
    return c


def reconstruct_signal(c: np.ndarray, T: float, t: np.ndarray) -> np.ndarray:
    """
    Reconstruct periodic signal from complex Fourier coefficients.

    x(t) = sum_{n=-N}^{N} c_n * exp(j * 2*pi*n*t / T)

    Parameters
    ----------
    c : complex Fourier coefficients, length 2*N+1
    T : period
    t : time array

    Returns
    -------
    x_reconstructed : real-valued signal (imaginary part should be ~0)
    """
    N = (len(c) - 1) // 2
    x_rec = np.zeros(len(t), dtype=complex)
    for idx, n in enumerate(range(-N, N + 1)):
        x_rec += c[idx] * np.exp(1j * 2 * np.pi * n * t / T)
    return x_rec.real


# ---------------------------------------------------------------------------
# Signal definitions
# ---------------------------------------------------------------------------

def square_wave(t: np.ndarray, T: float = 1.0, duty: float = 0.5) -> np.ndarray:
    """Square wave: +1 for first half-period, -1 for second half-period."""
    phase = (t % T) / T          # normalized phase in [0, 1)
    return np.where(phase < duty, 1.0, -1.0)


def sawtooth_wave(t: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Sawtooth wave: linear ramp from -1 to +1 over one period."""
    return 2 * ((t / T) % 1) - 1


def triangular_wave(t: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Triangular wave: linearly rises then falls symmetrically."""
    phase = (t / T) % 1
    return np.where(phase < 0.5, 4 * phase - 1, -4 * phase + 3)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_gibbs_phenomenon():
    """
    Show Gibbs phenomenon: partial Fourier sums of a square wave overshoot
    near discontinuities by ~9% regardless of the number of terms.

    Analytical Fourier series of the square wave (odd harmonics only):
        x(t) = (4/pi) * sum_{k=0}^{inf} sin((2k+1)*2*pi*t/T) / (2k+1)
    """
    print("=" * 60)
    print("GIBBS PHENOMENON (SQUARE WAVE RECONSTRUCTION)")
    print("=" * 60)

    T = 1.0
    t = np.linspace(0, 2 * T, 2000)
    x_true = square_wave(t, T)

    N_list = [1, 3, 7, 19, 49]

    fig, axes = plt.subplots(len(N_list), 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Gibbs Phenomenon: Square Wave Partial Sums", fontsize=13, fontweight='bold')

    for ax, N in zip(axes, N_list):
        # Reconstruct using only odd harmonics up to 2N-1
        x_rec = np.zeros_like(t)
        for k in range(N):
            n = 2 * k + 1       # odd harmonic index
            x_rec += (4 / np.pi) * np.sin(n * 2 * np.pi * t / T) / n

        overshoot = (x_rec.max() - 1.0) * 100
        ax.plot(t, x_true, 'lightgray', linewidth=2, label='True')
        ax.plot(t, x_rec,  'b',         linewidth=1, label=f'N={N} terms')
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-1.5, 1.5)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"N = {N} harmonics, overshoot ≈ {overshoot:.1f}%", fontsize=9)
        print(f"  N={N:2d}: overshoot = {overshoot:.2f}%")

    axes[-1].set_xlabel("t (s)")
    plt.tight_layout()
    plt.show()

    print("\nGibbs phenomenon: overshoot converges to ~9% (π/4 - 1 ≈ 8.9%)")
    print("Overshoot does NOT decrease with more terms (only gets narrower)")


def demo_spectra():
    """
    Plot magnitude and phase spectra for square, sawtooth, and triangular waves.

    The frequency spectrum shows which harmonics are present and their amplitudes.
    """
    print("\n" + "=" * 60)
    print("MAGNITUDE AND PHASE SPECTRA")
    print("=" * 60)

    T = 1.0
    N_terms = 15    # compute up to 15th harmonic
    f0 = 1.0 / T   # fundamental frequency

    signals = {
        'Square Wave':     lambda t: np.where((t % T) / T < 0.5, 1.0, -1.0),
        'Sawtooth Wave':   lambda t: 2 * ((t / T) % 1) - 1,
        'Triangular Wave': lambda t: triangular_wave(np.atleast_1d(t), T).item()
                           if np.isscalar(t) else triangular_wave(t, T),
    }

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Fourier Spectra of Periodic Signals", fontsize=13, fontweight='bold')

    for row, (name, func) in enumerate(signals.items()):
        c = fourier_coefficients_numerical(func, T, N_terms)
        n_vals = np.arange(-N_terms, N_terms + 1)
        freqs  = n_vals * f0

        magnitude = np.abs(c)
        phase     = np.angle(c)
        # Zero out negligible phases (below numerical noise floor)
        phase[magnitude < 1e-10] = 0.0

        # One-sided for clarity: show n >= 0 only
        pos_mask = n_vals >= 0
        n_pos    = n_vals[pos_mask]
        mag_pos  = 2 * magnitude[pos_mask]   # double-sided to one-sided
        mag_pos[n_pos == 0] /= 2             # DC term is not doubled
        pha_pos  = phase[pos_mask]

        axes[row, 0].stem(n_pos, mag_pos, basefmt='k-',
                          linefmt='C0-', markerfmt='C0o')
        axes[row, 0].set_title(f"{name}: Magnitude Spectrum")
        axes[row, 0].set_xlabel("Harmonic number n")
        axes[row, 0].set_ylabel("|c_n|")
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].stem(n_pos, pha_pos, basefmt='k-',
                          linefmt='C1-', markerfmt='C1o')
        axes[row, 1].set_title(f"{name}: Phase Spectrum")
        axes[row, 1].set_xlabel("Harmonic number n")
        axes[row, 1].set_ylabel("Phase (rad)")
        axes[row, 1].set_ylim(-np.pi - 0.3, np.pi + 0.3)
        axes[row, 1].axhline(0,         color='k', linewidth=0.5)
        axes[row, 1].axhline( np.pi/2, color='gray', linestyle='--', alpha=0.4)
        axes[row, 1].axhline(-np.pi/2, color='gray', linestyle='--', alpha=0.4)
        axes[row, 1].grid(True, alpha=0.3)

        # Print dominant harmonics
        print(f"{name}:")
        for idx_n, (n, m) in enumerate(zip(n_pos[:8], mag_pos[:8])):
            if m > 0.01:
                print(f"  n={n:2d}: |c_n|={m:.4f}, phase={pha_pos[idx_n]:.3f} rad")

    plt.tight_layout()
    plt.show()


def demo_parseval():
    """
    Verify Parseval's theorem numerically.

    Parseval's theorem states that the total signal power equals the sum
    of squared magnitudes of Fourier coefficients:

        (1/T) * integral_T |x(t)|^2 dt = sum_{n=-inf}^{inf} |c_n|^2

    This is a statement of energy conservation between time and frequency domains.
    """
    print("\n" + "=" * 60)
    print("PARSEVAL'S THEOREM VERIFICATION")
    print("=" * 60)

    T = 1.0
    dt = 0.0001
    t = np.arange(0, T, dt)
    N_terms = 50    # use many terms for good approximation

    signals = {
        'Square Wave':     square_wave(t, T),
        'Sawtooth Wave':   sawtooth_wave(t, T),
        'Triangular Wave': triangular_wave(t, T),
    }
    signal_funcs = {
        'Square Wave':     lambda t: np.where((t % T) / T < 0.5, 1.0, -1.0),
        'Sawtooth Wave':   lambda t: 2 * ((t / T) % 1) - 1,
        'Triangular Wave': lambda t: triangular_wave(np.atleast_1d(t), T).item()
                           if np.isscalar(t) else triangular_wave(t, T),
    }

    for name in signals:
        x = signals[name]
        func = signal_funcs[name]

        # Time-domain power: (1/T) * integral |x(t)|^2 dt
        power_time = np.mean(x ** 2)

        # Frequency-domain power: sum |c_n|^2
        c = fourier_coefficients_numerical(func, T, N_terms)
        power_freq = np.sum(np.abs(c) ** 2)

        print(f"{name}:")
        print(f"  Time-domain power:      {power_time:.6f}")
        print(f"  Frequency-domain power: {power_freq:.6f}")
        print(f"  Relative error:         {abs(power_time - power_freq) / power_time * 100:.3f}%")


def demo_convergence():
    """
    Show convergence rate of Fourier series for different waveforms.

    Convergence rate depends on signal smoothness:
    - Discontinuities (square, sawtooth): coefficients decay as 1/n  -> slow
    - Continuous but non-smooth (triangular): coefficients decay as 1/n^2 -> faster
    - Smooth signals: coefficients decay exponentially -> fastest
    """
    print("\n" + "=" * 60)
    print("FOURIER SERIES CONVERGENCE COMPARISON")
    print("=" * 60)

    T = 1.0
    dt = 0.001
    t = np.linspace(0, T, int(T / dt), endpoint=False)

    x_square = square_wave(t, T)
    x_saw    = sawtooth_wave(t, T)
    x_tri    = triangular_wave(t, T)

    signal_funcs = {
        'Square (1/n decay)':    lambda t: np.where((t % T) / T < 0.5, 1.0, -1.0),
        'Sawtooth (1/n decay)':  lambda t: 2 * ((t / T) % 1) - 1,
        'Triangular (1/n² decay)': lambda t: triangular_wave(np.atleast_1d(t), T).item()
                                   if np.isscalar(t) else triangular_wave(t, T),
    }

    N_max = 30
    N_list = np.arange(1, N_max + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fourier Series Convergence", fontsize=13, fontweight='bold')

    true_signals = [x_square, x_saw, x_tri]

    for (name, func), x_true in zip(signal_funcs.items(), true_signals):
        errors = []
        c_all = fourier_coefficients_numerical(func, T, N_max)

        for N in N_list:
            # Use coefficients up to harmonic N
            c_trunc = np.zeros_like(c_all)
            center = N_max
            c_trunc[center - N: center + N + 1] = c_all[center - N: center + N + 1]
            x_rec = reconstruct_signal(c_trunc, T, t)
            rmse = np.sqrt(np.mean((x_rec - x_true) ** 2))
            errors.append(rmse)

        axes[0].plot(N_list, errors, marker='o', markersize=3, label=name)
        axes[1].semilogy(N_list, errors, marker='o', markersize=3, label=name)

        print(f"{name}: RMSE at N=1: {errors[0]:.4f}, N=10: {errors[9]:.4f}, N=30: {errors[-1]:.4f}")

    axes[0].set_title("Convergence (linear scale)")
    axes[0].set_xlabel("Number of harmonics N")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Convergence (log scale)")
    axes[1].set_xlabel("Number of harmonics N")
    axes[1].set_ylabel("RMSE (log)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nSmoother signals require fewer harmonics for the same accuracy.")
    print("Triangular wave converges faster (1/n²) than square/sawtooth (1/n).")


if __name__ == "__main__":
    demo_gibbs_phenomenon()
    demo_spectra()
    demo_parseval()
    demo_convergence()

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
