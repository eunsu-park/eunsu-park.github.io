#!/usr/bin/env python3
"""
Discrete Fourier Transform (DFT) and Fast Fourier Transform (FFT)
=================================================================

The Discrete Fourier Transform of an N-point sequence x[n] is:

    X[k] = sum_{n=0}^{N-1}  x[n] * exp(-j*2*pi*k*n/N),   k = 0, 1, ..., N-1

The FFT is an efficient algorithm to compute the DFT in O(N log N) operations
instead of O(N^2) for the naive matrix-multiplication form.

Topics Covered:
    1. DFT matrix multiplication vs. np.fft.fft (correctness check)
    2. Zero-padding: increase apparent frequency resolution
    3. Windowing: reduce spectral leakage (rect / Hanning / Hamming / Blackman)
    4. Multi-tone signal analysis with FFT
    5. Spectral leakage and the effect of different windows
    6. FFT vs. DFT computation-time comparison

Author: Educational example for Signal Processing
License: MIT
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
# 1. DFT via matrix multiplication (O(N^2) — for illustration only)
# =============================================================================

def dft_matrix(N):
    """
    Build the N×N DFT matrix W where W[k,n] = exp(-j*2*pi*k*n/N).

    X = W @ x  gives the DFT of x.

    Args:
        N (int): DFT size

    Returns:
        ndarray: Complex N×N DFT matrix
    """
    n = np.arange(N)
    k = n.reshape((N, 1))    # column vector of output indices
    # W[k,n] = exp(-j*2*pi*k*n/N)
    W = np.exp(-1j * 2 * np.pi * k * n / N)
    return W


def dft_slow(x):
    """
    Compute DFT using explicit matrix multiplication (O(N^2)).

    Args:
        x (ndarray): Input sequence of length N

    Returns:
        ndarray: DFT coefficients X[k]
    """
    N = len(x)
    W = dft_matrix(N)
    return W @ x


# =============================================================================
# 2. Window functions
# =============================================================================

WINDOWS = {
    'Rectangular': lambda N: np.ones(N),
    'Hanning':     lambda N: np.hanning(N),
    'Hamming':     lambda N: np.hamming(N),
    'Blackman':    lambda N: np.blackman(N),
}


def apply_window(x, window_name):
    """
    Apply a named window function to signal x.

    Args:
        x           (ndarray): Input signal
        window_name (str):     One of 'Rectangular', 'Hanning', 'Hamming', 'Blackman'

    Returns:
        ndarray: Windowed signal
    """
    w = WINDOWS[window_name](len(x))
    return x * w


# =============================================================================
# 3. Demonstration functions
# =============================================================================

def demo_dft_vs_fft():
    """
    Verify that DFT matrix multiplication and np.fft.fft give identical results.

    Also compare computation times on increasing signal lengths.
    """
    print("=" * 60)
    print("Demo 1: DFT (Matrix) vs. FFT (numpy) — Correctness & Speed")
    print("=" * 60)

    # Correctness check on a small signal
    N = 32
    np.random.seed(42)
    x = np.random.randn(N) + 1j * np.random.randn(N)

    X_dft = dft_slow(x)
    X_fft = np.fft.fft(x)
    max_error = np.max(np.abs(X_dft - X_fft))
    print(f"  N={N}: max |DFT - FFT| = {max_error:.2e}  (should be ~machine epsilon)")

    # Timing comparison
    sizes = [64, 128, 256, 512, 1024, 2048]
    times_dft = []
    times_fft = []

    for N in sizes:
        x = np.random.randn(N)
        # Time DFT (matrix multiply)
        reps = max(1, 200 // N)
        t0 = time.perf_counter()
        for _ in range(reps):
            dft_slow(x)
        times_dft.append((time.perf_counter() - t0) / reps * 1000)
        # Time FFT
        reps_fft = 10000
        t0 = time.perf_counter()
        for _ in range(reps_fft):
            np.fft.fft(x)
        times_fft.append((time.perf_counter() - t0) / reps_fft * 1000)

    print(f"\n  {'N':>6}  {'DFT (ms)':>12}  {'FFT (ms)':>12}  {'Speedup':>10}")
    print(f"  {'-'*46}")
    for N, t_d, t_f in zip(sizes, times_dft, times_fft):
        print(f"  {N:>6}  {t_d:>12.4f}  {t_f:>12.6f}  {t_d/t_f:>9.1f}x")

    # Plot timing
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(sizes, times_dft, 'ro-', label='DFT O(N²)')
    ax.loglog(sizes, times_fft, 'bs-', label='FFT O(N log N)')
    # Reference curves
    n_arr = np.array(sizes, dtype=float)
    scale_dft = times_dft[0] / (sizes[0] ** 2)
    scale_fft = times_fft[0] / (sizes[0] * np.log2(sizes[0]))
    ax.loglog(n_arr, scale_dft * n_arr ** 2, 'r--', alpha=0.5, label='∝ N²')
    ax.loglog(n_arr, scale_fft * n_arr * np.log2(n_arr), 'b--', alpha=0.5, label='∝ N log N')
    ax.set_xlabel("Signal length N")
    ax.set_ylabel("Time (ms)")
    ax.set_title("DFT (matrix) vs. FFT Computation Time")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/06_dft_vs_fft_timing.png', dpi=150)
    print("\n  Saved timing plot to /tmp/06_dft_vs_fft_timing.png")
    plt.show()


def demo_zero_padding():
    """
    Show how zero-padding interpolates the DFT spectrum, giving finer
    frequency resolution in the displayed spectrum.

    IMPORTANT: Zero-padding does NOT add new information — it only
    interpolates the existing DFT bins, making peaks easier to locate.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Zero-Padding for Frequency Resolution")
    print("=" * 60)

    fs = 100.0    # Hz
    N = 64        # original number of samples
    t = np.arange(N) / fs

    # Signal: two closely spaced tones
    f1, f2 = 10.0, 13.5   # Hz
    x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

    zero_pad_lengths = [N, 2*N, 4*N, 16*N]
    labels = [f'N={n} (×{n//N})' for n in zero_pad_lengths]

    print(f"  Signal: {f1} Hz + {f2} Hz, N={N} samples, fs={fs} Hz")
    print(f"  Frequency resolution without padding: Δf = {fs/N:.2f} Hz")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Zero-Padding: Interpolating the DFT Spectrum", fontsize=14, fontweight='bold')

    for ax, nfft, label in zip(axes.flat, zero_pad_lengths, labels):
        # Pad signal to nfft points
        x_padded = np.zeros(nfft)
        x_padded[:N] = x
        X = np.fft.rfft(x_padded)
        f = np.fft.rfftfreq(nfft, d=1.0 / fs)
        mag = (2.0 / N) * np.abs(X)   # normalize by original N

        df = fs / nfft
        ax.plot(f, mag, label=f'Δf = {df:.2f} Hz')
        ax.axvline(f1, color='red',  linestyle='--', alpha=0.6, label=f'{f1} Hz')
        ax.axvline(f2, color='blue', linestyle='--', alpha=0.6, label=f'{f2} Hz')
        ax.set_xlim(0, 30)
        ax.set_title(f"NFFT = {nfft} ({label})")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/06_zero_padding.png', dpi=150)
    print("  Saved plot to /tmp/06_zero_padding.png")
    plt.show()


def demo_windowing_and_leakage():
    """
    Demonstrate spectral leakage and how windowing reduces it.

    Spectral leakage occurs when a non-integer number of cycles fits in
    the analysis window, causing energy to spread across many bins.

    Windows trade frequency resolution for reduced sidelobe level.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Spectral Leakage and Window Functions")
    print("=" * 60)

    fs = 100.0
    N = 128
    t = np.arange(N) / fs

    # Signal: strong tone at 10 Hz + weak tone at 18.7 Hz (non-integer bins)
    # The non-integer frequency causes severe leakage with rectangular window
    f_strong, f_weak = 10.0, 18.7
    A_strong, A_weak = 1.0, 0.02
    x = A_strong * np.sin(2 * np.pi * f_strong * t) + A_weak * np.sin(2 * np.pi * f_weak * t)

    print(f"  Signal: {A_strong}·sin(2π·{f_strong}·t) + {A_weak}·sin(2π·{f_weak}·t)")
    print(f"  Goal: detect weak {f_weak} Hz tone buried under strong {f_strong} Hz leakage\n")

    nfft = 4 * N   # zero-pad for display resolution
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)

    colors = ['b', 'g', 'r', 'purple']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Spectral Leakage and Windowing", fontsize=14, fontweight='bold')

    window_names = list(WINDOWS.keys())
    for ax, wname, color in zip(axes.flat, window_names, colors):
        x_win = apply_window(x, wname)
        w = WINDOWS[wname](N)

        # Pad and FFT
        x_padded = np.zeros(nfft)
        x_padded[:N] = x_win
        X = np.fft.rfft(x_padded)
        # Normalize by sum of window coefficients
        mag_db = 20 * np.log10(np.abs(X) / (np.sum(w) / 2) + 1e-12)

        ax.plot(f, mag_db, color=color, lw=1.5, label=wname)
        ax.axvline(f_weak, color='orange', linestyle='--', alpha=0.7,
                   label=f'Weak tone {f_weak} Hz')
        ax.axhline(20 * np.log10(A_weak), color='gray', linestyle=':', alpha=0.5,
                   label=f'True level ({20*np.log10(A_weak):.0f} dB)')
        ax.set_xlim(0, 40)
        ax.set_ylim(-80, 5)
        ax.set_title(wname)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Report peak near weak tone
        mask_weak = (f > 16) & (f < 22)
        peak_db = np.max(mag_db[mask_weak]) if mask_weak.any() else -80
        print(f"  {wname:12s}: weak tone peak near {f_weak} Hz = {peak_db:.1f} dB")

    plt.tight_layout()
    plt.savefig('/tmp/06_windowing.png', dpi=150)
    print("\n  Saved plot to /tmp/06_windowing.png")
    plt.show()


def demo_multitone_analysis():
    """
    Use FFT to identify frequency components in a multi-tone audio-like signal.

    Signal contains 5 tones of varying amplitudes and a noise floor.
    This simulates a real spectrum analysis task.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Multi-Tone Signal Analysis")
    print("=" * 60)

    fs = 1000.0
    duration = 2.0
    N = int(fs * duration)
    t = np.arange(N) / fs
    np.random.seed(7)

    # Define tones: (frequency_Hz, amplitude)
    tones = [(50, 1.0), (120, 0.5), (230, 0.8), (340, 0.3), (470, 0.6)]
    x = np.zeros(N)
    for f_tone, amp in tones:
        x += amp * np.sin(2 * np.pi * f_tone * t)
    # Add Gaussian noise (SNR ~ 20 dB)
    noise_level = 0.1
    x += noise_level * np.random.randn(N)

    print("  Tones: " + ", ".join(f"{f} Hz ({A})" for f, A in tones))
    print(f"  Noise level: {noise_level}")

    # Apply Hanning window before FFT
    x_windowed = apply_window(x, 'Hanning')
    w = np.hanning(N)
    X = np.fft.rfft(x_windowed)
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    mag = (2.0 / np.sum(w)) * np.abs(X)

    # Find peaks (simple threshold)
    threshold = 0.15
    peak_mask = mag > threshold
    peak_freqs = f[peak_mask]
    peak_mags = mag[peak_mask]
    # Keep only local maxima
    from scipy.signal import find_peaks
    peaks_idx, _ = find_peaks(mag, height=threshold, distance=20)

    print(f"\n  Detected peaks (threshold = {threshold}):")
    for idx in peaks_idx:
        print(f"    f = {f[idx]:.1f} Hz,  amplitude ≈ {mag[idx]:.3f}")

    fig = plt.figure(figsize=(13, 7))
    gs = GridSpec(2, 1, figure=fig, hspace=0.4)
    fig.suptitle("Multi-Tone Signal Analysis (Hanning Window)", fontsize=14, fontweight='bold')

    # Time domain
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t[:500], x[:500], lw=0.8, color='steelblue', label='Signal (first 0.5 s)')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Time Domain (zoomed)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Frequency domain
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(f, mag, color='steelblue', lw=1.2, label='Magnitude spectrum')
    ax2.plot(f[peaks_idx], mag[peaks_idx], 'rv', markersize=10, label='Detected peaks')
    for idx in peaks_idx:
        ax2.annotate(f'{f[idx]:.0f} Hz', xy=(f[idx], mag[idx]),
                     xytext=(0, 8), textcoords='offset points',
                     ha='center', fontsize=8, color='red')
    ax2.axhline(threshold, color='orange', linestyle='--', alpha=0.7, label=f'Threshold={threshold}')
    ax2.set_xlim(0, fs / 2)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Frequency Domain (FFT with Hanning window)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.savefig('/tmp/06_multitone.png', dpi=150)
    print("  Saved plot to /tmp/06_multitone.png")
    plt.show()


if __name__ == "__main__":
    print("Discrete Fourier Transform (DFT) and FFT Analysis")
    print("=" * 60)
    print("DFT:  X[k] = sum_{n=0}^{N-1} x[n] * exp(-j2πkn/N)")
    print("FFT:  Same result, but O(N log N) instead of O(N²)\n")

    demo_dft_vs_fft()
    demo_zero_padding()
    demo_windowing_and_leakage()
    demo_multitone_analysis()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - DFT via matrix multiply and FFT give identical results")
    print("  - FFT speedup grows with N: O(N²) → O(N log N)")
    print("  - Zero-padding interpolates spectrum (more display bins)")
    print("    but does NOT improve true frequency resolution (Δf = fs/N_orig)")
    print("  - Spectral leakage: non-integer-period signals smear energy")
    print("  - Windows reduce leakage at cost of slightly wider main lobe:")
    print("      Rectangular < Hanning ≈ Hamming < Blackman (sidelobe suppression)")
    print("  - Hanning or Hamming are good default choices for spectrum analysis")
    print("=" * 60)
