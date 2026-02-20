#!/usr/bin/env python3
"""
Digital Filter Fundamentals: FIR vs IIR
========================================

This script compares Finite Impulse Response (FIR) and Infinite Impulse
Response (IIR) filters, covering:

- Magnitude response, phase response, and group delay
- Simple FIR filter: moving average (lowpass)
- Simple IIR filter: 1st-order recursive (exponential smoother)
- Impulse response and step response comparison
- Applying both filters to a noisy signal

Key Concepts:
    FIR:  y[n] = sum_{k=0}^{M} b_k * x[n-k]          (always stable)
    IIR:  y[n] = sum b_k x[n-k] - sum a_k y[n-k]     (can be unstable)

    Group delay: τ(ω) = -dφ(ω)/dω
    FIR with symmetric coefficients → linear phase → constant group delay

Author: Educational example for Signal Processing
License: MIT
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# ============================================================================
# FILTER DEFINITIONS
# ============================================================================

def moving_average_fir(M):
    """
    Moving average FIR filter of length M.

    H(z) = (1/M) * (1 + z^{-1} + ... + z^{-(M-1)})
    Frequency response: H(e^{jω}) = (1/M) * sin(Mω/2) / sin(ω/2) * e^{-j(M-1)ω/2}

    This is a symmetric (linear-phase) FIR lowpass filter.
    The cutoff frequency is approximately 0.9/M * π rad/sample.

    Args:
        M (int): Filter length (number of taps)

    Returns:
        ndarray: Filter coefficients b (length M), a = [1]
    """
    b = np.ones(M) / M
    a = np.array([1.0])
    return b, a


def first_order_iir(alpha):
    """
    1st-order IIR recursive (exponential smoothing) filter.

    y[n] = α * x[n] + (1 - α) * y[n-1]

    Transfer function: H(z) = α / (1 - (1-α) z^{-1})
    Pole at z = (1 - α) — stable for 0 < α < 2.

    α close to 0 → heavy smoothing (narrow bandwidth)
    α close to 1 → minimal smoothing (wide bandwidth)

    Args:
        alpha (float): Smoothing factor, 0 < alpha < 1

    Returns:
        tuple: (b, a) coefficient arrays
    """
    b = np.array([alpha])
    a = np.array([1.0, -(1.0 - alpha)])
    return b, a


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_group_delay(b, a, worN=512):
    """
    Compute group delay τ(ω) = -dφ(ω)/dω using scipy.signal.group_delay.

    Args:
        b, a: Filter coefficients
        worN (int): Number of frequency points

    Returns:
        tuple: (frequencies in rad/sample, group delay in samples)
    """
    w, gd = signal.group_delay((b, a), w=worN)
    return w, gd


def impulse_response(b, a, n_samples=60):
    """
    Compute the filter's impulse response h[n] by filtering δ[n].

    Args:
        b, a: Filter coefficients
        n_samples (int): Number of output samples

    Returns:
        ndarray: h[n] for n = 0, 1, ..., n_samples-1
    """
    delta = np.zeros(n_samples)
    delta[0] = 1.0
    return signal.lfilter(b, a, delta)


def step_response(b, a, n_samples=60):
    """
    Compute the filter's step response s[n] by filtering u[n].

    Args:
        b, a: Filter coefficients
        n_samples (int): Number of output samples

    Returns:
        ndarray: s[n] for n = 0, 1, ..., n_samples-1
    """
    step = np.ones(n_samples)
    return signal.lfilter(b, a, step)


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

def demo_frequency_responses(M=15, alpha=0.2):
    """Compare FIR and IIR frequency responses."""
    print("=" * 65)
    print("1. FREQUENCY RESPONSE COMPARISON: FIR vs IIR")
    print("=" * 65)

    b_fir, a_fir = moving_average_fir(M)
    b_iir, a_iir = first_order_iir(alpha)

    w, H_fir = signal.freqz(b_fir, a_fir, worN=512)
    _, H_iir = signal.freqz(b_iir, a_iir, worN=512)

    _, gd_fir = compute_group_delay(b_fir, a_fir)
    _, gd_iir = compute_group_delay(b_iir, a_iir)

    print(f"\nFIR Moving Average (M={M} taps)")
    print(f"  DC gain (ω=0): {np.abs(H_fir[0]):.4f}  (should be 1.0)")
    print(f"  Half-power freq: ~{0.9 / M:.3f} × π rad/sample")
    print(f"  Group delay: constant = {np.mean(gd_fir[:50]):.1f} samples  (linear phase)")

    print(f"\nIIR Exponential Smoother (α={alpha})")
    print(f"  DC gain (ω=0): {np.abs(H_iir[0]):.4f}  (should be 1.0)")
    print(f"  Pole location: z = {1 - alpha:.2f}  (inside unit circle → stable)")
    print(f"  Group delay at ω=0: {gd_iir[0]:.2f} samples  (non-linear phase)")

    return (w, H_fir, H_iir, gd_fir, gd_iir), (b_fir, a_fir), (b_iir, a_iir)


def demo_impulse_step_response(b_fir, a_fir, b_iir, a_iir):
    """Show impulse and step responses for both filters."""
    print("\n" + "=" * 65)
    print("2. IMPULSE AND STEP RESPONSES")
    print("=" * 65)

    h_fir = impulse_response(b_fir, a_fir, n_samples=50)
    h_iir = impulse_response(b_iir, a_iir, n_samples=50)
    s_fir = step_response(b_fir, a_fir, n_samples=50)
    s_iir = step_response(b_iir, a_iir, n_samples=50)

    print(f"\nFIR impulse response: finite support, {np.sum(h_fir != 0)} nonzero samples")
    print(f"  Final value h[49] = {h_fir[49]:.6f}  (exactly zero after M taps)")
    print(f"  Steady-state step response: {s_fir[49]:.4f}  (DC gain = 1)")

    alpha = b_iir[0]
    print(f"\nIIR impulse response: infinite support (exponentially decaying)")
    print(f"  h[0]  = {h_iir[0]:.4f}  (= α = {alpha})")
    print(f"  h[10] = {h_iir[10]:.6f}  (= α * (1-α)^10 = {alpha * (1-alpha)**10:.6f})")
    print(f"  h[49] = {h_iir[49]:.2e}  (never exactly zero)")
    print(f"  Steady-state step response: {s_iir[49]:.4f}  (DC gain = 1)")

    return h_fir, h_iir, s_fir, s_iir


def demo_noise_filtering(b_fir, a_fir, b_iir, a_iir):
    """Apply both filters to a noisy signal and compare."""
    print("\n" + "=" * 65)
    print("3. FILTERING A NOISY SIGNAL")
    print("=" * 65)

    # Generate test signal: slow 5 Hz sine + high-frequency noise
    fs = 1000          # sampling frequency (Hz)
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    clean = np.sin(2 * np.pi * 5 * t)        # 5 Hz signal
    np.random.seed(0)
    noise = 0.5 * np.random.randn(len(t))    # broadband noise
    noisy = clean + noise

    y_fir = signal.lfilter(b_fir, a_fir, noisy)
    y_iir = signal.lfilter(b_iir, a_iir, noisy)

    # SNR calculation (over second half to avoid transient)
    half = len(t) // 2
    snr_orig  = 10 * np.log10(np.var(clean[half:]) / np.var(noise[half:]))
    snr_fir   = 10 * np.log10(np.var(clean[half:]) / np.var(y_fir[half:] - clean[half:]))
    snr_iir   = 10 * np.log10(np.var(clean[half:]) / np.var(y_iir[half:] - clean[half:]))

    print(f"\nTest signal: 5 Hz sine + Gaussian noise (SNR = {snr_orig:.1f} dB)")
    print(f"  FIR output SNR : {snr_fir:.1f} dB  (improvement: {snr_fir - snr_orig:.1f} dB)")
    print(f"  IIR output SNR : {snr_iir:.1f} dB  (improvement: {snr_iir - snr_orig:.1f} dB)")

    M_fir = len(b_fir)
    print(f"\nFIR introduces a constant delay of {(M_fir - 1) // 2} samples (linear phase)")
    print(f"IIR introduces a signal-dependent (non-linear) phase distortion")

    return t, noisy, clean, y_fir, y_iir


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("DIGITAL FILTER FUNDAMENTALS: FIR vs IIR")
    print("=" * 65)

    M = 15       # FIR moving average taps
    alpha = 0.2  # IIR smoothing factor (smaller = more smoothing)

    freq_data, (b_fir, a_fir), (b_iir, a_iir) = demo_frequency_responses(M, alpha)
    w, H_fir, H_iir, gd_fir, gd_iir = freq_data

    h_fir, h_iir, s_fir, s_iir = demo_impulse_step_response(b_fir, a_fir, b_iir, a_iir)
    t, noisy, clean, y_fir, y_iir = demo_noise_filtering(b_fir, a_fir, b_iir, a_iir)

    # -----------------------------------------------------------------------
    # VISUALIZATION — 3×2 grid
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle('Digital Filter Fundamentals: FIR vs IIR', fontsize=14, fontweight='bold')

    norm_freq = w / np.pi   # normalized frequency 0..1

    # (0,0) Magnitude response
    axes[0, 0].plot(norm_freq, 20 * np.log10(np.maximum(np.abs(H_fir), 1e-12)),
                    'b-', linewidth=2, label=f'FIR (M={M})')
    axes[0, 0].plot(norm_freq, 20 * np.log10(np.maximum(np.abs(H_iir), 1e-12)),
                    'r--', linewidth=2, label=f'IIR (α={alpha})')
    axes[0, 0].set_xlabel('Normalized Frequency (×π rad/sample)')
    axes[0, 0].set_ylabel('Magnitude (dB)')
    axes[0, 0].set_title('Magnitude Response')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(-50, 5)

    # (0,1) Phase response
    phase_fir = np.unwrap(np.angle(H_fir))
    phase_iir = np.unwrap(np.angle(H_iir))
    axes[0, 1].plot(norm_freq, phase_fir, 'b-', linewidth=2, label=f'FIR (linear phase)')
    axes[0, 1].plot(norm_freq, phase_iir, 'r--', linewidth=2, label='IIR (non-linear phase)')
    axes[0, 1].set_xlabel('Normalized Frequency (×π rad/sample)')
    axes[0, 1].set_ylabel('Phase (radians)')
    axes[0, 1].set_title('Phase Response')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 1)

    # (1,0) Group delay
    axes[1, 0].plot(norm_freq, gd_fir, 'b-', linewidth=2, label='FIR (constant)')
    axes[1, 0].plot(norm_freq, np.clip(gd_iir, -2, 30), 'r--', linewidth=2,
                    label='IIR (frequency-dependent)')
    axes[1, 0].set_xlabel('Normalized Frequency (×π rad/sample)')
    axes[1, 0].set_ylabel('Group Delay (samples)')
    axes[1, 0].set_title('Group Delay')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 1)

    # (1,1) Impulse response
    n = np.arange(len(h_fir))
    axes[1, 1].stem(n, h_fir, basefmt='k-', linefmt='C0-', markerfmt='C0o',
                    label=f'FIR h[n]')
    axes[1, 1].stem(n, h_iir, basefmt='k-', linefmt='C1--', markerfmt='C1^',
                    label='IIR h[n]')
    axes[1, 1].set_xlabel('Sample n')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Impulse Response h[n]\n(FIR: finite; IIR: infinite)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(-1, 50)

    # (2,0) Step response
    axes[2, 0].plot(n, s_fir, 'b-o', markersize=3, linewidth=1.5, label='FIR s[n]')
    axes[2, 0].plot(n, s_iir, 'r--^', markersize=3, linewidth=1.5, label='IIR s[n]')
    axes[2, 0].axhline(1.0, color='k', linestyle=':', linewidth=1, label='Steady state = 1')
    axes[2, 0].set_xlabel('Sample n')
    axes[2, 0].set_ylabel('Amplitude')
    axes[2, 0].set_title('Step Response s[n]')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlim(-1, 50)

    # (2,1) Noisy signal filtering
    axes[2, 1].plot(t, noisy, 'gray', linewidth=0.8, alpha=0.6, label='Noisy signal')
    axes[2, 1].plot(t, clean, 'k-', linewidth=2, alpha=0.5, label='Clean 5 Hz signal')
    axes[2, 1].plot(t, y_fir, 'b-', linewidth=1.8, label=f'FIR filtered')
    axes[2, 1].plot(t, y_iir, 'r--', linewidth=1.8, label=f'IIR filtered')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Amplitude')
    axes[2, 1].set_title('Filtering a Noisy Signal')
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = '/opt/projects/01_Personal/03_Study/examples/Signal_Processing/08_filter_response.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved visualization: {out_path}")
    plt.close()

    print("\n" + "=" * 65)
    print("Key Takeaways:")
    print("  - FIR: finite impulse response, always stable, linear phase")
    print("  - IIR: infinite impulse response, computationally efficient")
    print("  - Linear phase (FIR) → constant group delay → no waveform distortion")
    print("  - Group delay variation (IIR) distorts signal shape")
    print("  - Both achieve unity DC gain but differ in roll-off shape")
    print("=" * 65)
