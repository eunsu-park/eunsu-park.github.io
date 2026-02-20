#!/usr/bin/env python3
"""
FIR Filter Design Methods
==========================

This script demonstrates three principal FIR filter design techniques:

1. Window Method — Hamming window (lowpass)
2. Window Method — Kaiser window (lowpass, tunable stopband attenuation)
3. Window Method — Hamming window (bandpass)
4. Parks-McClellan (Remez) optimal equiripple design (lowpass)
5. Comparison of window method vs optimal design
6. Applying the designed filters to a test signal

Key Concepts:
    - Ideal filter → multiply by window → practical FIR filter
    - Window choice controls sidelobe level vs transition width trade-off
    - Parks-McClellan minimises the maximum error (Chebyshev criterion)
    - Kaiser window: adjustable β controls sidelobe attenuation

Design relationships (window method):
    - Filter order N determined by transition width and window type
    - Hamming: stopband attenuation ≈ 53 dB
    - Kaiser:  As = 2.285 * (N-1) * Δω + 7.95  (Harris formula)

Author: Educational example for Signal Processing
License: MIT
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# ============================================================================
# DESIGN HELPERS
# ============================================================================

def kaiser_parameters(As_dB, delta_omega):
    """
    Estimate Kaiser window β and filter order from design specifications.

    Using the Kaiser empirical formulas (Harris 1978):
        β = 0                            if As < 21 dB
        β = 0.5842*(As-21)^0.4 + 0.07886*(As-21)   if 21 ≤ As ≤ 50 dB
        β = 0.1102*(As-8.7)              if As > 50 dB

        N ≈ (As - 7.95) / (2.285 * Δω) + 1   (Δω in rad/sample)

    Args:
        As_dB (float): Desired stopband attenuation in dB (positive value)
        delta_omega (float): Transition width in rad/sample

    Returns:
        tuple: (N, beta) — filter order and Kaiser β parameter
    """
    if As_dB < 21:
        beta = 0.0
    elif As_dB <= 50:
        beta = 0.5842 * (As_dB - 21) ** 0.4 + 0.07886 * (As_dB - 21)
    else:
        beta = 0.1102 * (As_dB - 8.7)

    N = int(np.ceil((As_dB - 7.95) / (2.285 * delta_omega))) + 1
    if N % 2 == 0:
        N += 1   # ensure odd length for symmetric filter (type I)

    return N, beta


# ============================================================================
# DESIGN FUNCTIONS
# ============================================================================

def design_lowpass_hamming(cutoff_norm, N):
    """
    Design FIR lowpass filter using the Hamming window method.

    Steps:
        1. Compute ideal sinc-based impulse response h_ideal[n]
        2. Multiply by Hamming window w[n]
        3. Result is a linear-phase FIR filter

    Args:
        cutoff_norm (float): Normalized cutoff frequency (0 to 1, where 1 = Nyquist)
        N (int): Filter order (number of taps = N + 1, should be even for odd length)

    Returns:
        ndarray: FIR filter coefficients (length N+1)
    """
    # scipy.signal.firwin uses half-band convention: cutoff in cycles/sample
    # fs=2 maps cutoff_norm in [0,1] to Hz, where 1 corresponds to Nyquist
    b = signal.firwin(N + 1, cutoff_norm, window='hamming')
    return b


def design_lowpass_kaiser(cutoff_norm, As_dB, delta_norm):
    """
    Design FIR lowpass filter using the Kaiser window method.

    The Kaiser window allows direct specification of stopband attenuation
    and automatically computes the required filter order.

    Args:
        cutoff_norm (float): Normalized cutoff (0 to 1, where 1 = Nyquist)
        As_dB (float): Minimum stopband attenuation in dB
        delta_norm (float): Transition band width, normalized (0 to 1)

    Returns:
        tuple: (b, N, beta) — coefficients, actual order, and Kaiser β
    """
    delta_omega = delta_norm * np.pi   # convert to rad/sample
    N, beta = kaiser_parameters(As_dB, delta_omega)
    b = signal.firwin(N, cutoff_norm, window=('kaiser', beta))
    return b, N, beta


def design_bandpass_hamming(low_norm, high_norm, N):
    """
    Design FIR bandpass filter using the Hamming window method.

    Args:
        low_norm (float): Lower normalized cutoff (0 to 1)
        high_norm (float): Upper normalized cutoff (0 to 1)
        N (int): Filter order

    Returns:
        ndarray: FIR filter coefficients (length N+1)
    """
    # firwin with two cutoffs and pass_zero=False → bandpass
    b = signal.firwin(N + 1, [low_norm, high_norm], pass_zero=False, window='hamming')
    return b


def design_optimal_remez(cutoff_norm, trans_norm, As_dB):
    """
    Design optimal FIR lowpass filter using Parks-McClellan (Remez) algorithm.

    The Remez algorithm minimises the maximum weighted error between the
    desired and actual frequency response — equiripple in both bands.

    Estimate filter order using Kaiser's order formula as a starting point.

    Args:
        cutoff_norm (float): Normalized passband edge (0 to 1)
        trans_norm (float): Normalized transition band width (0 to 1)
        As_dB (float): Approximate stopband attenuation in dB

    Returns:
        tuple: (b, N) — filter coefficients and filter order
    """
    delta_omega = trans_norm * np.pi
    N, _ = kaiser_parameters(As_dB, delta_omega)

    # Define band edges: [0, f_pass, f_stop, 1] in normalized units (0..1)
    f_pass = cutoff_norm - trans_norm / 2
    f_stop = cutoff_norm + trans_norm / 2

    # Clamp to valid range
    f_pass = max(f_pass, 0.01)
    f_stop = min(f_stop, 0.99)

    bands = [0, f_pass, f_stop, 1.0]
    desired = [1, 0]             # passband gain = 1, stopband gain = 0

    # Convert passband ripple / stopband attenuation to relative weights
    # δ_p ≈ δ_s for equal-ripple; weight = 1 everywhere for simplicity
    b = signal.remez(N, bands, desired, fs=2.0)
    return b, N


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

def demo_lowpass_designs():
    """Design and compare lowpass FIR filters."""
    print("=" * 65)
    print("1. LOWPASS FIR FILTER DESIGNS")
    print("=" * 65)

    cutoff = 0.3        # normalized cutoff (fraction of Nyquist)
    N_hamming = 50      # fixed order for Hamming

    # Hamming window design
    b_hamming = design_lowpass_hamming(cutoff, N_hamming)
    w, H_hamming = signal.freqz(b_hamming, worN=1024)

    print(f"\nHamming Window Lowpass (N={N_hamming}, cutoff={cutoff}×Nyquist)")
    # Find -3 dB point
    mag = np.abs(H_hamming)
    idx_3dB = np.argmin(np.abs(mag - (1 / np.sqrt(2))))
    print(f"  Taps   : {len(b_hamming)}")
    print(f"  -3 dB  : {w[idx_3dB] / np.pi:.3f} × π rad/sample")
    # Stopband attenuation (at 0.4 Nyquist and above)
    idx_stop = np.where(w / np.pi > cutoff + 0.1)[0][0]
    stop_atten = -20 * np.log10(np.maximum(np.max(mag[idx_stop:]), 1e-12))
    print(f"  Stopband attenuation: ≥ {stop_atten:.1f} dB")

    # Kaiser window design
    As_dB = 60.0
    delta_norm = 0.08
    b_kaiser, N_kaiser, beta = design_lowpass_kaiser(cutoff, As_dB, delta_norm)
    w, H_kaiser = signal.freqz(b_kaiser, worN=1024)

    print(f"\nKaiser Window Lowpass (As={As_dB} dB, Δf={delta_norm}×Nyquist)")
    print(f"  Computed N  : {N_kaiser}")
    print(f"  Kaiser β    : {beta:.4f}")
    print(f"  Taps        : {len(b_kaiser)}")
    mag_k = np.abs(H_kaiser)
    stop_atten_k = -20 * np.log10(np.maximum(np.max(mag_k[idx_stop:]), 1e-12))
    print(f"  Stopband attenuation: ≥ {stop_atten_k:.1f} dB")

    return b_hamming, b_kaiser, cutoff


def demo_bandpass_design():
    """Design a bandpass FIR filter."""
    print("\n" + "=" * 65)
    print("2. BANDPASS FIR FILTER (Hamming Window)")
    print("=" * 65)

    low_norm = 0.2     # lower cutoff (fraction of Nyquist)
    high_norm = 0.5    # upper cutoff
    N = 60

    b_bp = design_bandpass_hamming(low_norm, high_norm, N)
    w, H_bp = signal.freqz(b_bp, worN=1024)

    print(f"\nBandpass: passband {low_norm}–{high_norm} × Nyquist, N={N}")
    print(f"  Taps: {len(b_bp)}")

    # Report gain at passband center and stopband edges
    center = (low_norm + high_norm) / 2
    mag_bp = np.abs(H_bp)
    idx_center = np.argmin(np.abs(w / np.pi - center))
    print(f"  Gain at center ({center:.2f}×π): {20*np.log10(mag_bp[idx_center]):.2f} dB")

    idx_stop_lo = np.argmin(np.abs(w / np.pi - 0.05))
    idx_stop_hi = np.argmin(np.abs(w / np.pi - 0.85))
    print(f"  Gain at lower stopband (0.05×π): {20*np.log10(max(mag_bp[idx_stop_lo], 1e-12)):.1f} dB")
    print(f"  Gain at upper stopband (0.85×π): {20*np.log10(max(mag_bp[idx_stop_hi], 1e-12)):.1f} dB")

    return b_bp, low_norm, high_norm


def demo_remez_vs_window(cutoff, b_hamming):
    """Compare Parks-McClellan optimal design with Hamming window method."""
    print("\n" + "=" * 65)
    print("3. PARKS-McCLELLAN (REMEZ) vs WINDOW METHOD")
    print("=" * 65)

    As_dB = 53.0       # target matching Hamming's ~53 dB
    trans_norm = 0.08

    b_remez, N_remez = design_optimal_remez(cutoff, trans_norm, As_dB)
    w, H_remez = signal.freqz(b_remez, worN=1024)
    _, H_hamming = signal.freqz(b_hamming, worN=1024)

    mag_r = np.abs(H_remez)
    mag_h = np.abs(H_hamming)

    # Passband ripple
    idx_pb_end = np.argmin(np.abs(w / np.pi - (cutoff - trans_norm / 2)))
    pb_ripple_r = 20 * np.log10(np.max(mag_r[:idx_pb_end]) / np.min(mag_r[1:idx_pb_end]))
    pb_ripple_h = 20 * np.log10(np.max(mag_h[:idx_pb_end]) / np.min(mag_h[1:idx_pb_end]))

    # Stopband attenuation
    idx_sb_start = np.argmin(np.abs(w / np.pi - (cutoff + trans_norm / 2)))
    sb_atten_r = -20 * np.log10(np.maximum(np.max(mag_r[idx_sb_start:]), 1e-12))
    sb_atten_h = -20 * np.log10(np.maximum(np.max(mag_h[idx_sb_start:]), 1e-12))

    print(f"\nCutoff = {cutoff}×Nyquist, target As = {As_dB} dB")
    print(f"\n{'Method':<20} {'Taps':>6} {'PB Ripple':>12} {'SB Atten':>12}")
    print("-" * 54)
    print(f"{'Hamming Window':<20} {len(b_hamming):>6} {pb_ripple_h:>11.2f} dB {sb_atten_h:>10.1f} dB")
    print(f"{'Remez (optimal)':<20} {len(b_remez):>6} {pb_ripple_r:>11.2f} dB {sb_atten_r:>10.1f} dB")
    print(f"\nRemez achieves equiripple in both bands (Chebyshev criterion).")
    print(f"Window method is simpler but less flexible in band control.")

    return b_remez, w, H_remez, H_hamming


def demo_apply_to_signal(b_hamming, b_kaiser, b_bp):
    """Apply designed filters to a multi-tone test signal."""
    print("\n" + "=" * 65)
    print("4. APPLYING FILTERS TO A TEST SIGNAL")
    print("=" * 65)

    # Test signal: sum of sinusoids at 5, 15, 30, and 45 Hz
    # Sampling at 100 Hz → Nyquist = 50 Hz
    fs = 100.0
    t = np.linspace(0, 2.0, int(fs * 2), endpoint=False)
    f1, f2, f3, f4 = 5, 15, 30, 45     # Hz
    # Normalized: f1/Nyq=0.1, f2=0.3, f3=0.6, f4=0.9
    x = (np.sin(2 * np.pi * f1 * t) +
         np.sin(2 * np.pi * f2 * t) +
         np.sin(2 * np.pi * f3 * t) +
         np.sin(2 * np.pi * f4 * t))

    # Lowpass (Hamming, cutoff=0.3 → 15 Hz): keep f1, f2; reject f3, f4
    y_lp = signal.lfilter(b_hamming, 1, x)

    # Lowpass (Kaiser, cutoff=0.3, 60 dB): same goal, better attenuation
    y_kaiser = signal.lfilter(b_kaiser, 1, x)

    # Bandpass (0.2–0.5 Nyquist → 10–25 Hz): keep f2; reject f1, f3, f4
    y_bp = signal.lfilter(b_bp, 1, x)

    print(f"\nTest signal: {f1}, {f2}, {f3}, {f4} Hz  (fs={fs} Hz)")
    print(f"  Normalized:  {f1/50:.2f}, {f2/50:.2f}, {f3/50:.2f}, {f4/50:.2f} × Nyquist")
    print(f"\nLowpass Hamming (cutoff=0.3×Nyq = {0.3*50:.0f} Hz):")
    print(f"  Expected: pass {f1}+{f2} Hz, reject {f3}+{f4} Hz")
    print(f"\nKaiser (same cutoff, 60 dB stopband):")
    print(f"  Better stopband attenuation than Hamming")
    print(f"\nBandpass (0.2–0.5 × Nyq = {0.2*50:.0f}–{0.5*50:.0f} Hz):")
    print(f"  Expected: pass {f2} Hz, reject {f1}+{f3}+{f4} Hz")

    return t, x, y_lp, y_kaiser, y_bp


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("FIR FILTER DESIGN METHODS")
    print("=" * 65)

    b_hamming, b_kaiser, cutoff = demo_lowpass_designs()
    b_bp, low_norm, high_norm = demo_bandpass_design()
    b_remez, w_cmp, H_remez, H_hamming_cmp = demo_remez_vs_window(cutoff, b_hamming)
    t, x, y_lp, y_kaiser, y_bp = demo_apply_to_signal(b_hamming, b_kaiser, b_bp)

    # -----------------------------------------------------------------------
    # VISUALIZATION — 3×2 grid
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle('FIR Filter Design Methods', fontsize=14, fontweight='bold')

    w_h, H_h = signal.freqz(b_hamming, worN=1024)
    w_k, H_k = signal.freqz(b_kaiser,  worN=1024)
    w_bp, H_bp = signal.freqz(b_bp,    worN=1024)

    # (0,0) Lowpass magnitude responses (Hamming vs Kaiser)
    axes[0, 0].plot(w_h / np.pi, 20 * np.log10(np.maximum(np.abs(H_h), 1e-12)),
                    'b-', linewidth=2, label=f'Hamming (N={len(b_hamming)-1})')
    axes[0, 0].plot(w_k / np.pi, 20 * np.log10(np.maximum(np.abs(H_k), 1e-12)),
                    'r--', linewidth=2, label=f'Kaiser (N={len(b_kaiser)-1}, β)')
    axes[0, 0].axvline(cutoff, color='k', linestyle=':', linewidth=1, label=f'Cutoff {cutoff}×π')
    axes[0, 0].set_xlabel('Normalized Frequency (×π rad/sample)')
    axes[0, 0].set_ylabel('Magnitude (dB)')
    axes[0, 0].set_title('Lowpass: Hamming vs Kaiser Window')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(-90, 5)

    # (0,1) Lowpass impulse responses
    axes[0, 1].plot(b_hamming, 'b-o', markersize=3, linewidth=1.5, label='Hamming')
    axes[0, 1].plot(np.linspace(0, len(b_hamming) - 1, len(b_kaiser)),
                    b_kaiser, 'r--^', markersize=3, linewidth=1.5, label='Kaiser')
    axes[0, 1].set_xlabel('Tap index n')
    axes[0, 1].set_ylabel('Coefficient value')
    axes[0, 1].set_title('Impulse Response (Filter Coefficients)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # (1,0) Bandpass magnitude response
    axes[1, 0].plot(w_bp / np.pi, 20 * np.log10(np.maximum(np.abs(H_bp), 1e-12)),
                    'g-', linewidth=2)
    axes[1, 0].axvspan(low_norm, high_norm, alpha=0.15, color='green', label='Passband')
    axes[1, 0].set_xlabel('Normalized Frequency (×π rad/sample)')
    axes[1, 0].set_ylabel('Magnitude (dB)')
    axes[1, 0].set_title(f'Bandpass: {low_norm}–{high_norm} × Nyquist (Hamming)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(-90, 5)

    # (1,1) Remez vs Window comparison
    axes[1, 1].plot(w_cmp / np.pi, 20 * np.log10(np.maximum(np.abs(H_hamming_cmp), 1e-12)),
                    'b-', linewidth=2, label=f'Hamming (N={len(b_hamming)-1})')
    axes[1, 1].plot(w_cmp / np.pi, 20 * np.log10(np.maximum(np.abs(H_remez), 1e-12)),
                    'm--', linewidth=2, label=f'Remez/PM (N={len(b_remez)-1})')
    axes[1, 1].axvline(cutoff, color='k', linestyle=':', linewidth=1)
    axes[1, 1].set_xlabel('Normalized Frequency (×π rad/sample)')
    axes[1, 1].set_ylabel('Magnitude (dB)')
    axes[1, 1].set_title('Window Method vs Parks-McClellan (Remez)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(-90, 5)

    # (2,0) Test signal time domain — first 0.5 s
    idx = int(0.5 * 100)   # first 0.5 s at fs=100 Hz
    axes[2, 0].plot(t[:idx], x[:idx], 'gray', linewidth=0.8, alpha=0.5, label='Input (4 tones)')
    axes[2, 0].plot(t[:idx], y_lp[:idx], 'b-', linewidth=2, label='Lowpass (Hamming)')
    axes[2, 0].plot(t[:idx], y_bp[:idx], 'g--', linewidth=2, label='Bandpass')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Amplitude')
    axes[2, 0].set_title('Filtered Test Signal (time domain)')
    axes[2, 0].legend(fontsize=9)
    axes[2, 0].grid(True, alpha=0.3)

    # (2,1) Spectrum of input vs filtered signals
    N_fft = len(x)
    freqs = np.fft.rfftfreq(N_fft, d=1.0 / 100.0)
    X_mag = np.abs(np.fft.rfft(x)) / N_fft
    YLP_mag = np.abs(np.fft.rfft(y_lp)) / N_fft
    YBP_mag = np.abs(np.fft.rfft(y_bp)) / N_fft

    axes[2, 1].plot(freqs, X_mag, 'gray', linewidth=1.5, alpha=0.6, label='Input spectrum')
    axes[2, 1].plot(freqs, YLP_mag, 'b-', linewidth=2, label='Lowpass out')
    axes[2, 1].plot(freqs, YBP_mag, 'g--', linewidth=2, label='Bandpass out')
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('Magnitude')
    axes[2, 1].set_title('Filtered Test Signal (frequency domain)')
    axes[2, 1].legend(fontsize=9)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_xlim(0, 50)

    plt.tight_layout()
    out_path = '/opt/projects/01_Personal/03_Study/examples/Signal_Processing/09_fir_design.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved visualization: {out_path}")
    plt.close()

    print("\n" + "=" * 65)
    print("Key Takeaways:")
    print("  - Window method: simple, fixed sidelobe shape per window type")
    print("  - Hamming: ~53 dB stopband; Kaiser: adjustable via β parameter")
    print("  - Parks-McClellan: optimal equiripple, minimum order for given spec")
    print("  - All FIR methods produce linear-phase (symmetric) filters")
    print("  - Higher stopband attenuation requires more taps (longer delay)")
    print("=" * 65)
