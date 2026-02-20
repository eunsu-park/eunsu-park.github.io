#!/usr/bin/env python3
"""
IIR Filter Design Methods
==========================

This script demonstrates the four classical IIR lowpass filter designs:

1. Butterworth  — maximally flat magnitude (no ripple in either band)
2. Chebyshev I  — equiripple in passband, monotone in stopband
3. Chebyshev II — monotone in passband, equiripple in stopband
4. Elliptic     — equiripple in both bands (minimum order for given spec)

Key Concepts:
    - Analog prototype → bilinear transform → digital IIR filter
    - Filter order N controls transition band sharpness
    - Passband ripple (Rp) and stopband attenuation (Rs) are design knobs
    - Pole-zero plot reveals where the filter's gain is amplified (poles)
      and nulled (zeros, for Chebyshev II and Elliptic)

Design Trade-offs:
    - Butterworth : Flattest passband, widest transition band, all-pole
    - Chebyshev I : Sharper roll-off than Butterworth (same order),
                    ripple in passband
    - Chebyshev II: Sharper roll-off, ripple in stopband,
                    finite zeros on unit circle
    - Elliptic    : Sharpest roll-off, ripple in both bands,
                    most sensitive to coefficient quantization

Author: Educational example for Signal Processing
License: MIT
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# ============================================================================
# DESIGN PARAMETERS
# ============================================================================

FS = 1000.0          # Sampling frequency (Hz)
FC = 150.0           # Passband cutoff (Hz)  — 3 dB point for Butterworth
RP = 1.0             # Passband ripple (dB)  — used by Cheby I and Elliptic
RS = 40.0            # Stopband attenuation (dB) — Cheby II, Elliptic
ORDER = 5            # Filter order (common across all types for fair comparison)
FC_NORM = FC / (FS / 2)   # Normalized cutoff (0–1, where 1 = Nyquist)


# ============================================================================
# DESIGN FUNCTIONS
# ============================================================================

def design_all_filters(order, fc_norm, rp_db, rs_db):
    """
    Design four classical IIR lowpass filters of the same order.

    All filters use the bilinear transform (analog=False) so that the
    design frequency fc_norm maps directly to the normalized digital
    frequency axis (0–1 where 1 = Nyquist).

    Args:
        order   (int)  : Filter order N
        fc_norm (float): Normalized cutoff (0–1)
        rp_db   (float): Passband ripple in dB (Cheby I, Elliptic)
        rs_db   (float): Stopband attenuation in dB (Cheby II, Elliptic)

    Returns:
        dict: {name: (b, a)} second-order sections not used here for clarity
    """
    filters = {}

    # Butterworth: maximally flat, -3 dB exactly at fc_norm
    b, a = signal.butter(order, fc_norm, btype='low', analog=False)
    filters['Butterworth'] = (b, a)

    # Chebyshev Type I: equiripple in passband, monotone stopband
    b, a = signal.cheby1(order, rp_db, fc_norm, btype='low', analog=False)
    filters['Chebyshev I'] = (b, a)

    # Chebyshev Type II: monotone passband, equiripple stopband
    # fc_norm here marks the beginning of the stopband (Rs attenuation)
    b, a = signal.cheby2(order, rs_db, fc_norm, btype='low', analog=False)
    filters['Chebyshev II'] = (b, a)

    # Elliptic (Cauer): equiripple in both bands — minimum order
    b, a = signal.ellip(order, rp_db, rs_db, fc_norm, btype='low', analog=False)
    filters['Elliptic'] = (b, a)

    return filters


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_responses(filters, worN=2048):
    """
    Compute frequency responses for all filters.

    Returns:
        dict: {name: (w, H)} where w is normalized freq (0–π) and H is complex
    """
    responses = {}
    for name, (b, a) in filters.items():
        w, H = signal.freqz(b, a, worN=worN)
        responses[name] = (w, H)
    return responses


def print_filter_stats(filters, fc_norm, rp_db, rs_db):
    """Print key numerical metrics for each filter."""
    print("=" * 70)
    print(f"IIR LOWPASS FILTER COMPARISON  (order={ORDER}, fc={FC:.0f} Hz, fs={FS:.0f} Hz)")
    print("=" * 70)
    print(f"{'Filter':<16} {'Taps':>5} {'3 dB point':>13} {'PB ripple':>12} {'SB atten':>10}")
    print("-" * 70)

    for name, (b, a) in filters.items():
        w, H = signal.freqz(b, a, worN=4096)
        mag = np.abs(H)
        mag_db = 20 * np.log10(np.maximum(mag, 1e-15))

        # -3 dB crossing
        idx_3dB = np.argmin(np.abs(mag - (1.0 / np.sqrt(2))))
        freq_3dB = w[idx_3dB] / np.pi   # normalized (fraction of Nyquist)

        # Passband ripple: max deviation from 0 dB up to fc_norm
        idx_pb = np.searchsorted(w / np.pi, fc_norm)
        pb_ripple = np.max(mag_db[:idx_pb + 1]) - np.min(mag_db[1:idx_pb + 1])

        # Stopband attenuation: max gain past fc_norm + 20 % transition
        idx_sb = np.searchsorted(w / np.pi, min(fc_norm * 1.4, 0.99))
        sb_atten = -np.max(mag_db[idx_sb:])

        taps = len(b) + len(a) - 1   # numerator + denominator - shared gain
        print(f"{name:<16} {taps:>5}  {freq_3dB:>8.3f}×Nyq  "
              f"{pb_ripple:>10.2f} dB  {sb_atten:>8.1f} dB")

    print()
    print(f"Design specs: Rp={rp_db} dB passband ripple, Rs={rs_db} dB stopband atten.")
    print(f"Note: Chebyshev II fc marks start of stopband, not -3 dB point.")


def minimum_order_comparison(fc_norm, rp_db, rs_db):
    """
    Show what order each type needs to meet the same specification.

    Specification: Rp dB ripple in passband up to fc_norm,
                   Rs dB attenuation in stopband from 1.3×fc_norm upward.
    """
    print("\n" + "=" * 70)
    print("MINIMUM ORDER TO MEET SPEC  (Rp={} dB, Rs={} dB, fc={} Hz)".format(
        rp_db, rs_db, FC))
    print("Stopband edge = 1.3 × passband edge")
    print("=" * 70)

    fs_norm = min(fc_norm * 1.3, 0.99)   # stopband edge (normalized)

    types = ['butter', 'cheby1', 'cheby2', 'ellip']
    labels = ['Butterworth', 'Chebyshev I', 'Chebyshev II', 'Elliptic']

    print(f"{'Filter':<16} {'Min order':>10}")
    print("-" * 30)
    for ftype, label in zip(types, labels):
        try:
            N, _ = signal.buttord(fc_norm, fs_norm, rp_db, rs_db)  # not used
        except Exception:
            N = '—'

        if ftype == 'butter':
            N, Wn = signal.buttord(fc_norm, fs_norm, rp_db, rs_db)
        elif ftype == 'cheby1':
            N, Wn = signal.cheb1ord(fc_norm, fs_norm, rp_db, rs_db)
        elif ftype == 'cheby2':
            N, Wn = signal.cheb2ord(fc_norm, fs_norm, rp_db, rs_db)
        else:
            N, Wn = signal.ellipord(fc_norm, fs_norm, rp_db, rs_db)

        print(f"{label:<16} {N:>10}")

    print("\n  → Elliptic always needs the fewest taps for any given spec.")
    print("  → Butterworth needs the most (but has the flattest passband).")


# ============================================================================
# SIGNAL APPLICATION
# ============================================================================

def apply_filters_to_signal(filters):
    """
    Apply all four filters to a noisy test signal.

    Signal: 50 Hz tone (in passband) + 300 Hz tone (in stopband)
            + broadband Gaussian noise
    """
    print("\n" + "=" * 70)
    print("FILTERING A NOISY TEST SIGNAL")
    print("=" * 70)

    rng = np.random.default_rng(42)
    t = np.linspace(0, 0.5, int(FS * 0.5), endpoint=False)
    f_pass = 50.0    # Hz — inside passband
    f_stop = 300.0   # Hz — inside stopband

    # Construct noisy signal
    clean = np.sin(2 * np.pi * f_pass * t)
    interference = 0.8 * np.sin(2 * np.pi * f_stop * t)
    noise = 0.3 * rng.standard_normal(len(t))
    x = clean + interference + noise

    print(f"  Signal : {f_pass} Hz (passband) + {f_stop} Hz (stopband) + noise")
    print(f"  Cutoff : {FC} Hz   (filters should remove {f_stop} Hz)")

    outputs = {}
    for name, (b, a) in filters.items():
        # zero-phase filtering removes group-delay distortion for comparison
        y = signal.filtfilt(b, a, x)
        outputs[name] = y

        # Power of residual at stopband frequency
        N = len(y)
        Y = np.fft.rfft(y) / N
        freqs = np.fft.rfftfreq(N, d=1.0 / FS)
        idx = np.argmin(np.abs(freqs - f_stop))
        stop_power_db = 20 * np.log10(max(np.abs(Y[idx]), 1e-15))
        print(f"  {name:<16}: residual at {f_stop:.0f} Hz = {stop_power_db:.1f} dBFS")

    return t, x, outputs


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("IIR FILTER DESIGN METHODS")

    # Design all four filters with the same order
    filters = design_all_filters(ORDER, FC_NORM, RP, RS)
    responses = compute_responses(filters)

    # Print statistics
    print_filter_stats(filters, FC_NORM, RP, RS)
    minimum_order_comparison(FC_NORM, RP, RS)

    # Apply to a test signal
    t, x, outputs = apply_filters_to_signal(filters)

    # -----------------------------------------------------------------------
    # VISUALIZATION — 3×2 grid
    # -----------------------------------------------------------------------
    colors = {'Butterworth': 'tab:blue',
              'Chebyshev I': 'tab:orange',
              'Chebyshev II': 'tab:green',
              'Elliptic': 'tab:red'}
    styles = {'Butterworth': '-',
              'Chebyshev I': '--',
              'Chebyshev II': '-.',
              'Elliptic': ':'}

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle(f'IIR Filter Design Comparison  (N={ORDER}, fc={FC} Hz, fs={FS} Hz)',
                 fontsize=13, fontweight='bold')

    # (0,0) Magnitude response (linear)
    ax = axes[0, 0]
    for name, (w, H) in responses.items():
        ax.plot(w / np.pi * (FS / 2), np.abs(H),
                color=colors[name], ls=styles[name], lw=2, label=name)
    ax.axvline(FC, color='k', ls=':', lw=1, label=f'fc={FC} Hz')
    ax.axhline(1.0 / np.sqrt(2), color='gray', ls='--', lw=1, alpha=0.6, label='-3 dB')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|H(f)|')
    ax.set_title('Magnitude Response (linear)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, FS / 2)

    # (0,1) Magnitude response (dB)
    ax = axes[0, 1]
    for name, (w, H) in responses.items():
        mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-15))
        ax.plot(w / np.pi * (FS / 2), mag_db,
                color=colors[name], ls=styles[name], lw=2, label=name)
    ax.axvline(FC, color='k', ls=':', lw=1)
    ax.axhline(-RP, color='gray', ls='--', lw=1, alpha=0.6, label=f'-{RP} dB (Rp)')
    ax.axhline(-RS, color='gray', ls='-.', lw=1, alpha=0.6, label=f'-{RS} dB (Rs)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Magnitude Response (dB)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, FS / 2)
    ax.set_ylim(-60, 5)

    # (1,0) Phase response
    ax = axes[1, 0]
    for name, (w, H) in responses.items():
        phase = np.unwrap(np.angle(H))
        ax.plot(w / np.pi * (FS / 2), np.degrees(phase),
                color=colors[name], ls=styles[name], lw=2, label=name)
    ax.axvline(FC, color='k', ls=':', lw=1)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase (degrees)')
    ax.set_title('Phase Response (unwrapped)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, FS / 2)

    # (1,1) Pole-zero plots (2×2 sub-grid within this subplot area)
    # Use inset axes for each filter's pole-zero diagram
    ax = axes[1, 1]
    ax.set_visible(False)   # hide the parent; draw four insets manually

    filter_list = list(filters.items())
    inset_positions = [(0.01, 0.51, 0.48, 0.46),   # Butterworth (top-left)
                       (0.51, 0.51, 0.48, 0.46),   # Cheby I (top-right)
                       (0.01, 0.01, 0.48, 0.46),   # Cheby II (bottom-left)
                       (0.51, 0.01, 0.48, 0.46)]   # Elliptic (bottom-right)

    for (name, (b, a)), pos in zip(filter_list, inset_positions):
        ax_in = fig.add_axes([
            axes[1, 1].get_position().x0 + pos[0] * axes[1, 1].get_position().width,
            axes[1, 1].get_position().y0 + pos[1] * axes[1, 1].get_position().height,
            pos[2] * axes[1, 1].get_position().width,
            pos[3] * axes[1, 1].get_position().height,
        ])
        zeros, poles, _ = signal.tf2zpk(b, a)
        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 300)
        ax_in.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.8, alpha=0.4)
        ax_in.plot(np.real(poles), np.imag(poles), 'rx', ms=7, mew=2, label='Poles')
        ax_in.plot(np.real(zeros), np.imag(zeros), 'bo', ms=5, mfc='none', mew=1.5,
                   label='Zeros')
        ax_in.axhline(0, color='gray', lw=0.5)
        ax_in.axvline(0, color='gray', lw=0.5)
        ax_in.set_title(name, fontsize=8)
        ax_in.set_aspect('equal')
        ax_in.set_xlim(-1.6, 1.6)
        ax_in.set_ylim(-1.6, 1.6)
        ax_in.tick_params(labelsize=6)
        ax_in.grid(True, alpha=0.2)
        if name == 'Butterworth':
            ax_in.legend(fontsize=6, loc='lower right')

    # Add a title for the pole-zero panel area
    fig.text(
        axes[1, 1].get_position().x0 + 0.5 * axes[1, 1].get_position().width,
        axes[1, 1].get_position().y1 + 0.005,
        'Pole-Zero Diagrams (unit circle shown)',
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )

    # (2,0) Filtered signal — time domain (first 100 ms)
    ax = axes[2, 0]
    idx_end = int(0.1 * FS)
    ax.plot(t[:idx_end] * 1e3, x[:idx_end], color='gray', lw=0.8,
            alpha=0.5, label='Noisy input')
    for name, y in outputs.items():
        ax.plot(t[:idx_end] * 1e3, y[:idx_end],
                color=colors[name], ls=styles[name], lw=1.8, label=name)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Filtered Signal (first 100 ms, zero-phase)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2,1) Spectrum of filtered outputs
    ax = axes[2, 1]
    N_sig = len(x)
    freqs = np.fft.rfftfreq(N_sig, d=1.0 / FS)
    X_mag = np.abs(np.fft.rfft(x)) / N_sig
    ax.plot(freqs, 20 * np.log10(np.maximum(X_mag, 1e-15)),
            color='gray', lw=1.0, alpha=0.5, label='Input')
    for name, y in outputs.items():
        Y_mag = np.abs(np.fft.rfft(y)) / N_sig
        ax.plot(freqs, 20 * np.log10(np.maximum(Y_mag, 1e-15)),
                color=colors[name], ls=styles[name], lw=1.8, label=name)
    ax.axvline(FC, color='k', ls=':', lw=1)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dBFS)')
    ax.set_title('Output Spectrum')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, FS / 2)
    ax.set_ylim(-80, 10)

    plt.tight_layout()
    out_path = '/opt/projects/01_Personal/03_Study/examples/Signal_Processing/10_iir_design.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved visualization: {out_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("  - Butterworth: flattest passband, but needs highest order for sharp roll-off")
    print("  - Chebyshev I: sharper than Butterworth, at cost of passband ripple")
    print("  - Chebyshev II: all ripple pushed into stopband, flat passband")
    print("  - Elliptic: sharpest roll-off for given order, ripple in both bands")
    print("  - All IIR filters have nonlinear phase — use filtfilt for zero-phase")
    print("  - Poles inside unit circle → stable; zeros on unit circle → deep nulls")
    print("=" * 70)
