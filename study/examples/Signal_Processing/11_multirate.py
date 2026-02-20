#!/usr/bin/env python3
"""
Multirate Signal Processing
============================

This script demonstrates the core concepts of multirate DSP:

1. Downsampling (decimation) — with and without anti-aliasing filter
2. Upsampling (interpolation) — with and without interpolation filter
3. Aliasing visualization — what happens when Nyquist theorem is violated
4. Rational rate conversion — resampling from 44100 Hz to 48000 Hz (L/M = 160/147)
5. Polyphase view — why decimation and interpolation filters work

Key Concepts:
    - Decimation by M : keep every M-th sample → Nyquist halves → aliasing possible
    - Interpolation by L : insert L-1 zeros, then lowpass → smooth the zeros
    - Rational resampling: upsample by L, lowpass, then downsample by M
    - Anti-aliasing filter cutoff must be ≤ π/M (normalized) before decimation
    - Interpolation filter cutoff must be ≤ π/L and gain = L after upsampling

Practical note:
    scipy.signal.decimate  = lowpass filter + downsample (in one call)
    scipy.signal.resample  = FFT-based resampling (handles arbitrary ratios)
    scipy.signal.resample_poly = polyphase rational resampling (exact L/M)

Author: Educational example for Signal Processing
License: MIT
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# ============================================================================
# HELPERS
# ============================================================================

def spectrum_db(x, fs, nfft=2048):
    """
    Compute one-sided power spectrum in dBFS.

    Args:
        x   (ndarray): Input signal
        fs  (float)  : Sampling rate (Hz)
        nfft (int)   : FFT size

    Returns:
        (freqs, mag_db): frequency axis (Hz) and magnitude (dBFS)
    """
    X = np.fft.rfft(x, n=nfft)
    mag = np.abs(X) / nfft
    mag_db = 20 * np.log10(np.maximum(mag, 1e-15))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freqs, mag_db


def make_test_signal(fs, duration=0.5, freqs_hz=None, amps=None, seed=0):
    """
    Construct a multi-tone signal.

    Args:
        fs       (float): Sampling rate
        duration (float): Signal duration in seconds
        freqs_hz (list) : Sinusoid frequencies in Hz
        amps     (list) : Corresponding amplitudes

    Returns:
        (t, x): time vector and signal
    """
    if freqs_hz is None:
        freqs_hz = [1000, 5000, 9000]
    if amps is None:
        amps = [1.0, 0.8, 0.6]

    t = np.arange(0, duration, 1.0 / fs)
    x = np.zeros(len(t))
    for f, a in zip(freqs_hz, amps):
        x += a * np.sin(2 * np.pi * f * t)
    return t, x


# ============================================================================
# DEMO 1: DECIMATION (WITH AND WITHOUT ANTI-ALIASING)
# ============================================================================

def demo_decimation():
    """
    Decimate a signal by factor M = 4, comparing:
      (a) naive downsampling — no anti-aliasing filter → aliasing
      (b) scipy.signal.decimate — built-in anti-aliasing Chebyshev lowpass
    """
    print("=" * 65)
    print("1. DECIMATION BY M=4  (44100 → 11025 Hz)")
    print("=" * 65)

    fs = 44100
    M = 4                          # decimation factor
    fs_out = fs // M               # 11025 Hz
    nyquist_out = fs_out / 2       # new Nyquist = 5512.5 Hz

    # Test signal: three tones at 1 kHz (safe), 4 kHz (safe), 8 kHz (alias!)
    # 8 kHz > nyquist_out → will alias to 8000 − 11025 = −3025 → 3025 Hz
    freqs = [1000, 4000, 8000]
    t, x = make_test_signal(fs, duration=0.2, freqs_hz=freqs, amps=[1.0, 0.8, 0.6])

    print(f"  Input: fs={fs} Hz,  {freqs} Hz tones")
    print(f"  Decimation factor M={M}  →  output fs={fs_out} Hz")
    print(f"  New Nyquist = {nyquist_out} Hz  → tones above {nyquist_out} Hz WILL alias")

    # (a) Naive downsampling — select every M-th sample, NO filtering
    x_naive = x[::M]
    t_naive = t[::M]

    alias_freq = abs(freqs[-1] - fs_out)   # predicted aliased frequency
    print(f"\n  (a) Naive downsampling: {freqs[-1]} Hz aliases to {alias_freq} Hz")

    # (b) scipy.signal.decimate — applies Chebyshev I lowpass, then downsample
    #     Default: order-8 Chebyshev I with cutoff at fs_out/2
    x_decimated = signal.decimate(x, M, ftype='fir', zero_phase=True)
    t_decimated = np.arange(len(x_decimated)) / fs_out

    print(f"  (b) decimate(): anti-aliasing FIR lowpass → clean output")

    # Power at the alias frequency after each method
    for label, sig, sr in [('Naive', x_naive, fs_out),
                             ('Decimated', x_decimated, fs_out)]:
        _, mag_db = spectrum_db(sig, sr, nfft=4096)
        freqs_axis = np.fft.rfftfreq(4096, d=1.0 / sr)
        idx = np.argmin(np.abs(freqs_axis - alias_freq))
        print(f"  {label:<12}: power at alias ({alias_freq} Hz) = "
              f"{mag_db[idx]:.1f} dBFS")

    return fs, fs_out, M, t, x, t_naive, x_naive, t_decimated, x_decimated, freqs


# ============================================================================
# DEMO 2: UPSAMPLING / INTERPOLATION
# ============================================================================

def demo_interpolation():
    """
    Upsample a signal by factor L = 4, comparing:
      (a) zero insertion only — no interpolation filter → imaging
      (b) scipy.signal.resample_poly with L, 1 — polyphase interpolation
    """
    print("\n" + "=" * 65)
    print("2. INTERPOLATION BY L=4  (11025 → 44100 Hz)")
    print("=" * 65)

    fs_in = 11025
    L = 4
    fs_out = fs_in * L    # 44100 Hz

    freqs = [1000, 4000]
    t_in, x_in = make_test_signal(fs_in, duration=0.2, freqs_hz=freqs)

    print(f"  Input: fs={fs_in} Hz,  {freqs} Hz tones")
    print(f"  Interpolation factor L={L}  →  output fs={fs_out} Hz")

    # (a) Zero insertion only: insert L-1 zeros between every sample
    x_upsampled = np.zeros(len(x_in) * L)
    x_upsampled[::L] = x_in   # place original samples at every L-th position
    t_up = np.arange(len(x_upsampled)) / fs_out

    print(f"\n  (a) Zero insertion: creates images at multiples of {fs_in} Hz")

    # (b) Polyphase interpolation via resample_poly
    x_interp = signal.resample_poly(x_in, L, 1)   # upsample by L, downsample by 1
    t_interp = np.arange(len(x_interp)) / fs_out

    print(f"  (b) resample_poly(L=4, M=1): built-in polyphase filter removes images")

    # Check power in image band (around 11025 − 1000 = 10025 Hz)
    image_freq = fs_in - freqs[0]
    for label, sig, sr in [('Zero-insert', x_upsampled, fs_out),
                             ('resample_poly', x_interp, fs_out)]:
        _, mag_db = spectrum_db(sig, sr, nfft=8192)
        freqs_axis = np.fft.rfftfreq(8192, d=1.0 / sr)
        idx = np.argmin(np.abs(freqs_axis - image_freq))
        print(f"  {label:<16}: image power at {image_freq} Hz = {mag_db[idx]:.1f} dBFS")

    return fs_in, fs_out, L, t_in, x_in, t_up, x_upsampled, t_interp, x_interp


# ============================================================================
# DEMO 3: RATIONAL RATE CONVERSION  44100 → 48000 Hz
# ============================================================================

def demo_rate_conversion():
    """
    Convert from CD rate (44100 Hz) to professional audio (48000 Hz).

    Rational ratio: 48000/44100 = 160/147  (L=160, M=147)
    Process: upsample by 160, lowpass filter, downsample by 147.
    scipy.signal.resample_poly handles this efficiently using a polyphase filter.
    """
    print("\n" + "=" * 65)
    print("3. RATIONAL RATE CONVERSION  44100 Hz → 48000 Hz  (L/M = 160/147)")
    print("=" * 65)

    fs_in = 44100
    fs_out = 48000
    L, M = 160, 147    # exact rational ratio

    duration = 0.05   # short segment — polyphase is memory-intensive for large L
    freqs = [440, 4000, 10000]   # A4, ~mid, ~high
    t_in, x_in = make_test_signal(fs_in, duration=duration, freqs_hz=freqs)
    t_out_expected = np.arange(int(duration * fs_out)) / fs_out

    print(f"  Input: {fs_in} Hz, {len(x_in)} samples")
    print(f"  Ratio: {fs_out}/{fs_in} = {L}/{M}")

    # scipy resample_poly: efficient polyphase implementation
    x_resampled = signal.resample_poly(x_in, L, M)
    print(f"  Output (resample_poly): {len(x_resampled)} samples  "
          f"(expected ≈ {int(round(len(x_in) * L / M))})")

    # FFT-based resample (scipy.signal.resample) — alternative
    n_out = int(round(len(x_in) * fs_out / fs_in))
    x_fft_rs = signal.resample(x_in, n_out)
    print(f"  Output (FFT resample) : {len(x_fft_rs)} samples")

    # Verify spectral content is preserved
    print(f"\n  Verifying tonal content ({freqs} Hz) is preserved:")
    _, mag_in_db = spectrum_db(x_in, fs_in, nfft=8192)
    freqs_in = np.fft.rfftfreq(8192, d=1.0 / fs_in)
    _, mag_out_db = spectrum_db(x_resampled, fs_out, nfft=8192)
    freqs_out = np.fft.rfftfreq(8192, d=1.0 / fs_out)

    print(f"  {'Freq':>6}  {'In (dBFS)':>12}  {'Out (dBFS)':>12}  {'Diff':>8}")
    for f in freqs:
        idx_in = np.argmin(np.abs(freqs_in - f))
        idx_out = np.argmin(np.abs(freqs_out - f))
        diff = mag_out_db[idx_out] - mag_in_db[idx_in]
        print(f"  {f:>6} Hz  {mag_in_db[idx_in]:>10.1f}  "
              f"{mag_out_db[idx_out]:>10.1f}  {diff:>+7.2f} dB")

    return (fs_in, fs_out, t_in, x_in, x_resampled, x_fft_rs,
            freqs_in, mag_in_db, freqs_out, mag_out_db)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("MULTIRATE SIGNAL PROCESSING")

    # Run demos and collect results
    (fs, fs_dec, M, t, x, t_naive, x_naive,
     t_dec, x_dec, sig_freqs) = demo_decimation()

    (fs_in_up, fs_up, L, t_in_up, x_in_up,
     t_zins, x_zins, t_interp, x_interp) = demo_interpolation()

    (fs_conv_in, fs_conv_out, t_conv_in, x_conv_in,
     x_conv_poly, x_conv_fft,
     freqs_in, mag_in_db, freqs_out, mag_out_db) = demo_rate_conversion()

    # -----------------------------------------------------------------------
    # VISUALIZATION — 3×2 grid
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle('Multirate Signal Processing', fontsize=13, fontweight='bold')

    # (0,0) Decimation: spectra comparison
    ax = axes[0, 0]
    nfft_dec = 4096
    f_orig, m_orig = spectrum_db(x, fs, nfft=nfft_dec)
    f_naive, m_naive = spectrum_db(x_naive, fs_dec, nfft=nfft_dec)
    f_dec, m_dec = spectrum_db(x_dec, fs_dec, nfft=nfft_dec)

    ax.plot(f_orig, m_orig, 'gray', lw=1.2, alpha=0.7, label=f'Input ({fs} Hz)')
    ax.plot(f_naive, m_naive, 'r-', lw=1.8, label='Naive downsample (aliased)')
    ax.plot(f_dec, m_dec, 'b--', lw=1.8, label='decimate() anti-aliased')
    ax.axvline(fs_dec / 2, color='k', ls=':', lw=1, label=f'New Nyquist ({fs_dec//2} Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dBFS)')
    ax.set_title(f'Decimation by M={M}: aliasing vs. anti-aliasing')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, fs_dec / 2)
    ax.set_ylim(-80, 10)

    # (0,1) Decimation: time domain (first 5 ms)
    ax = axes[0, 1]
    n_show = int(0.005 * fs)
    ax.plot(t[:n_show] * 1e3, x[:n_show], 'gray', lw=1, alpha=0.5, label='Input')
    n_show_d = int(0.005 * fs_dec)
    ax.plot(t_naive[:n_show_d] * 1e3, x_naive[:n_show_d],
            'r-o', ms=4, lw=1.5, label='Naive (aliased)')
    ax.plot(t_dec[:n_show_d] * 1e3, x_dec[:n_show_d],
            'b--s', ms=4, lw=1.5, label='decimate() clean')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Decimation: time domain (first 5 ms)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) Interpolation: spectra comparison
    ax = axes[1, 0]
    nfft_up = 8192
    f_in_up, m_in_up = spectrum_db(x_in_up, fs_in_up, nfft=nfft_up)
    f_zins, m_zins = spectrum_db(x_zins, fs_up, nfft=nfft_up)
    f_interp, m_interp = spectrum_db(x_interp, fs_up, nfft=nfft_up)

    ax.plot(f_in_up, m_in_up, 'gray', lw=1.2, alpha=0.7,
            label=f'Input ({fs_in_up} Hz)')
    ax.plot(f_zins, m_zins, 'r-', lw=1.5, label='Zero-insert (images visible)')
    ax.plot(f_interp, m_interp, 'b--', lw=1.8, label='resample_poly (filtered)')
    ax.axvline(fs_in_up / 2, color='k', ls=':', lw=1,
               label=f'Old Nyquist ({fs_in_up//2} Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dBFS)')
    ax.set_title(f'Interpolation by L={L}: imaging vs. polyphase')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, fs_up / 2)
    ax.set_ylim(-80, 10)

    # (1,1) Interpolation: time domain (first 5 ms)
    ax = axes[1, 1]
    n_show_in = int(0.005 * fs_in_up)
    ax.plot(t_in_up[:n_show_in] * 1e3, x_in_up[:n_show_in],
            'ko', ms=5, label=f'Input samples ({fs_in_up} Hz)')
    n_show_up = int(0.005 * fs_up)
    ax.plot(np.arange(n_show_up) / fs_up * 1e3, x_zins[:n_show_up],
            'r-', lw=1.2, alpha=0.7, label='Zero-insert (jagged)')
    ax.plot(np.arange(len(x_interp[:n_show_up])) / fs_up * 1e3,
            x_interp[:n_show_up], 'b--', lw=1.8, label='Interpolated (smooth)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Interpolation: time domain (first 5 ms)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2,0) Rate conversion: input vs output spectra
    ax = axes[2, 0]
    ax.plot(freqs_in, mag_in_db, 'gray', lw=1.5, alpha=0.8,
            label=f'Input ({fs_conv_in} Hz)')
    ax.plot(freqs_out, mag_out_db, 'b--', lw=1.8,
            label=f'resample_poly ({fs_conv_out} Hz)')
    _, mag_fft_db = spectrum_db(x_conv_fft, fs_conv_out, nfft=8192)
    freqs_fft = np.fft.rfftfreq(8192, d=1.0 / fs_conv_out)
    ax.plot(freqs_fft, mag_fft_db, 'r:', lw=1.5,
            label=f'FFT resample ({fs_conv_out} Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dBFS)')
    ax.set_title(f'Rate Conversion {fs_conv_in}→{fs_conv_out} Hz (L/M=160/147)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 22050)
    ax.set_ylim(-80, 10)

    # (2,1) Rate conversion: time domain overlay (first 5 ms)
    ax = axes[2, 1]
    n_in_show = int(0.005 * fs_conv_in)
    n_out_show = int(0.005 * fs_conv_out)
    t_in_axis = np.arange(n_in_show) / fs_conv_in * 1e3
    t_out_axis = np.arange(n_out_show) / fs_conv_out * 1e3
    ax.plot(t_in_axis, x_conv_in[:n_in_show], 'ko', ms=3, alpha=0.6,
            label=f'Input ({fs_conv_in} Hz)')
    ax.plot(t_out_axis, x_conv_poly[:n_out_show], 'b-', lw=1.8,
            label=f'resample_poly ({fs_conv_out} Hz)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Rate Conversion: time domain (first 5 ms)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = '/opt/projects/01_Personal/03_Study/examples/Signal_Processing/11_multirate.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved visualization: {out_path}")
    plt.close()

    print("\n" + "=" * 65)
    print("Key Takeaways:")
    print("  - Decimation without anti-aliasing causes aliasing (folding)")
    print("  - Anti-aliasing cutoff must be ≤ fs_out/2 BEFORE downsampling")
    print("  - Upsampling without lowpass causes spectral images (copies)")
    print("  - Interpolation filter gain = L to restore original amplitude")
    print("  - Rational resampling: upsample L, lowpass at min(π/L, π/M), downsample M")
    print("  - resample_poly is memory-efficient; resample uses FFT (exact length)")
    print("=" * 65)
