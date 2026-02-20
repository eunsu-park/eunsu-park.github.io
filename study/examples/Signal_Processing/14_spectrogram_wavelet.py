#!/usr/bin/env python3
"""
Time-Frequency Analysis: STFT Spectrogram and Wavelet Transform
================================================================

Classical Fourier analysis reveals *which* frequencies are present in a
signal, but not *when* they occur.  Time-frequency analysis resolves both
dimensions simultaneously.

Short-Time Fourier Transform (STFT)
-------------------------------------
The STFT divides the signal into overlapping short frames, applies a window
function to each frame, and computes the DFT:

    STFT{x}(m, k) = sum_{n} x[n] * w[n - mH] * exp(-j*2*pi*k*n/N)

where H is the hop size and N is the DFT length (window length).

Resolution trade-off (Heisenberg uncertainty principle):
    - Long window  → good frequency resolution, poor time resolution
    - Short window → good time resolution,      poor frequency resolution

Continuous Wavelet Transform (CWT)
------------------------------------
The CWT uses a scaled and translated 'mother wavelet' ψ(t):

    CWT{x}(a, b) = (1/sqrt(a)) * integral x(t) * ψ*((t-b)/a) dt

where a is the scale (~ 1/frequency) and b is the translation (time).

The CWT provides adaptive resolution:
    - High-frequency components: good time resolution (narrow ψ)
    - Low-frequency components:  good frequency resolution (wide ψ)

This is a key advantage over the fixed-resolution STFT.

Author: Educational example for Signal Processing
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# ============================================================================
# SIGNAL GENERATORS
# ============================================================================

def make_chirp(fs=1000.0, duration=2.0, f0=10.0, f1=200.0):
    """
    Generate a linear chirp: instantaneous frequency increases linearly from
    f0 to f1 over the specified duration.

    Args:
        fs       (float): Sample rate (Hz)
        duration (float): Signal length (seconds)
        f0       (float): Start frequency (Hz)
        f1       (float): End frequency (Hz)

    Returns:
        t (ndarray): Time axis
        x (ndarray): Chirp signal
    """
    t = np.arange(int(fs * duration)) / fs
    x = signal.chirp(t, f0=f0, f1=f1, t1=duration, method='linear')
    return t, x


def make_multicomponent(fs=1000.0, duration=2.0):
    """
    Construct a signal with both transient and stationary components:
        - Stationary 80 Hz tone (full duration)
        - Gaussian-modulated transient at t = 0.5 s  (burst)
        - Stationary 200 Hz tone in second half only

    This tests whether the time-frequency method can separate
    persistent tones from brief events.

    Returns:
        t (ndarray): Time axis
        x (ndarray): Composite signal
    """
    t = np.arange(int(fs * duration)) / fs

    # Persistent low-frequency tone
    tone_80 = np.sin(2 * np.pi * 80 * t)

    # Short Gaussian burst at t=0.5 s, centre frequency 300 Hz
    burst_env = np.exp(-((t - 0.5) ** 2) / (2 * 0.01 ** 2))
    burst = burst_env * np.sin(2 * np.pi * 300 * t)

    # High-frequency tone that starts at t=1 s
    tone_200 = np.where(t >= 1.0, np.sin(2 * np.pi * 200 * t), 0.0)

    rng = np.random.default_rng(0)
    noise = 0.05 * rng.standard_normal(len(t))

    return t, tone_80 + burst + tone_200 + noise


# ============================================================================
# SECTION 1: STFT SPECTROGRAM AND WINDOW LENGTH TRADE-OFF
# ============================================================================

def demo_stft_resolution_tradeoff(fs, t, x):
    """
    Compute STFT spectrograms with three different window lengths and display
    the time-frequency resolution trade-off for a chirp signal.
    """
    print("=" * 60)
    print("SECTION 1: STFT Resolution Trade-off (chirp signal)")
    print("=" * 60)

    window_lengths = [32, 128, 512]   # short, medium, long
    labels = [
        "Short window (32)\nGood time, poor freq",
        "Medium window (128)\nBalanced",
        "Long window (512)\nPoor time, good freq",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("STFT Resolution Trade-off for a Linear Chirp\n"
                 "Heisenberg uncertainty: Δt · Δf ≥ 1/(4π)",
                 fontsize=12, fontweight='bold')

    for ax, nperseg, label in zip(axes, window_lengths, labels):
        f, t_stft, Zxx = signal.stft(x, fs=fs, nperseg=nperseg,
                                      noverlap=nperseg // 2,
                                      window='hann')
        power_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
        ax.pcolormesh(t_stft, f, power_db, vmin=-80, vmax=0,
                      cmap='inferno', shading='gouraud')
        ax.set_ylim(0, fs / 2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(label)
        print(f"  Window={nperseg:4d}  "
              f"Time res ≈ {nperseg/fs*1000:.0f} ms  "
              f"Freq res ≈ {fs/nperseg:.1f} Hz")

    plt.tight_layout()
    plt.savefig("14_stft_resolution.png", dpi=120)
    print("  Saved: 14_stft_resolution.png")
    plt.show()


# ============================================================================
# SECTION 2: CWT WITH MORLET WAVELET
# ============================================================================

def morlet_wavelet(M, w0=6.0):
    """
    Return a complex Morlet wavelet of length M.

    The Morlet wavelet is a Gaussian-modulated complex sinusoid:
        ψ(t) = pi^(-1/4) * exp(j*w0*t) * exp(-t^2 / 2)

    At scale a, the peak frequency is approximately w0 / (2*pi*a).

    Args:
        M  (int)  : Wavelet length in samples.
        w0 (float): Angular frequency of the carrier (controls freq resolution).

    Returns:
        ndarray: Complex Morlet wavelet, normalised.
    """
    x = np.linspace(-w0 / 2, w0 / 2, M)
    wavelet = np.exp(1j * w0 * x) * np.exp(-0.5 * x ** 2) * np.pi ** (-0.25)
    return wavelet


def demo_cwt(fs, t, x, signal_name=""):
    """
    Compute and display the Continuous Wavelet Transform scalogram using
    scipy.signal.cwt with a Morlet wavelet.

    scipy.signal.cwt uses the *real* part of the Morlet (ricker-like) by
    default, so we directly call signal.cwt with morlet2 via a wrapper, or
    equivalently compute it manually using convolution.
    """
    print("\n" + "=" * 60)
    print(f"SECTION 2: Continuous Wavelet Transform  [{signal_name}]")
    print("=" * 60)

    # Define scales: relate scale to pseudo-frequency via f ≈ w0 / (2*pi*a/fs)
    w0 = 6.0
    # Frequencies we want to resolve (Hz)
    freqs_hz = np.logspace(np.log10(5), np.log10(fs / 2 - 1), 80)
    # Corresponding scales (in samples): a = w0 * fs / (2*pi*f)
    scales = w0 * fs / (2 * np.pi * freqs_hz)

    # CWT via scipy.signal.cwt (uses ricker/morlet based on the function passed)
    # We use scipy's built-in morlet2 wavelet
    cwt_matrix = signal.cwt(x, signal.morlet2, scales, w=w0)
    scalogram = np.abs(cwt_matrix)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    fig.suptitle(f"Continuous Wavelet Transform (Morlet) — {signal_name}",
                 fontsize=12, fontweight='bold')

    # Scalogram
    ax = axes[0]
    im = ax.pcolormesh(t, freqs_hz, scalogram, cmap='hot_r', shading='gouraud')
    ax.set_yscale('log')
    ax.set_ylabel("Pseudo-frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("CWT Scalogram (log frequency axis)\n"
                 "High freq: narrow wavelet (good time res) | "
                 "Low freq: wide wavelet (good freq res)")
    fig.colorbar(im, ax=ax, label="Magnitude")

    # Morlet wavelet at two scales (visualise what the filter looks like)
    ax2 = axes[1]
    for scale, color in [(5, 'C0'), (50, 'C1')]:
        M = int(10 * scale) | 1    # odd length
        wav = morlet_wavelet(M, w0=w0)
        t_wav = np.linspace(-M / (2 * fs), M / (2 * fs), M)
        ax2.plot(t_wav * 1000, wav.real, color=color,
                 label=f'scale={scale} (f≈{w0*fs/(2*np.pi*scale):.0f} Hz)')
        ax2.plot(t_wav * 1000, wav.imag, color=color, linestyle='--', alpha=0.5)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Morlet Wavelets at Two Scales (solid=real, dashed=imag)\n"
                  "Smaller scale → shorter wavelet → better time resolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"14_cwt_{signal_name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=120)
    print(f"  Saved: {fname}")
    plt.show()


# ============================================================================
# SECTION 3: SPECTROGRAM VS CWT — SIDE-BY-SIDE COMPARISON
# ============================================================================

def demo_stft_vs_cwt(fs, t, x, signal_name=""):
    """
    Direct visual comparison of STFT spectrogram and CWT scalogram for the
    same signal, illustrating their complementary strengths.
    """
    print("\n" + "=" * 60)
    print(f"SECTION 3: Spectrogram vs CWT — {signal_name}")
    print("=" * 60)

    # STFT (medium window)
    nperseg = 128
    f_stft, t_stft, Zxx = signal.stft(x, fs=fs, nperseg=nperseg,
                                        noverlap=nperseg * 3 // 4,
                                        window='hann')
    power_stft = 20 * np.log10(np.abs(Zxx) + 1e-12)

    # CWT
    w0 = 6.0
    freqs_hz = np.logspace(np.log10(5), np.log10(fs / 2 - 1), 80)
    scales = w0 * fs / (2 * np.pi * freqs_hz)
    cwt_matrix = signal.cwt(x, signal.morlet2, scales, w=w0)
    scalogram = np.abs(cwt_matrix)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    fig.suptitle(f"STFT vs CWT for '{signal_name}'\n"
                 "STFT: fixed resolution | CWT: adaptive resolution",
                 fontsize=12, fontweight='bold')

    # Raw signal
    axes[0].plot(t, x, color='C0', linewidth=0.8)
    axes[0].set_title("(a) Input Signal")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    # STFT spectrogram
    axes[1].pcolormesh(t_stft, f_stft, power_stft, vmin=-80, vmax=0,
                       cmap='inferno', shading='gouraud')
    axes[1].set_title(f"(b) STFT Spectrogram (window={nperseg} samples, "
                      f"Δt≈{nperseg/fs*1000:.0f} ms, Δf≈{fs/nperseg:.1f} Hz)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")

    # CWT scalogram
    im = axes[2].pcolormesh(t, freqs_hz, scalogram, cmap='hot_r', shading='gouraud')
    axes[2].set_yscale('log')
    axes[2].set_title("(c) CWT Scalogram (Morlet, log frequency)\n"
                      "High-freq transients resolve sharply in time; "
                      "low-freq tones resolve sharply in frequency")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Pseudo-frequency (Hz)")
    fig.colorbar(im, ax=axes[2], label="Magnitude")

    plt.tight_layout()
    fname = f"14_stft_vs_cwt_{signal_name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=120)
    print(f"  Saved: {fname}")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Time-Frequency Analysis: STFT Spectrogram and Wavelet Transform")
    print("=" * 60)

    fs = 1000.0   # sample rate

    # --- Chirp signal ---------------------------------------------------------
    t_chirp, x_chirp = make_chirp(fs=fs, duration=2.0, f0=10.0, f1=200.0)
    print(f"\nChirp signal:  {len(t_chirp)} samples, "
          f"f: 10→200 Hz over {len(t_chirp)/fs:.1f} s")

    demo_stft_resolution_tradeoff(fs, t_chirp, x_chirp)
    demo_cwt(fs, t_chirp, x_chirp, signal_name="chirp")
    demo_stft_vs_cwt(fs, t_chirp, x_chirp, signal_name="chirp")

    # --- Multi-component signal -----------------------------------------------
    t_mc, x_mc = make_multicomponent(fs=fs, duration=2.0)
    print(f"\nMulti-component signal: {len(t_mc)} samples")
    print("  Components: 80 Hz tone (full) + 300 Hz burst at t=0.5 s "
          "+ 200 Hz tone (t≥1 s)")

    demo_stft_vs_cwt(fs, t_mc, x_mc, signal_name="multicomponent")

    print("\nDone.  PNG files saved.")
    print("\nKey takeaways:")
    print("  - STFT: fixed Δt·Δf product; choose window for your application")
    print("  - CWT : adaptive resolution; better for multi-scale signals")
    print("  - Heisenberg uncertainty: you cannot have perfect Δt AND Δf")
