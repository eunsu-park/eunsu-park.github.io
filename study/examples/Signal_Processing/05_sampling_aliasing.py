#!/usr/bin/env python3
"""
Sampling, Aliasing, and Signal Reconstruction
==============================================

The Nyquist-Shannon Sampling Theorem states:

    A bandlimited signal with maximum frequency f_max can be perfectly
    reconstructed from samples taken at a rate fs > 2 * f_max.

    The minimum rate fs = 2 * f_max is called the Nyquist rate.

When fs < 2 * f_max (under-sampling), spectral copies overlap and the
signal cannot be recovered — this is called ALIASING.

Topics Covered:
    1. Sampling a continuous-time signal at multiple rates
    2. Aliasing demonstration (fs < 2*fmax violates Nyquist)
    3. Proper sampling without aliasing (fs > 2*fmax)
    4. Sinc (ideal) interpolation for perfect reconstruction
    5. Spectrum before and after sampling (showing spectral copies)
    6. Anti-aliasing (low-pass) filter applied before sampling

Author: Educational example for Signal Processing
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample


# =============================================================================
# 1. Signal and spectrum utilities
# =============================================================================

def make_continuous_signal(t):
    """
    Create a test bandlimited signal composed of two sinusoids.

    x(t) = sin(2π·3·t) + 0.5·sin(2π·7·t)

    Maximum frequency: f_max = 7 Hz
    Nyquist rate:      fs_nyquist = 14 Hz
    """
    f1, f2 = 3.0, 7.0   # Hz
    return np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)


def compute_spectrum(x, fs):
    """
    Compute one-sided magnitude spectrum using FFT.

    Args:
        x  (ndarray): Signal samples
        fs (float):   Sampling frequency (Hz)

    Returns:
        tuple: (f, mag) — positive frequency axis and magnitude
    """
    N = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    mag = (2.0 / N) * np.abs(X)   # two-sided → one-sided amplitude
    mag[0] /= 2                    # DC component is not doubled
    return f, mag


def sinc_interpolation(x_samples, t_samples, t_recon):
    """
    Reconstruct a continuous signal from samples using ideal sinc interpolation.

    Whittaker-Shannon formula:
        x_r(t) = sum_n  x[n] * sinc( fs*(t - n/fs) )
               = sum_n  x[n] * sinc( (t - t_n) * fs )

    This is exact for bandlimited signals sampled above the Nyquist rate.

    Args:
        x_samples (ndarray): Sampled signal values
        t_samples (ndarray): Sample time instants
        t_recon   (ndarray): Time points for reconstruction

    Returns:
        ndarray: Reconstructed signal at t_recon
    """
    fs = 1.0 / (t_samples[1] - t_samples[0])
    # Broadcast: (len(t_recon), len(t_samples))
    T = t_recon[:, np.newaxis] - t_samples[np.newaxis, :]
    # np.sinc(x) = sin(pi*x) / (pi*x)
    sinc_matrix = np.sinc(T * fs)
    return sinc_matrix @ x_samples


def lowpass_filter(x, fs, cutoff_hz, order=6):
    """
    Apply a Butterworth low-pass filter (anti-aliasing filter).

    Args:
        x          (ndarray): Input signal
        fs         (float):   Sampling frequency (Hz)
        cutoff_hz  (float):   Cutoff frequency (Hz)
        order      (int):     Filter order

    Returns:
        ndarray: Filtered signal
    """
    nyq = fs / 2.0
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, x)


# =============================================================================
# 2. Demonstration functions
# =============================================================================

def demo_aliasing():
    """
    Show what happens when we sample below the Nyquist rate.

    The signal has f_max = 7 Hz, so Nyquist rate = 14 Hz.
    We sample at fs = 10 Hz (< 14 Hz) and observe aliasing:
    the 7 Hz component aliases to 10-7 = 3 Hz, adding to the real 3 Hz tone.
    """
    print("=" * 60)
    print("Demo 1: Aliasing (Under-Sampling)")
    print("=" * 60)

    # "Continuous-time" reference: very high sampling rate
    fs_ref = 1000.0
    t_ref = np.arange(0, 2.0, 1.0 / fs_ref)
    x_ref = make_continuous_signal(t_ref)

    # Under-sampling: fs < 2 * f_max
    fs_low = 10.0   # Hz  (Nyquist = 14 Hz → aliased!)
    t_low = np.arange(0, 2.0, 1.0 / fs_low)
    x_low = make_continuous_signal(t_low)

    # Nyquist-rate sampling
    fs_nyq = 14.0
    t_nyq = np.arange(0, 2.0, 1.0 / fs_nyq)
    x_nyq = make_continuous_signal(t_nyq)

    # Over-sampling (safe)
    fs_high = 40.0
    t_high = np.arange(0, 2.0, 1.0 / fs_high)
    x_high = make_continuous_signal(t_high)

    # Spectra
    f_ref, S_ref = compute_spectrum(x_ref, fs_ref)
    f_low, S_low = compute_spectrum(x_low, fs_low)
    f_nyq, S_nyq = compute_spectrum(x_nyq, fs_nyq)
    f_high, S_high = compute_spectrum(x_high, fs_high)

    # Aliased frequency for 7 Hz component at fs=10:
    # alias = fs - f  = 10 - 7 = 3 Hz  (coincides with 3 Hz component!)
    print(f"  f_max = 7 Hz, Nyquist rate = 14 Hz")
    print(f"  Sampling at fs={fs_low} Hz:  7 Hz aliases to {fs_low-7.0:.0f} Hz")
    print(f"  Sampling at fs={fs_nyq} Hz: exactly at Nyquist (edge case)")
    print(f"  Sampling at fs={fs_high} Hz: safely above Nyquist — no aliasing")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Aliasing: Effect of Sampling Rate", fontsize=14, fontweight='bold')

    configs = [
        (f_ref, S_ref, fs_ref, "Reference (fs=1000 Hz)", "green"),
        (f_low, S_low, fs_low, f"Under-sampled (fs={fs_low} Hz) — ALIASED", "red"),
        (f_nyq, S_nyq, fs_nyq, f"Nyquist rate (fs={fs_nyq} Hz)", "orange"),
        (f_high, S_high, fs_high, f"Over-sampled (fs={fs_high} Hz) — safe", "blue"),
    ]

    for ax, (fi, Si, fsi, title, color) in zip(axes.flat, configs):
        ax.stem(fi, Si, linefmt=color, markerfmt=color+'o', basefmt='k-')
        ax.axvline(7.0, color='gray', linestyle='--', alpha=0.5, label='f_max=7 Hz')
        ax.axvline(3.0, color='purple', linestyle=':', alpha=0.5, label='f=3 Hz')
        if fsi <= 14.0:
            ax.axvline(fsi / 2, color='red', linestyle='--', alpha=0.4,
                       label=f'fs/2={fsi/2:.0f} Hz')
        ax.set_xlim(0, min(fsi / 2 + 1, 25))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/05_aliasing_spectra.png', dpi=150)
    print("  Saved plot to /tmp/05_aliasing_spectra.png")
    plt.show()


def demo_reconstruction():
    """
    Demonstrate perfect reconstruction via sinc interpolation when fs > 2*fmax.

    Compare:
      - Under-sampled (aliased): reconstruction fails
      - Properly sampled: reconstruction matches original perfectly
    """
    print("\n" + "=" * 60)
    print("Demo 2: Signal Reconstruction via Sinc Interpolation")
    print("=" * 60)

    # Original "continuous" signal
    t_ref = np.arange(0, 1.0, 0.001)
    x_ref = make_continuous_signal(t_ref)

    # Reconstruction time axis
    t_recon = np.arange(0.05, 0.95, 0.002)

    # --- Under-sampling (aliased) ---
    fs_bad = 10.0
    t_bad = np.arange(0, 1.0, 1.0 / fs_bad)
    x_bad = make_continuous_signal(t_bad)
    x_recon_bad = sinc_interpolation(x_bad, t_bad, t_recon)

    # --- Proper sampling ---
    fs_good = 30.0
    t_good = np.arange(0, 1.0, 1.0 / fs_good)
    x_good = make_continuous_signal(t_good)
    x_recon_good = sinc_interpolation(x_good, t_good, t_recon)

    # Reconstruction error
    x_ref_recon_pts = make_continuous_signal(t_recon)
    err_bad = np.sqrt(np.mean((x_recon_bad - x_ref_recon_pts) ** 2))
    err_good = np.sqrt(np.mean((x_recon_good - x_ref_recon_pts) ** 2))

    print(f"  Under-sampled  (fs={fs_bad} Hz): RMS reconstruction error = {err_bad:.4f}")
    print(f"  Properly sampled (fs={fs_good} Hz): RMS reconstruction error = {err_good:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Sinc Interpolation: Reconstruction Quality", fontsize=14, fontweight='bold')

    for ax, (fs, t_s, x_s, x_r, err, title_extra, color) in zip(axes, [
        (fs_bad,  t_bad,  x_bad,  x_recon_bad,  err_bad,  "ALIASED", "red"),
        (fs_good, t_good, x_good, x_recon_good, err_good, "Correct", "blue"),
    ]):
        ax.plot(t_ref, x_ref, 'k-', lw=1.5, label='Original', alpha=0.6)
        ax.stem(t_s, x_s, linefmt=color, markerfmt=color+'o',
                basefmt='none', label=f'Samples (fs={fs} Hz)')
        ax.plot(t_recon, x_r, color, lw=2, linestyle='--', label='Reconstructed')
        ax.set_xlim(0, 1)
        ax.set_ylim(-2, 2)
        ax.set_title(f"fs={fs} Hz — {title_extra}\nRMS error = {err:.4f}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/05_reconstruction.png', dpi=150)
    print("  Saved plot to /tmp/05_reconstruction.png")
    plt.show()


def demo_antialias_filter():
    """
    Show how a low-pass anti-aliasing filter prevents aliasing before sampling.

    Workflow:
        1. Original broadband signal (contains high-frequency components)
        2. Apply anti-aliasing LPF with cutoff = fs/2
        3. Sample the filtered signal — no aliasing occurs
    """
    print("\n" + "=" * 60)
    print("Demo 3: Anti-Aliasing Filter")
    print("=" * 60)

    # Broadband signal: 3 Hz + 7 Hz + 15 Hz + 22 Hz
    fs_ref = 2000.0
    t_ref = np.arange(0, 1.0, 1.0 / fs_ref)
    x_broad = (np.sin(2 * np.pi * 3 * t_ref) +
               np.sin(2 * np.pi * 7 * t_ref) +
               np.sin(2 * np.pi * 15 * t_ref) +
               0.5 * np.sin(2 * np.pi * 22 * t_ref))

    # Target sampling rate
    fs_target = 20.0   # Hz — Nyquist = 10 Hz

    # Without anti-aliasing: sample directly
    t_sampled = np.arange(0, 1.0, 1.0 / fs_target)
    x_no_aa = np.interp(t_sampled, t_ref, x_broad)

    # With anti-aliasing: apply LPF first
    cutoff = fs_target / 2.0 * 0.9   # slightly below Nyquist
    x_filtered = lowpass_filter(x_broad, fs_ref, cutoff_hz=cutoff)
    x_with_aa = np.interp(t_sampled, t_ref, x_filtered)

    # Spectra
    f_broad, S_broad = compute_spectrum(x_broad, fs_ref)
    f_filtered, S_filtered = compute_spectrum(x_filtered, fs_ref)
    f_no_aa, S_no_aa = compute_spectrum(x_no_aa, fs_target)
    f_with_aa, S_with_aa = compute_spectrum(x_with_aa, fs_target)

    print(f"  Broadband signal: 3, 7, 15, 22 Hz components")
    print(f"  Target fs = {fs_target} Hz  →  LPF cutoff = {cutoff:.1f} Hz")
    print(f"  Without AA filter: components at 15 Hz alias to {fs_target-15:.0f} Hz")
    print(f"                     components at 22 Hz alias to {fs_target-22+fs_target:.0f} Hz -> {abs(22-fs_target):.0f} Hz")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Anti-Aliasing Filter Effect", fontsize=14, fontweight='bold')

    # Original broadband spectrum
    mask_broad = f_broad <= 30
    axes[0, 0].plot(f_broad[mask_broad], S_broad[mask_broad], 'b')
    axes[0, 0].axvline(fs_target / 2, color='red', linestyle='--', label=f'fs/2={fs_target/2} Hz')
    axes[0, 0].set_title("Original Broadband Spectrum")
    axes[0, 0].set_xlabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # After LPF
    axes[0, 1].plot(f_filtered[mask_broad], S_filtered[mask_broad], 'g')
    axes[0, 1].axvline(fs_target / 2, color='red', linestyle='--', label=f'LPF cutoff≈{cutoff:.0f} Hz')
    axes[0, 1].set_title("After Anti-Aliasing LPF")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Sampled without AA — aliased
    axes[1, 0].stem(f_no_aa, S_no_aa, linefmt='r', markerfmt='ro', basefmt='k-')
    axes[1, 0].set_title(f"Sampled at fs={fs_target} Hz (NO anti-aliasing) — ALIASED")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].set_xlim(0, fs_target / 2 + 1)
    axes[1, 0].grid(True, alpha=0.3)

    # Sampled with AA — clean
    axes[1, 1].stem(f_with_aa, S_with_aa, linefmt='g', markerfmt='go', basefmt='k-')
    axes[1, 1].set_title(f"Sampled at fs={fs_target} Hz (WITH anti-aliasing) — Clean")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].set_xlim(0, fs_target / 2 + 1)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/05_antialias.png', dpi=150)
    print("  Saved plot to /tmp/05_antialias.png")
    plt.show()


if __name__ == "__main__":
    print("Sampling, Aliasing, and Signal Reconstruction")
    print("=" * 60)
    print("Test signal: x(t) = sin(2π·3·t) + 0.5·sin(2π·7·t)")
    print("  f_max = 7 Hz  →  Nyquist rate = 14 Hz\n")

    demo_aliasing()
    demo_reconstruction()
    demo_antialias_filter()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Nyquist theorem: fs > 2*f_max required for perfect reconstruction")
    print("  - Aliasing folds high frequencies back: f_alias = |f - k*fs|")
    print("  - Sinc interpolation gives exact reconstruction (bandlimited signals)")
    print("  - Anti-aliasing LPF removes components above fs/2 before sampling")
    print("  - Practical systems oversample (typically fs > 5*f_max) for safety")
    print("=" * 60)
