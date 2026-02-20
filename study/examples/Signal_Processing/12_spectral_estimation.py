#!/usr/bin/env python3
"""
Spectral Estimation Methods
=============================

This script compares non-parametric and parametric spectral estimation
techniques applied to a signal with known spectral content:

Non-parametric methods:
    1. Periodogram        — squared magnitude of FFT; high variance, good resolution
    2. Bartlett's method  — average of non-overlapping periodograms; reduces variance
    3. Welch's method     — overlapping windowed segments; variance vs. resolution tradeoff

Parametric methods:
    4. AR model (Yule-Walker) — models signal as output of all-pole filter driven by
                                white noise; high resolution for short data records

Key Concepts:
    - Variance and resolution are fundamentally traded off in spectral estimation
    - Averaging reduces variance by factor K (segments), but reduces resolution
      by same factor (segment length N/K vs full N)
    - Windowing reduces spectral leakage at cost of main-lobe width
    - AR model can resolve closely-spaced peaks that DFT-based methods blur
    - Yule-Walker equations: R * a = -r  (autocorrelation matrix system)

Author: Educational example for Signal Processing
License: MIT
"""

import numpy as np
from scipy import signal
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def make_test_signal(fs=1000.0, duration=2.0, seed=42):
    """
    Construct a test signal with known spectral content:
        - Two closely-spaced sinusoids at 100 and 120 Hz
        - One isolated sinusoid at 300 Hz
        - White Gaussian noise (SNR ≈ 10 dB)

    This combination tests each estimator's ability to:
      (a) resolve two nearby peaks (resolution test)
      (b) correctly locate an isolated peak (frequency accuracy test)

    Args:
        fs       (float): Sampling rate in Hz
        duration (float): Signal length in seconds
        seed     (int)  : RNG seed for reproducibility

    Returns:
        (t, x): time vector and signal ndarray
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / fs)
    N = len(t)

    # Sinusoidal components
    x = (1.0 * np.sin(2 * np.pi * 100 * t) +
         0.8 * np.sin(2 * np.pi * 120 * t) +
         0.6 * np.sin(2 * np.pi * 300 * t))

    # White noise — std chosen so broadband SNR ≈ 10 dB
    noise_std = 0.3
    x += noise_std * rng.standard_normal(N)

    return t, x


# ============================================================================
# NON-PARAMETRIC ESTIMATORS
# ============================================================================

def periodogram(x, fs):
    """
    Classic periodogram: P(k) = |X(k)|² / N.

    The periodogram has the same resolution as the DFT (Δf = fs/N),
    but its variance does NOT decrease as N increases — it remains
    proportional to the true spectrum squared at each frequency.

    Args:
        x  (ndarray): Input signal (length N)
        fs (float)  : Sampling rate (Hz)

    Returns:
        (freqs, Pxx): frequency axis and one-sided PSD (V²/Hz)
    """
    N = len(x)
    # No windowing (rectangular window) → highest frequency resolution
    X = np.fft.rfft(x) / N
    Pxx = 2 * np.abs(X) ** 2 / (fs / N)   # one-sided PSD: multiply by 2 (except DC/Nyq)
    Pxx[0] /= 2      # DC bin is not doubled
    if N % 2 == 0:
        Pxx[-1] /= 2  # Nyquist bin (only if N even)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return freqs, Pxx


def bartlett(x, fs, K):
    """
    Bartlett's method: average K non-overlapping periodograms.

    Divides x into K non-overlapping segments of length L = N//K.
    Each segment's periodogram is computed (rectangular window),
    and the K periodograms are averaged.

    Effect:
        - Variance reduced by factor K  (σ² → σ²/K)
        - Frequency resolution reduced: Δf = fs/L = fs*K/N  (worse by K)

    Args:
        x  (ndarray): Input signal
        fs (float)  : Sampling rate
        K  (int)    : Number of non-overlapping segments

    Returns:
        (freqs, Pxx_avg): frequency axis and averaged PSD
    """
    N = len(x)
    L = N // K   # segment length

    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    Pxx_sum = np.zeros(len(freqs))

    for k in range(K):
        seg = x[k * L:(k + 1) * L]
        _, P_seg = periodogram(seg, fs)
        # P_seg may differ in length if rfftfreq gives different bins;
        # take the minimum length to be safe
        n = min(len(Pxx_sum), len(P_seg))
        Pxx_sum[:n] += P_seg[:n]

    Pxx_avg = Pxx_sum / K
    return freqs, Pxx_avg


def welch_method(x, fs, nperseg, noverlap=None, window='hann'):
    """
    Welch's method: overlapping windowed periodograms averaged.

    Improvements over Bartlett:
      - Overlapping segments (typically 50 %) increase the number of
        averages without reducing the segment length as much.
      - Windowing (e.g. Hann) reduces spectral leakage.

    The resolution-variance tradeoff is parameterized by nperseg:
      - Larger nperseg → better resolution, higher variance
      - Smaller nperseg → worse resolution, lower variance

    Uses scipy.signal.welch for a robust, well-tested implementation.

    Args:
        x        (ndarray): Input signal
        fs       (float)  : Sampling rate (Hz)
        nperseg  (int)    : Samples per segment
        noverlap (int)    : Overlap samples (default: nperseg//2)
        window   (str)    : Window function name

    Returns:
        (freqs, Pxx): scipy.signal.welch output
    """
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, Pxx = signal.welch(x, fs=fs, window=window,
                               nperseg=nperseg, noverlap=noverlap,
                               scaling='density')
    return freqs, Pxx


# ============================================================================
# PARAMETRIC ESTIMATOR: AR / YULE-WALKER
# ============================================================================

def ar_yule_walker(x, order, fs):
    """
    Parametric spectral estimate using an AR(p) model.

    The AR model assumes the signal is the output of an all-pole IIR
    filter driven by white noise:

        x[n] = -a[1]*x[n-1] - ... - a[p]*x[n-p] + e[n]

    The Yule-Walker equations express this in matrix form:
        R_xx * a = -r_xx
    where R_xx is the p×p autocorrelation (Toeplitz) matrix and r_xx
    is the vector of lags 1…p.

    Once 'a' is found, the PSD estimate is:
        P(ω) = σ²_e / |A(e^jω)|²
    where A(e^jω) = 1 + a[1]e^{-jω} + … + a[p]e^{-jpω}.

    This method can resolve closely-spaced peaks far better than DFT-based
    methods when the model order matches the signal structure.

    Args:
        x     (ndarray): Input signal
        order (int)    : AR model order p (number of poles)
        fs    (float)  : Sampling rate (Hz)

    Returns:
        (freqs, Pxx): frequency axis and PSD estimate
    """
    N = len(x)

    # Estimate autocorrelation lags 0 … order (biased estimator)
    r = np.array([np.dot(x[:N - k], x[k:]) / N for k in range(order + 1)])

    # Build Toeplitz matrix from lags 0 … order-1
    R = toeplitz(r[:order])     # p × p symmetric Toeplitz
    r_vec = r[1:order + 1]      # lags 1 … p

    # Solve Yule-Walker: R * a = -r_vec
    a = np.linalg.solve(R, -r_vec)   # AR coefficients a[1]…a[p]

    # Noise variance: σ²_e = r[0] + a · r[1:p+1]
    sigma2 = r[0] + np.dot(a, r_vec)

    # Evaluate AR PSD on dense frequency grid
    nfft = 4096
    # Transfer function denominator: A(z) = 1 + a[0]*z^{-1} + … + a[p-1]*z^{-p}
    A_coeffs = np.concatenate([[1.0], a])
    _, H = signal.freqz(1.0, A_coeffs, worN=nfft // 2 + 1, fs=fs)

    # PSD = sigma^2 / |H(ω)|^2  — already one-sided from freqz on [0, Nyq]
    freqs = np.linspace(0, fs / 2, nfft // 2 + 1)
    Pxx = sigma2 / (fs / 2) * np.abs(H) ** (-2)   # normalize to V²/Hz
    # Note: we use |A|^{-2} = |H|^2 since H = 1/A for an all-pole filter

    return freqs, Pxx


# ============================================================================
# ANALYSIS HELPERS
# ============================================================================

def find_peaks_above(freqs, Pxx_db, threshold_db=-20.0):
    """Return (freq, power_dB) pairs for peaks above threshold."""
    peak_idx, _ = signal.find_peaks(Pxx_db, height=threshold_db, distance=5)
    return [(freqs[i], Pxx_db[i]) for i in peak_idx]


def print_method_summary(results, true_freqs):
    """Print detected peak frequencies for each method."""
    print("\n" + "=" * 65)
    print("PEAK DETECTION SUMMARY")
    print(f"True sinusoid frequencies: {true_freqs} Hz")
    print("=" * 65)
    print(f"{'Method':<20} {'Detected peaks (Hz)':>40}")
    print("-" * 65)

    for name, (freqs, Pxx) in results.items():
        Pxx_db = 10 * np.log10(np.maximum(Pxx, 1e-20))
        threshold = np.max(Pxx_db) - 25   # peaks within 25 dB of max
        peaks = find_peaks_above(freqs, Pxx_db, threshold_db=threshold)
        peak_str = ', '.join(f'{f:.1f}' for f, _ in sorted(peaks))
        print(f"  {name:<18} {peak_str:>38}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("SPECTRAL ESTIMATION METHODS")

    # Signal parameters
    FS = 1000.0        # Hz
    DURATION = 2.0     # seconds
    TRUE_FREQS = [100, 120, 300]   # Hz (known ground truth)
    AR_ORDER = 12      # AR model order — choose ≥ 2× number of sinusoids

    t, x = make_test_signal(fs=FS, duration=DURATION)
    N = len(x)

    print(f"\nSignal: {N} samples at {FS} Hz  ({DURATION} s)")
    print(f"True frequencies: {TRUE_FREQS} Hz  (100 and 120 Hz are closely spaced)")
    print(f"White noise std: 0.3")

    # -----------------------------------------------------------------------
    # 1. Periodogram
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("1. PERIODOGRAM  (rectangular window, no averaging)")
    print("=" * 65)
    f_pgram, P_pgram = periodogram(x, FS)
    print(f"   Resolution: Δf = {FS / N:.3f} Hz")
    print(f"   Variance  : high (not reduced by N)")

    # -----------------------------------------------------------------------
    # 2. Bartlett's method
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("2. BARTLETT'S METHOD  (non-overlapping segments)")
    print("=" * 65)
    K_bartlett = 8   # 8 segments → variance reduced by 8
    f_bart, P_bart = bartlett(x, FS, K_bartlett)
    L_bart = N // K_bartlett
    print(f"   Segments K = {K_bartlett}  →  segment length L = {L_bart}")
    print(f"   Resolution  : Δf = {FS / L_bart:.3f} Hz  (×{K_bartlett} worse than periodogram)")
    print(f"   Variance    : reduced by factor {K_bartlett}")

    # -----------------------------------------------------------------------
    # 3. Welch's method — multiple segment lengths
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("3. WELCH'S METHOD  (overlapping Hann windows)")
    print("=" * 65)
    welch_configs = [
        ('Welch (large seg)', N // 2),    # fewer averages, better resolution
        ('Welch (small seg)', N // 8),    # more averages, lower variance
    ]
    welch_results = {}
    for label, nperseg in welch_configs:
        f_w, P_w = welch_method(x, FS, nperseg=nperseg)
        n_seg_eff = 2 * N // nperseg - 1   # approx effective segments (50 % overlap)
        welch_results[label] = (f_w, P_w)
        print(f"   {label}: nperseg={nperseg}, Δf={FS/nperseg:.2f} Hz, "
              f"≈{n_seg_eff} effective averages")

    # -----------------------------------------------------------------------
    # 4. AR Yule-Walker
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"4. AR MODEL (Yule-Walker, order p={AR_ORDER})")
    print("=" * 65)
    f_ar, P_ar = ar_yule_walker(x, order=AR_ORDER, fs=FS)
    print(f"   Model order p = {AR_ORDER}  (rule of thumb: p ≥ N/3 for short records)")
    print(f"   Resolution    : super-resolution — can distinguish sub-DFT peaks")
    print(f"   Caution       : wrong order → spurious peaks or missed peaks")

    # -----------------------------------------------------------------------
    # Print peak summary
    # -----------------------------------------------------------------------
    all_results = {
        'Periodogram': (f_pgram, P_pgram),
        'Bartlett': (f_bart, P_bart),
        welch_configs[0][0]: welch_results[welch_configs[0][0]],
        welch_configs[1][0]: welch_results[welch_configs[1][0]],
        'AR Yule-Walker': (f_ar, P_ar),
    }
    print_method_summary(all_results, TRUE_FREQS)

    # -----------------------------------------------------------------------
    # VISUALIZATION — 3×2 grid
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle('Spectral Estimation Methods Comparison', fontsize=13, fontweight='bold')

    # Helper: dB conversion with floor
    def to_db(P):
        return 10 * np.log10(np.maximum(P, 1e-20))

    # (0,0) Periodogram
    ax = axes[0, 0]
    ax.plot(f_pgram, to_db(P_pgram), 'b-', lw=0.8, label='Periodogram')
    for f_true in TRUE_FREQS:
        ax.axvline(f_true, color='r', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB re V²/Hz)')
    ax.set_title(f'Periodogram (N={N}, Δf={FS/N:.2f} Hz)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, FS / 2)

    # (0,1) Periodogram — zoom in on 80–140 Hz to show 100 vs 120 Hz
    ax = axes[0, 1]
    ax.plot(f_pgram, to_db(P_pgram), 'b-', lw=1.2, label='Periodogram')
    for f_true in [100, 120]:
        ax.axvline(f_true, color='r', ls='--', lw=1.2, alpha=0.8, label=f'{f_true} Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB re V²/Hz)')
    ax.set_title('Zoom: 80–140 Hz (can periodogram resolve 100 & 120 Hz?)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(80, 140)

    # (1,0) Bartlett vs Periodogram
    ax = axes[1, 0]
    ax.plot(f_pgram, to_db(P_pgram), 'gray', lw=0.8, alpha=0.6, label='Periodogram')
    ax.plot(f_bart, to_db(P_bart), 'b-', lw=2,
            label=f'Bartlett (K={K_bartlett}, Δf={FS/(N//K_bartlett):.1f} Hz)')
    for f_true in TRUE_FREQS:
        ax.axvline(f_true, color='r', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB re V²/Hz)')
    ax.set_title(f"Bartlett: variance ↓ {K_bartlett}×, resolution ↓ {K_bartlett}×")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, FS / 2)

    # (1,1) Welch — two segment lengths
    ax = axes[1, 1]
    ax.plot(f_pgram, to_db(P_pgram), 'gray', lw=0.8, alpha=0.5, label='Periodogram')
    wcolors = ['tab:blue', 'tab:orange']
    for (label, _), color in zip(welch_configs, wcolors):
        fw, Pw = welch_results[label]
        ax.plot(fw, to_db(Pw), color=color, lw=1.8, label=label)
    for f_true in TRUE_FREQS:
        ax.axvline(f_true, color='r', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB re V²/Hz)')
    ax.set_title('Welch: resolution vs. variance tradeoff')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, FS / 2)

    # (2,0) AR Yule-Walker vs Welch (full range)
    ax = axes[2, 0]
    f_w_large, P_w_large = welch_results[welch_configs[0][0]]
    ax.plot(f_w_large, to_db(P_w_large), 'b-', lw=1.5, label='Welch (large seg)')
    ax.plot(f_ar, to_db(P_ar), 'r--', lw=2,
            label=f'AR Yule-Walker (p={AR_ORDER})')
    for f_true in TRUE_FREQS:
        ax.axvline(f_true, color='k', ls=':', lw=1.2, alpha=0.8)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB re V²/Hz)')
    ax.set_title(f'AR(p={AR_ORDER}) vs Welch: parametric super-resolution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, FS / 2)

    # (2,1) AR zoom — 80–140 Hz (resolution comparison)
    ax = axes[2, 1]
    ax.plot(f_pgram, to_db(P_pgram), 'gray', lw=1, alpha=0.6, label='Periodogram')
    ax.plot(f_w_large, to_db(P_w_large), 'b-', lw=1.8, label='Welch (large seg)')
    ax.plot(f_ar, to_db(P_ar), 'r--', lw=2, label=f'AR (p={AR_ORDER})')
    for f_true in [100, 120]:
        ax.axvline(f_true, color='k', ls=':', lw=1.5, alpha=0.8, label=f'{f_true} Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB re V²/Hz)')
    ax.set_title('Zoom 80–140 Hz: AR resolves 100 & 120 Hz; Welch may not')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(80, 140)

    plt.tight_layout()
    out_path = '/opt/projects/01_Personal/03_Study/examples/Signal_Processing/12_spectral_estimation.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved visualization: {out_path}")
    plt.close()

    print("\n" + "=" * 65)
    print("Key Takeaways:")
    print("  - Periodogram: maximum resolution (Δf=fs/N), but high variance")
    print("  - Bartlett: averaging K segments reduces variance by K, resolution K× worse")
    print("  - Welch: 50% overlap + windowing — best practical non-parametric method")
    print("  - AR / Yule-Walker: parametric, super-resolution for short records")
    print("  - Resolution-variance tradeoff is fundamental (Heisenberg-like)")
    print("  - AR order too small → merged peaks; too large → spurious peaks")
    print("=" * 65)
