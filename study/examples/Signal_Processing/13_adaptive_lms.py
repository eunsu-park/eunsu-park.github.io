#!/usr/bin/env python3
"""
Adaptive Filters: LMS and NLMS Algorithms
==========================================

Adaptive filters adjust their coefficients automatically to minimise an error
signal.  Unlike fixed FIR/IIR filters, they require no a-priori knowledge of
the signal statistics — they learn on-line.

The Least Mean Squares (LMS) Algorithm
---------------------------------------
Update rule:
    w[n+1] = w[n] + mu * e[n] * x[n]

where
    x[n]  : input vector (most recent M samples)
    d[n]  : desired signal
    y[n]  = w[n]^T x[n]   : filter output
    e[n]  = d[n] - y[n]   : error signal
    mu    : step size (controls speed vs stability)

Stability condition:
    0 < mu < 2 / (M * max_input_power)

Normalised LMS (NLMS)
----------------------
NLMS divides the step size by the instantaneous input power, making it
insensitive to input amplitude variations:

    w[n+1] = w[n] + (mu_n / (epsilon + x[n]^T x[n])) * e[n] * x[n]

where mu_n is now dimensionless (0 < mu_n < 2) and epsilon prevents
division by zero.

Applications demonstrated:
    1. System identification — estimate an unknown FIR filter's coefficients
    2. Noise cancellation — remove correlated noise from a desired signal

Author: Educational example for Signal Processing
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# LMS ADAPTIVE FILTER
# ============================================================================

def lms_filter(x, d, mu, M):
    """
    Least Mean Squares (LMS) adaptive filter.

    Args:
        x  (ndarray): Input signal, shape (N,)
        d  (ndarray): Desired signal, shape (N,)
        mu (float)  : Step size.  Larger mu → faster but less stable.
        M  (int)    : Filter length (number of taps).

    Returns:
        y    (ndarray): Filter output, shape (N,)
        e    (ndarray): Error signal e[n] = d[n] - y[n], shape (N,)
        W    (ndarray): Weight history, shape (N, M)
        mse  (ndarray): Instantaneous squared error |e[n]|^2, shape (N,)
    """
    N = len(x)
    w = np.zeros(M)          # filter weights initialised to zero
    y = np.zeros(N)
    e = np.zeros(N)
    W = np.zeros((N, M))     # weight trajectory (for analysis)

    for n in range(N):
        # Build input vector: x[n], x[n-1], ..., x[n-M+1]
        if n < M:
            x_vec = np.concatenate([x[:n+1][::-1], np.zeros(M - n - 1)])
        else:
            x_vec = x[n:n-M:-1]     # reversed window

        y[n] = w @ x_vec             # filter output
        e[n] = d[n] - y[n]           # error
        w = w + mu * e[n] * x_vec    # weight update (LMS rule)
        W[n] = w

    mse = e ** 2
    return y, e, W, mse


# ============================================================================
# NLMS ADAPTIVE FILTER
# ============================================================================

def nlms_filter(x, d, mu_n, M, epsilon=1e-6):
    """
    Normalised Least Mean Squares (NLMS) adaptive filter.

    The step size is normalised by the instantaneous input energy, so the
    algorithm is robust to changes in input power.

    Args:
        x       (ndarray): Input signal, shape (N,)
        d       (ndarray): Desired signal, shape (N,)
        mu_n    (float)  : Normalised step size, 0 < mu_n < 2.
        M       (int)    : Filter length (number of taps).
        epsilon (float)  : Small regularisation constant (prevents /0).

    Returns:
        Same as lms_filter.
    """
    N = len(x)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    W = np.zeros((N, M))

    for n in range(N):
        if n < M:
            x_vec = np.concatenate([x[:n+1][::-1], np.zeros(M - n - 1)])
        else:
            x_vec = x[n:n-M:-1]

        y[n] = w @ x_vec
        e[n] = d[n] - y[n]

        # NLMS: divide step size by input power
        power = x_vec @ x_vec
        w = w + (mu_n / (epsilon + power)) * e[n] * x_vec
        W[n] = w

    mse = e ** 2
    return y, e, W, mse


# ============================================================================
# APPLICATION 1: SYSTEM IDENTIFICATION
# ============================================================================

def demo_system_identification():
    """
    Use LMS/NLMS to identify the impulse response of an unknown FIR system.

    Setup:
        - 'Unknown system' H(z) is a random FIR filter of length M_true.
        - The adaptive filter w of the same length learns H iteratively.
        - After convergence, w ≈ H (system identified).
    """
    print("=" * 60)
    print("APPLICATION 1: System Identification")
    print("=" * 60)

    rng = np.random.default_rng(42)
    N = 2000          # number of samples
    M = 16            # filter length (same for unknown system and adaptive filter)

    # Unknown system impulse response (what we want to identify)
    h_true = rng.standard_normal(M)
    h_true /= np.linalg.norm(h_true)   # normalise for numerical convenience
    print(f"True impulse response (first 8 coefficients): {h_true[:8].round(3)}")

    # White noise excitation signal (wide-band → good identification)
    x = rng.standard_normal(N)

    # Desired signal = output of the unknown system + small observation noise
    d = np.convolve(x, h_true, mode='full')[:N]
    d += 0.05 * rng.standard_normal(N)     # SNR ≈ 26 dB

    # Run LMS with three different step sizes to illustrate the trade-off
    results = {}
    for mu in [0.01, 0.05, 0.2]:
        y, e, W, mse = lms_filter(x, d, mu=mu, M=M)
        results[f'LMS mu={mu}'] = (W, mse)
        final_weights = W[-1]
        err = np.linalg.norm(final_weights - h_true)
        print(f"  LMS mu={mu:4.2f}  |  final ||w - h_true|| = {err:.4f}")

    # Run NLMS
    y_n, e_n, W_n, mse_n = nlms_filter(x, d, mu_n=0.5, M=M)
    final_nlms = W_n[-1]
    err_nlms = np.linalg.norm(final_nlms - h_true)
    print(f"  NLMS mu_n=0.5   |  final ||w - h_true|| = {err_nlms:.4f}")

    # ---- Plot ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("System Identification via LMS / NLMS", fontsize=14, fontweight='bold')

    # (a) Learning curves (smoothed MSE)
    ax = axes[0, 0]
    smooth = 50   # running average window
    for label, (_, mse) in results.items():
        kernel = np.ones(smooth) / smooth
        mse_smooth = np.convolve(mse, kernel, mode='valid')
        ax.semilogy(mse_smooth, label=label)
    mse_smooth_nlms = np.convolve(mse_n, np.ones(smooth) / smooth, mode='valid')
    ax.semilogy(mse_smooth_nlms, label='NLMS mu_n=0.5', linestyle='--')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE (smoothed, log scale)")
    ax.set_title("(a) Learning Curves — Effect of Step Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) True vs identified impulse response (best LMS)
    ax = axes[0, 1]
    best_W, _ = results['LMS mu=0.05']
    ax.stem(h_true, linefmt='C0-', markerfmt='C0o', basefmt='k-', label='True h')
    ax.stem(best_W[-1], linefmt='C1--', markerfmt='C1x', basefmt='k-', label='LMS (mu=0.05)')
    ax.stem(final_nlms, linefmt='C2:', markerfmt='C2^', basefmt='k-', label='NLMS')
    ax.set_xlabel("Tap index")
    ax.set_ylabel("Coefficient value")
    ax.set_title("(b) Identified vs True Impulse Response")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Weight evolution over time (LMS mu=0.05, first 4 taps)
    ax = axes[1, 0]
    W_mid, _ = results['LMS mu=0.05']
    for k in range(4):
        ax.plot(W_mid[:, k], label=f'w[{k}]')
        ax.axhline(h_true[k], color=f'C{k}', linestyle=':', linewidth=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Weight value")
    ax.set_title("(c) Weight Convergence (LMS mu=0.05, taps 0-3)\nDotted = true value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Residual error over time
    ax = axes[1, 1]
    for label, (W_hist, _) in results.items():
        residuals = [np.linalg.norm(W_hist[n] - h_true) for n in range(0, N, 10)]
        ax.semilogy(range(0, N, 10), residuals, label=label)
    residuals_nlms = [np.linalg.norm(W_n[n] - h_true) for n in range(0, N, 10)]
    ax.semilogy(range(0, N, 10), residuals_nlms, label='NLMS mu_n=0.5', linestyle='--')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("||w[n] - h_true||")
    ax.set_title("(d) Weight Error Norm over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("13_system_identification.png", dpi=120)
    print("  Saved: 13_system_identification.png")
    plt.show()


# ============================================================================
# APPLICATION 2: NOISE CANCELLATION
# ============================================================================

def demo_noise_cancellation():
    """
    Use NLMS to cancel correlated noise from a desired signal.

    Setup (classic adaptive noise canceller, Widrow 1975):
        - Primary input   : d[n] = s[n] + v1[n]   (signal + noise)
        - Reference input : x[n] = v2[n]           (correlated noise, no signal)
        - The adaptive filter estimates v1 from v2:
              y[n] ≈ v1[n]
        - Output: e[n] = d[n] - y[n] ≈ s[n]        (cleaned signal)

    The key requirement is that the reference noise v2 is correlated with
    the primary noise v1 but uncorrelated with the desired signal s.
    """
    print("\n" + "=" * 60)
    print("APPLICATION 2: Adaptive Noise Cancellation")
    print("=" * 60)

    rng = np.random.default_rng(7)
    fs = 1000            # sample rate (Hz)
    t = np.arange(2000) / fs

    # Desired signal: 50 Hz sinusoid
    s = np.sin(2 * np.pi * 50 * t)

    # Noise source: band-limited random noise
    v_source = rng.standard_normal(len(t))
    # Simulate two sensors picking up the same noise source through different paths
    noise_path1 = np.array([0.8, 0.3, -0.2, 0.1])   # path to primary sensor
    noise_path2 = np.array([0.5, -0.4, 0.6])          # path to reference sensor
    v1 = np.convolve(v_source, noise_path1, mode='full')[:len(t)]
    v2 = np.convolve(v_source, noise_path2, mode='full')[:len(t)]

    # Primary: signal + noise
    primary = s + v1
    reference = v2    # reference: correlated noise only

    # Input SNR before cancellation
    snr_in = 10 * np.log10(np.var(s) / np.var(v1))
    print(f"  Input SNR  : {snr_in:.1f} dB")

    # Run NLMS noise canceller
    M = 12    # adaptive filter length (must capture the path difference)
    y, e, W, mse = nlms_filter(reference, primary, mu_n=0.8, M=M)
    # e[n] is the cleaned output ≈ s[n]

    # Output SNR after cancellation
    # Compare cleaned signal to true s (using later half after convergence)
    half = len(t) // 2
    snr_out = 10 * np.log10(np.var(s[half:]) / np.var(e[half:] - s[half:]))
    print(f"  Output SNR : {snr_out:.1f} dB  (improvement: {snr_out - snr_in:.1f} dB)")

    # ---- Plot ---------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    fig.suptitle("Adaptive Noise Cancellation (NLMS)", fontsize=14, fontweight='bold')
    seg = slice(0, 300)   # show first 0.3 s

    axes[0, 0].plot(t[seg], s[seg], 'C0')
    axes[0, 0].set_title("(a) Desired Signal s[n] (50 Hz sinusoid)")
    axes[0, 0].set_ylabel("Amplitude")

    axes[0, 1].plot(t[seg], primary[seg], 'C1')
    axes[0, 1].set_title(f"(b) Primary Input d[n] = s + noise  (SNR = {snr_in:.1f} dB)")
    axes[0, 1].set_ylabel("Amplitude")

    axes[1, 0].plot(t[seg], reference[seg], 'C2')
    axes[1, 0].set_title("(c) Reference Input (correlated noise)")
    axes[1, 0].set_ylabel("Amplitude")

    axes[1, 1].plot(t[seg], e[seg], 'C3')
    axes[1, 1].set_title(f"(d) Cleaned Output e[n] ≈ s[n]  (SNR = {snr_out:.1f} dB)")
    axes[1, 1].set_ylabel("Amplitude")

    # Learning curve
    smooth = 30
    kernel = np.ones(smooth) / smooth
    mse_smooth = np.convolve(mse, kernel, mode='valid')
    axes[2, 0].semilogy(mse_smooth, 'C4')
    axes[2, 0].set_title("(e) Learning Curve (MSE)")
    axes[2, 0].set_xlabel("Iteration")
    axes[2, 0].set_ylabel("MSE (log)")
    axes[2, 0].grid(True, alpha=0.3)

    # Overlay comparison (last 200 samples after convergence)
    seg2 = slice(1700, 2000)
    axes[2, 1].plot(t[seg2], s[seg2], 'C0', label='True s[n]', linewidth=2)
    axes[2, 1].plot(t[seg2], e[seg2], 'C3--', label='Cleaned e[n]', linewidth=1.5)
    axes[2, 1].set_title("(f) True vs Cleaned (after convergence)")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel("Amplitude")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("13_noise_cancellation.png", dpi=120)
    print("  Saved: 13_noise_cancellation.png")
    plt.show()


# ============================================================================
# BONUS: LMS vs NLMS CONVERGENCE COMPARISON
# ============================================================================

def demo_lms_vs_nlms():
    """
    Direct comparison of LMS and NLMS when input amplitude changes mid-way.

    LMS is sensitive to input power (step size must be retuned).
    NLMS adapts automatically.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: LMS vs NLMS under non-stationary input")
    print("=" * 60)

    rng = np.random.default_rng(99)
    N = 3000
    M = 8
    h_true = np.array([0.5, 0.3, -0.2, 0.1, 0.05, -0.05, 0.02, 0.01])

    # Input signal: amplitude doubles at n=1500 (non-stationary scenario)
    x = rng.standard_normal(N)
    x[1500:] *= 5.0      # sudden power increase

    d = np.convolve(x, h_true, mode='full')[:N]
    d += 0.02 * rng.standard_normal(N)

    _, _, _, mse_lms = lms_filter(x, d, mu=0.01, M=M)
    _, _, _, mse_nlms = nlms_filter(x, d, mu_n=0.5, M=M)

    smooth = 40
    kernel = np.ones(smooth) / smooth
    mse_lms_s = np.convolve(mse_lms, kernel, mode='valid')
    mse_nlms_s = np.convolve(mse_nlms, kernel, mode='valid')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(mse_lms_s, label='LMS (mu=0.01)', color='C0')
    ax.semilogy(mse_nlms_s, label='NLMS (mu_n=0.5)', color='C1', linestyle='--')
    ax.axvline(1500, color='red', linestyle=':', linewidth=1.5, label='Amplitude×5 at n=1500')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("LMS vs NLMS: Non-Stationary Input (amplitude jump at n=1500)\n"
                 "LMS diverges; NLMS adapts automatically")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("13_lms_vs_nlms.png", dpi=120)
    print("  Saved: 13_lms_vs_nlms.png")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Adaptive Filters: LMS and NLMS")
    print("=" * 60)
    print("Key parameters:")
    print("  mu  (LMS step size) : controls speed vs stability trade-off")
    print("  mu_n (NLMS)         : normalised step size, 0 < mu_n < 2")
    print("  M   (filter order)  : must be >= true system order")
    print()

    demo_system_identification()
    demo_noise_cancellation()
    demo_lms_vs_nlms()

    print("\nDone.  Three PNG files saved.")
