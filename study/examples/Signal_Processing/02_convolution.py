#!/usr/bin/env python3
"""
LTI Systems and Convolution

Demonstrates linear time-invariant (LTI) system concepts:
- Discrete convolution implemented from scratch
- Comparison with np.convolve for verification
- LTI system response: output = input * impulse_response
- Moving average filter as a convolution example
- Step-by-step visualization of the sliding-window convolution
- Commutativity, associativity, and distributivity properties
- RC circuit impulse response and system output

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def convolve_scratch(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Compute discrete linear convolution of x and h from scratch.

    The convolution sum:
        y[n] = sum_{k=-inf}^{inf} x[k] * h[n - k]

    For finite-length sequences of length N and M,
    the output has length N + M - 1.

    Parameters
    ----------
    x : input signal, length N
    h : impulse response (kernel), length M

    Returns
    -------
    y : convolution output, length N + M - 1
    """
    N = len(x)
    M = len(h)
    L = N + M - 1               # output length
    y = np.zeros(L)

    for n in range(L):
        # Accumulate x[k] * h[n - k] for valid k values
        for k in range(N):
            m = n - k           # index into h
            if 0 <= m < M:
                y[n] += x[k] * h[m]
    return y


def demo_scratch_vs_numpy():
    """Verify the from-scratch convolution against np.convolve."""
    print("=" * 60)
    print("CONVOLUTION: FROM SCRATCH vs np.convolve")
    print("=" * 60)

    x = np.array([1, 2, 3, 4, 5], dtype=float)
    h = np.array([1, 0, -1], dtype=float)   # simple difference filter

    y_scratch = convolve_scratch(x, h)
    y_numpy   = np.convolve(x, h)            # 'full' mode by default

    print(f"Input x     = {x}")
    print(f"Kernel h    = {h}")
    print(f"y (scratch) = {y_scratch}")
    print(f"y (numpy)   = {y_numpy}")
    print(f"Max absolute error = {np.max(np.abs(y_scratch - y_numpy)):.2e}")
    print(f"Results match: {np.allclose(y_scratch, y_numpy)}")


def demo_lti_response():
    """
    LTI system response: y = x * h

    A system is LTI if it satisfies:
      - Linearity:   response to a*x1 + b*x2 is a*y1 + b*y2
      - Time-invariance: if x[n] -> y[n], then x[n-k] -> y[n-k]

    The output of any LTI system is completely described by its
    impulse response h[n] via convolution: y[n] = (x * h)[n].
    """
    print("\n" + "=" * 60)
    print("LTI SYSTEM RESPONSE")
    print("=" * 60)

    n = np.arange(0, 20)

    # Input: x[n] = u[n] (unit step)
    x = (n >= 0).astype(float)

    # Impulse response: h[n] = (0.7)^n * u[n]  (first-order IIR)
    alpha = 0.7
    h = alpha ** n * (n >= 0).astype(float)

    # Output via convolution
    y = np.convolve(x, h)[:len(n)]

    # Analytical step response: y[n] = (1 - alpha^(n+1)) / (1 - alpha)
    y_analytical = (1 - alpha ** (n + 1)) / (1 - alpha)

    print(f"Input x[n] = u[n] (unit step)")
    print(f"Impulse response h[n] = ({alpha})^n · u[n]")
    print(f"System: y[n] = sum_k x[k]·h[n-k]")
    print(f"Max error vs analytical = {np.max(np.abs(y - y_analytical)):.2e}")

    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    fig.suptitle("LTI System Response", fontsize=13, fontweight='bold')

    axes[0].stem(n, x, basefmt='k-', linefmt='C0-', markerfmt='C0o')
    axes[0].set_title("Input x[n] = u[n] (Unit Step)")
    axes[0].set_ylabel("Amplitude")

    axes[1].stem(n, h, basefmt='k-', linefmt='C1-', markerfmt='C1o')
    axes[1].set_title(f"Impulse Response h[n] = ({alpha})^n · u[n]")
    axes[1].set_ylabel("Amplitude")

    axes[2].stem(n, y, basefmt='k-', linefmt='C2-', markerfmt='C2o')
    axes[2].plot(n, y_analytical, 'r--', label='Analytical')
    axes[2].set_title("Output y[n] = x[n] * h[n]")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("n (samples)")
    axes[2].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("n (samples)")

    plt.tight_layout()
    plt.show()


def demo_moving_average():
    """
    Moving average filter as convolution.

    A length-M moving average is a FIR (Finite Impulse Response) filter
    with impulse response h[n] = 1/M for 0 <= n < M, else 0.

    Convolving a noisy signal with h smooths out rapid fluctuations
    (acts as a low-pass filter).
    """
    print("\n" + "=" * 60)
    print("MOVING AVERAGE FILTER (CONVOLUTION EXAMPLE)")
    print("=" * 60)

    rng = np.random.default_rng(0)
    n = np.arange(0, 100)
    clean   = np.sin(2 * np.pi * 0.05 * n)
    noisy   = clean + 0.5 * rng.standard_normal(len(n))

    M = 7                           # filter length
    h_ma = np.ones(M) / M          # impulse response of moving average

    # 'same' mode keeps output length equal to input
    filtered = np.convolve(noisy, h_ma, mode='same')

    mse_noisy    = np.mean((noisy    - clean) ** 2)
    mse_filtered = np.mean((filtered - clean) ** 2)
    print(f"Filter length M = {M}")
    print(f"MSE (noisy vs clean)    = {mse_noisy:.4f}")
    print(f"MSE (filtered vs clean) = {mse_filtered:.4f}")
    print(f"Noise reduction factor  = {mse_noisy / mse_filtered:.2f}x")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(n, noisy,    'lightblue', linewidth=1,   label='Noisy signal')
    ax.plot(n, clean,    'gray',      linewidth=2,   label='Clean signal', linestyle='--')
    ax.plot(n, filtered, 'red',       linewidth=2,   label=f'Moving average (M={M})')
    ax.set_title("Moving Average Filter via Convolution")
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def demo_convolution_stepwise():
    """
    Visualize convolution as a sliding-window operation.

    At each output index n, we:
      1. Flip (reverse) h to get h[-k].
      2. Shift h[-k] by n to get h[n-k].
      3. Multiply element-wise with x[k].
      4. Sum all products to get y[n].
    """
    print("\n" + "=" * 60)
    print("STEP-BY-STEP CONVOLUTION VISUALIZATION")
    print("=" * 60)

    x = np.array([1, 2, 1, 3, 1], dtype=float)
    h = np.array([1, 1, 1], dtype=float)   # 3-point moving sum
    N, M = len(x), len(h)
    L = N + M - 1

    y = np.zeros(L)
    n_steps = [1, 2, 4]   # which output indices to visualize

    fig, axes = plt.subplots(len(n_steps), 3, figsize=(13, 8))
    fig.suptitle("Convolution Step-by-Step (x*h)[n] = Σ x[k]·h[n-k]",
                 fontsize=12, fontweight='bold')

    # Pad x and flipped h into full output-length arrays for visualization
    x_padded = np.concatenate([x, np.zeros(M - 1)])

    for row, n_idx in enumerate(n_steps):
        # h reversed and shifted: h[n-k] for k=0..L-1
        h_flipped = np.zeros(L)
        for k in range(L):
            m = n_idx - k
            if 0 <= m < M:
                h_flipped[k] = h[m]

        product = x_padded * h_flipped
        y[n_idx] = np.sum(product)

        k_ax = np.arange(L)
        axes[row, 0].stem(k_ax, x_padded,  basefmt='k-', linefmt='C0-', markerfmt='C0o')
        axes[row, 0].set_title(f"x[k]  (n={n_idx})")

        axes[row, 1].stem(k_ax, h_flipped, basefmt='k-', linefmt='C1-', markerfmt='C1o')
        axes[row, 1].set_title(f"h[{n_idx}-k]  (flipped & shifted h)")

        axes[row, 2].stem(k_ax, product,   basefmt='k-', linefmt='C2-', markerfmt='C2o')
        axes[row, 2].set_title(f"x[k]·h[{n_idx}-k], sum={y[n_idx]:.1f} = y[{n_idx}]")

    for ax in axes.flat:
        ax.set_xlabel("k")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Compute all outputs and compare
    y_full = np.convolve(x, h)
    print(f"x = {x}")
    print(f"h = {h}")
    print(f"y = x*h = {y_full}")


def demo_properties():
    """
    Verify convolution properties:
      1. Commutativity:   x * h = h * x
      2. Associativity:   (x * h1) * h2 = x * (h1 * h2)
      3. Distributivity:  x * (h1 + h2) = x*h1 + x*h2
    """
    print("\n" + "=" * 60)
    print("CONVOLUTION PROPERTIES")
    print("=" * 60)

    x  = np.array([1, 2, 3, 4], dtype=float)
    h1 = np.array([1, -1],      dtype=float)
    h2 = np.array([1, 0, 1],    dtype=float)

    # 1. Commutativity
    lhs = np.convolve(x, h1)
    rhs = np.convolve(h1, x)
    print(f"1. Commutativity  x*h1 = h1*x:       {np.allclose(lhs, rhs)}")

    # 2. Associativity
    lhs = np.convolve(np.convolve(x, h1), h2)
    rhs = np.convolve(x, np.convolve(h1, h2))
    print(f"2. Associativity  (x*h1)*h2 = x*(h1*h2): {np.allclose(lhs, rhs)}")

    # 3. Distributivity
    lhs = np.convolve(x, h1 + np.pad(h2, (0, len(h1) - len(h2))
                                     if len(h2) < len(h1) else (len(h2) - len(h1), 0)))
    # Simpler: pad to equal length
    max_len = max(len(h1), len(h2))
    h1_p = np.pad(h1, (0, max_len - len(h1)))
    h2_p = np.pad(h2, (0, max_len - len(h2)))
    lhs = np.convolve(x, h1_p + h2_p)
    rhs = np.convolve(x, h1_p) + np.pad(np.convolve(x, h2_p),
                                          (0, len(lhs) - len(np.convolve(x, h2_p))))
    rhs = np.convolve(x, h1_p) + np.convolve(x, h2_p)
    # Both outputs have the same length here
    print(f"3. Distributivity x*(h1+h2) = x*h1+x*h2: {np.allclose(lhs, rhs)}")


def demo_rc_circuit():
    """
    RC circuit impulse response and output for a step input.

    An RC low-pass filter has the continuous-time impulse response:
        h(t) = (1/RC) * e^(-t/RC) * u(t)

    We discretize at sampling period Ts and compute the output for
    a unit step input using convolution.
    """
    print("\n" + "=" * 60)
    print("RC CIRCUIT IMPULSE RESPONSE")
    print("=" * 60)

    RC = 0.1        # time constant (seconds)
    Ts = 0.001      # sampling period (seconds)
    T_total = 1.0   # total observation time

    t = np.arange(0, T_total, Ts)

    # Discretized impulse response (scaled so convolution * Ts approximates integral)
    h = (1 / RC) * np.exp(-t / RC) * Ts

    # Unit step input: x[n] = 1 for all n >= 0
    x = np.ones(len(t))

    # System output: y = x * h  (truncated to input length)
    y = np.convolve(x, h)[:len(t)]

    # Analytical step response: y(t) = 1 - e^(-t/RC)
    y_analytical = 1 - np.exp(-t / RC)

    print(f"RC time constant: {RC} s")
    print(f"Sampling period:  {Ts} s")
    print(f"At t = RC = {RC} s, step response should reach 1 - 1/e ≈ {1 - 1/np.e:.4f}")
    idx_RC = int(RC / Ts)
    print(f"Numerical value at t=RC:   {y[idx_RC]:.4f}")
    print(f"Analytical value at t=RC:  {y_analytical[idx_RC]:.4f}")
    print(f"Max error: {np.max(np.abs(y - y_analytical)):.2e}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("RC Low-Pass Filter (Convolution)", fontsize=13, fontweight='bold')

    axes[0].plot(t * 1000, h / Ts, 'b')   # scale back to density
    axes[0].set_title("Impulse Response h(t) = (1/RC)·e^(-t/RC)")
    axes[0].set_xlabel("t (ms)")
    axes[0].set_ylabel("h(t)")

    axes[1].plot(t * 1000, x, 'g', linewidth=2)
    axes[1].set_title("Input x(t) = u(t) (Unit Step)")
    axes[1].set_xlabel("t (ms)")
    axes[1].set_ylim(-0.1, 1.5)

    axes[2].plot(t * 1000, y,            'r',    linewidth=2, label='Convolution')
    axes[2].plot(t * 1000, y_analytical, 'k--',  linewidth=1, label='Analytical')
    axes[2].axhline(1 - 1/np.e, color='gray', linestyle=':', label='63.2% (t=RC)')
    axes[2].axvline(RC * 1000,  color='gray', linestyle=':')
    axes[2].set_title("Output y(t) = 1 - e^(-t/RC)")
    axes[2].set_xlabel("t (ms)")
    axes[2].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_scratch_vs_numpy()
    demo_lti_response()
    demo_moving_average()
    demo_convolution_stepwise()
    demo_properties()
    demo_rc_circuit()

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
