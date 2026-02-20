#!/usr/bin/env python3
"""
Structure Functions and Intermittency in MHD Turbulence

This script computes structure functions of velocity and magnetic field
to characterize the scaling properties and intermittency of MHD turbulence.

Structure functions:
    S_p(ℓ) = <|δv(ℓ)|^p>
    B_p(ℓ) = <|δB(ℓ)|^p>

where δv(ℓ) = v(x + ℓ) - v(x) is the velocity increment.

Key results:
- Scaling exponents ζ(p) deviate from Kolmogorov prediction p/3
- Intermittency: rare, intense fluctuations at small scales
- Probability density functions (PDFs) show non-Gaussian tails
- Magnetic field shows similar intermittency to velocity

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import curve_fit
from scipy.stats import kurtosis, skew


def generate_multifractal_field(N, alpha, intermittency=0.3):
    """
    Generate a multifractal field with intermittency.

    Uses a multiplicative cascade model to create fields with
    non-Gaussian statistics and scale-dependent structure.

    Parameters
    ----------
    N : int
        Number of grid points
    alpha : float
        Spectral index
    intermittency : float
        Intermittency parameter (0 = no intermittency)

    Returns
    -------
    field : ndarray
        1D field with intermittent structures
    """
    # Generate Gaussian random field with power-law spectrum
    np.random.seed(42)
    k = np.fft.fftfreq(N, d=1.0) * N
    k[0] = 1e-10  # Avoid division by zero

    # Power spectrum
    P_k = np.abs(k)**(-alpha)
    P_k[0] = 0

    # Random phases
    phases = np.random.uniform(0, 2*np.pi, N)
    field_k = np.sqrt(P_k) * np.exp(1j * phases)

    # Transform to real space
    field = np.fft.ifft(field_k).real

    # Add intermittency via multiplicative cascade
    if intermittency > 0:
        # Generate random multipliers
        n_levels = int(np.log2(N))
        multipliers = np.ones(N)

        for level in range(n_levels):
            block_size = 2**(n_levels - level)
            n_blocks = N // block_size

            for i in range(n_blocks):
                # Log-normal multiplier
                sigma = intermittency / np.sqrt(level + 1)
                mult = np.exp(np.random.normal(0, sigma) - sigma**2/2)
                multipliers[i*block_size:(i+1)*block_size] *= mult

        field = field * multipliers

    # Normalize
    field = field / np.std(field)

    return field


def compute_structure_function(field, order, max_lag=None):
    """
    Compute structure function S_p(ℓ) = <|δfield(ℓ)|^p>.

    Parameters
    ----------
    field : ndarray
        1D field (velocity or magnetic)
    order : float
        Order p of structure function
    max_lag : int or None
        Maximum lag (None = N/4)

    Returns
    -------
    lags : ndarray
        Lag distances ℓ
    S_p : ndarray
        Structure function values
    """
    N = len(field)
    if max_lag is None:
        max_lag = N // 4

    lags = np.arange(1, max_lag)
    S_p = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        # Compute increments
        delta_field = field[lag:] - field[:-lag]
        # Structure function: average of |δfield|^p
        S_p[i] = np.mean(np.abs(delta_field)**order)

    return lags, S_p


def compute_scaling_exponent(lags, S_p, fit_range=(0.1, 0.5)):
    """
    Compute scaling exponent ζ(p) from S_p(ℓ) ~ ℓ^ζ(p).

    Parameters
    ----------
    lags : ndarray
        Lag distances
    S_p : ndarray
        Structure function
    fit_range : tuple
        (min_fraction, max_fraction) of lags to use for fitting

    Returns
    -------
    zeta : float
        Scaling exponent
    """
    # Select fitting range
    N = len(lags)
    i_min = int(fit_range[0] * N)
    i_max = int(fit_range[1] * N)

    # Log-log fit
    log_lags = np.log(lags[i_min:i_max])
    log_S_p = np.log(S_p[i_min:i_max])

    # Linear fit
    coeffs = np.polyfit(log_lags, log_S_p, 1)
    zeta = coeffs[0]

    return zeta


def compute_pdf(field, n_bins=50):
    """
    Compute probability density function of field increments.

    Parameters
    ----------
    field : ndarray
        1D field
    n_bins : int
        Number of bins for histogram

    Returns
    -------
    bins : ndarray
        Bin centers
    pdf : ndarray
        Probability density
    """
    # Compute increments at small scale
    lag = len(field) // 20
    delta_field = field[lag:] - field[:-lag]

    # Normalize by standard deviation
    delta_field = delta_field / np.std(delta_field)

    # Compute histogram
    counts, bin_edges = np.histogram(delta_field, bins=n_bins, density=True)
    bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    return bins, counts


def gaussian_pdf(x):
    """Standard Gaussian PDF."""
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)


def plot_structure_functions():
    """
    Compute and plot structure functions for MHD turbulence.
    """
    # Generate turbulent fields
    N = 2048
    print("Generating turbulent fields...")

    # Velocity field (with intermittency)
    v_field = generate_multifractal_field(N, alpha=5/3, intermittency=0.4)

    # Magnetic field (with intermittency)
    B_field = generate_multifractal_field(N, alpha=5/3, intermittency=0.4)

    # Gaussian field (no intermittency, for comparison)
    v_gaussian = generate_multifractal_field(N, alpha=5/3, intermittency=0.0)

    # Compute structure functions for different orders
    orders = [1, 2, 3, 4, 5, 6]
    print("Computing structure functions...")

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Structure functions for velocity
    ax1 = plt.subplot(3, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(orders)))

    for i, p in enumerate(orders):
        lags, S_p = compute_structure_function(v_field, p)
        ax1.loglog(lags, S_p, 'o-', color=colors[i], markersize=3,
                   linewidth=1.5, alpha=0.7, label=f'p={p}')

        # Kolmogorov prediction: S_p ~ ℓ^(p/3)
        if p <= 3:
            lags_theory = lags[::10]
            S_theory = lags_theory**(p/3) * S_p[5] / lags[5]**(p/3)
            ax1.loglog(lags_theory, S_theory, '--', color=colors[i],
                       linewidth=1, alpha=0.5)

    ax1.set_xlabel('Lag ℓ', fontsize=12)
    ax1.set_ylabel('S_p(ℓ)', fontsize=12)
    ax1.set_title('Velocity Structure Functions', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')

    # Plot 2: Structure functions for magnetic field
    ax2 = plt.subplot(3, 3, 2)

    for i, p in enumerate(orders):
        lags, B_p = compute_structure_function(B_field, p)
        ax2.loglog(lags, B_p, 'o-', color=colors[i], markersize=3,
                   linewidth=1.5, alpha=0.7, label=f'p={p}')

    ax2.set_xlabel('Lag ℓ', fontsize=12)
    ax2.set_ylabel('B_p(ℓ)', fontsize=12)
    ax2.set_title('Magnetic Structure Functions', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Scaling exponents ζ(p)
    ax3 = plt.subplot(3, 3, 3)
    orders_full = np.arange(1, 9)
    zeta_v = np.zeros(len(orders_full))
    zeta_B = np.zeros(len(orders_full))
    zeta_gaussian = np.zeros(len(orders_full))

    for i, p in enumerate(orders_full):
        lags, S_p_v = compute_structure_function(v_field, p)
        lags, S_p_B = compute_structure_function(B_field, p)
        lags, S_p_g = compute_structure_function(v_gaussian, p)

        zeta_v[i] = compute_scaling_exponent(lags, S_p_v)
        zeta_B[i] = compute_scaling_exponent(lags, S_p_B)
        zeta_gaussian[i] = compute_scaling_exponent(lags, S_p_g)

    # Kolmogorov prediction: ζ(p) = p/3
    zeta_kolm = orders_full / 3

    ax3.plot(orders_full, zeta_v, 'bo-', linewidth=2, markersize=6,
             label='Velocity (intermittent)', alpha=0.7)
    ax3.plot(orders_full, zeta_B, 'rs-', linewidth=2, markersize=6,
             label='Magnetic (intermittent)', alpha=0.7)
    ax3.plot(orders_full, zeta_gaussian, 'g^-', linewidth=2, markersize=6,
             label='Gaussian (no intermittency)', alpha=0.7)
    ax3.plot(orders_full, zeta_kolm, 'k--', linewidth=2,
             label='Kolmogorov: ζ = p/3', alpha=0.5)

    ax3.set_xlabel('Order p', fontsize=12)
    ax3.set_ylabel('Scaling Exponent ζ(p)', fontsize=12)
    ax3.set_title('Scaling Exponents vs Order', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Intermittency measure
    ax4 = plt.subplot(3, 3, 4)
    # Intermittency: deviation from Kolmogorov
    Delta_zeta_v = zeta_kolm - zeta_v
    Delta_zeta_B = zeta_kolm - zeta_B

    ax4.plot(orders_full, Delta_zeta_v, 'bo-', linewidth=2, markersize=6,
             label='Velocity', alpha=0.7)
    ax4.plot(orders_full, Delta_zeta_B, 'rs-', linewidth=2, markersize=6,
             label='Magnetic', alpha=0.7)
    ax4.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)

    ax4.set_xlabel('Order p', fontsize=12)
    ax4.set_ylabel('Δζ(p) = p/3 - ζ(p)', fontsize=12)
    ax4.set_title('Intermittency Correction', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Plot 5: PDF of velocity increments
    ax5 = plt.subplot(3, 3, 5)

    x_bins_v, pdf_v = compute_pdf(v_field)
    x_bins_B, pdf_B = compute_pdf(B_field)
    x_bins_g, pdf_g = compute_pdf(v_gaussian)

    x_theory = np.linspace(-5, 5, 100)
    pdf_theory = gaussian_pdf(x_theory)

    ax5.semilogy(x_bins_v, pdf_v, 'b-', linewidth=2,
                 label='Velocity (intermittent)', alpha=0.7)
    ax5.semilogy(x_bins_g, pdf_g, 'g-', linewidth=2,
                 label='Gaussian field', alpha=0.7)
    ax5.semilogy(x_theory, pdf_theory, 'k--', linewidth=2,
                 label='Gaussian', alpha=0.5)

    ax5.set_xlabel('δv / σ', fontsize=12)
    ax5.set_ylabel('PDF', fontsize=12)
    ax5.set_title('Velocity Increment PDF', fontsize=13, fontweight='bold')
    ax5.set_ylim([1e-4, 10])
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Plot 6: PDF of magnetic increments
    ax6 = plt.subplot(3, 3, 6)

    ax6.semilogy(x_bins_B, pdf_B, 'r-', linewidth=2,
                 label='Magnetic (intermittent)', alpha=0.7)
    ax6.semilogy(x_theory, pdf_theory, 'k--', linewidth=2,
                 label='Gaussian', alpha=0.5)

    ax6.set_xlabel('δB / σ', fontsize=12)
    ax6.set_ylabel('PDF', fontsize=12)
    ax6.set_title('Magnetic Increment PDF', fontsize=13, fontweight='bold')
    ax6.set_ylim([1e-4, 10])
    ax6.legend(loc='upper right', fontsize=10)
    ax6.grid(True, alpha=0.3)

    # Plot 7: Higher-order statistics
    ax7 = plt.subplot(3, 3, 7)

    # Compute statistics at different scales
    n_scales = 20
    scales = np.logspace(0, np.log10(N//4), n_scales).astype(int)
    kurt_v = np.zeros(n_scales)
    kurt_B = np.zeros(n_scales)
    skew_v = np.zeros(n_scales)

    for i, scale in enumerate(scales):
        delta_v = v_field[scale:] - v_field[:-scale]
        delta_B = B_field[scale:] - B_field[:-scale]
        kurt_v[i] = kurtosis(delta_v)
        kurt_B[i] = kurtosis(delta_B)
        skew_v[i] = skew(delta_v)

    ax7.semilogx(scales, kurt_v, 'b-', linewidth=2, markersize=5,
                 label='Kurtosis (velocity)', alpha=0.7)
    ax7.semilogx(scales, kurt_B, 'r-', linewidth=2, markersize=5,
                 label='Kurtosis (magnetic)', alpha=0.7)
    ax7.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5,
                label='Gaussian value')

    ax7.set_xlabel('Scale ℓ', fontsize=12)
    ax7.set_ylabel('Kurtosis', fontsize=12)
    ax7.set_title('Scale-Dependent Kurtosis', fontsize=13, fontweight='bold')
    ax7.legend(loc='upper right', fontsize=10)
    ax7.grid(True, alpha=0.3)

    # Plot 8: Skewness
    ax8 = plt.subplot(3, 3, 8)

    ax8.semilogx(scales, skew_v, 'b-', linewidth=2, alpha=0.7)
    ax8.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)

    ax8.set_xlabel('Scale ℓ', fontsize=12)
    ax8.set_ylabel('Skewness', fontsize=12)
    ax8.set_title('Scale-Dependent Skewness', fontsize=13, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # Plot 9: Summary text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    summary_text = f"""
    Intermittency Statistics
    ────────────────────────

    Velocity Field:
      • Kurtosis (small scale): {kurt_v[0]:.2f}
      • Skewness: {skew_v[0]:.2f}
      • ζ(3) deviation: {Delta_zeta_v[2]:.3f}

    Magnetic Field:
      • Kurtosis (small scale): {kurt_B[0]:.2f}
      • ζ(3) deviation: {Delta_zeta_B[2]:.3f}

    Key Observations:
      • Non-Gaussian PDFs
      • Scale-dependent statistics
      • ζ(p) < p/3 for p > 3
      • Enhanced small-scale fluctuations

    Models:
      • She-Leveque (1994)
      • Log-normal cascades
      • Multifractal formalism
    """

    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax9.transAxes)

    plt.suptitle('Structure Functions and Intermittency in MHD Turbulence',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('structure_functions.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print numerical results
    print("\n" + "="*60)
    print("Scaling Exponents ζ(p)")
    print("="*60)
    print(f"{'Order p':<10} {'Velocity':<12} {'Magnetic':<12} {'Kolmogorov':<12}")
    print("-"*60)
    for i, p in enumerate(orders_full):
        print(f"{p:<10} {zeta_v[i]:<12.4f} {zeta_B[i]:<12.4f} {zeta_kolm[i]:<12.4f}")

    print("\n" + "="*60)
    print("Higher-Order Statistics (small scales)")
    print("="*60)
    print(f"Velocity kurtosis: {kurt_v[0]:.3f} (Gaussian = 0)")
    print(f"Velocity skewness: {skew_v[0]:.3f} (Gaussian = 0)")
    print(f"Magnetic kurtosis: {kurt_B[0]:.3f} (Gaussian = 0)")
    print("\nInterpretation:")
    print("  Positive kurtosis => Heavy tails (intermittency)")
    print("  Negative skewness => Asymmetry (downward spikes)")


def main():
    """Main execution function."""
    print("="*60)
    print("Structure Functions and Intermittency Analysis")
    print("="*60)

    plot_structure_functions()

    print("\nPlot saved as 'structure_functions.png'")

    print("\n" + "="*60)
    print("Theoretical Background")
    print("="*60)
    print("\nKolmogorov 1941 (K41):")
    print("  S_p(ℓ) ~ ℓ^(p/3)  (self-similar scaling)")
    print("\nIntermittency Corrections:")
    print("  S_p(ℓ) ~ ℓ^ζ(p)  with ζ(p) < p/3 for p > 3")
    print("  Due to spatially concentrated dissipation")
    print("\nMHD Turbulence:")
    print("  Similar intermittency in velocity and magnetic fields")
    print("  Current sheets, magnetic islands enhance small-scale activity")
    print("\nReferences:")
    print("  - Kolmogorov (1941, 1962)")
    print("  - She & Leveque (1994)")
    print("  - Frisch (1995) - Turbulence textbook")


if __name__ == '__main__':
    main()
