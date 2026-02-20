#!/usr/bin/env python3
"""
Velocity Distribution Functions

This script plots and compares various velocity distribution functions used
in plasma physics: Maxwellian, bi-Maxwellian, kappa, shifted Maxwellian,
and bump-on-tail distributions.

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k, m_p, m_e
from scipy.special import gamma


def maxwellian_1d(v, n, m, T):
    """
    1D Maxwellian distribution.

    f(v) = n * sqrt(m / (2πkT)) * exp(-m*v² / (2kT))

    Parameters:
    -----------
    v : array
        Velocity [m/s]
    n : float
        Density [m^-3]
    m : float
        Mass [kg]
    T : float
        Temperature [K]

    Returns:
    --------
    f : array
        Distribution function [s/m^4]
    """
    v_th = np.sqrt(2 * k * T / m)
    return n * np.sqrt(m / (2 * np.pi * k * T)) * np.exp(-v**2 / v_th**2)


def maxwellian_3d(v, n, m, T):
    """
    3D Maxwellian speed distribution (spherical).

    f(v) = n * (m / (2πkT))^(3/2) * exp(-m*v² / (2kT))

    Speed distribution: 4πv² * f(v)

    Parameters:
    -----------
    v : array
        Speed [m/s]
    n, m, T : float
        Density, mass, temperature

    Returns:
    --------
    f_speed : array
        Speed distribution [s/m^4]
    """
    v_th = np.sqrt(2 * k * T / m)
    f = n * (m / (2 * np.pi * k * T))**(3/2) * np.exp(-v**2 / v_th**2)
    return 4 * np.pi * v**2 * f


def bi_maxwellian(v_perp, v_para, n, m, T_perp, T_para):
    """
    Bi-Maxwellian distribution with different perpendicular and parallel temps.

    f(v_perp, v_para) = n * (m / (2π))^(3/2) / (T_perp * sqrt(T_para))
                         * exp(-m*v_perp² / (2*T_perp) - m*v_para² / (2*T_para))

    Parameters:
    -----------
    v_perp, v_para : array
        Perpendicular and parallel velocities [m/s]
    n, m : float
        Density and mass
    T_perp, T_para : float
        Perpendicular and parallel temperatures [K]

    Returns:
    --------
    f : array
        Distribution function [s/m^4]
    """
    v_th_perp = np.sqrt(2 * k * T_perp / m)
    v_th_para = np.sqrt(2 * k * T_para / m)

    return (n * (m / (2 * np.pi))**(3/2) /
            (T_perp * np.sqrt(T_para)) *
            np.exp(-v_perp**2 / v_th_perp**2 - v_para**2 / v_th_para**2))


def kappa_distribution_1d(v, n, m, T, kappa):
    """
    Kappa distribution (superthermal tails).

    f(v) ∝ [1 + v² / (κ * v_th²)]^(-(κ+1))

    Parameters:
    -----------
    v : array
        Velocity [m/s]
    n, m, T : float
        Density, mass, temperature
    kappa : float
        Kappa parameter (κ → ∞ recovers Maxwellian)

    Returns:
    --------
    f : array
        Distribution function [s/m^4]
    """
    v_th = np.sqrt(2 * k * T / m)

    # Normalization factor
    norm = (n * gamma(kappa + 1) /
            (np.sqrt(np.pi * kappa) * gamma(kappa - 0.5) * v_th))

    return norm * (1 + v**2 / (kappa * v_th**2))**(-kappa - 1)


def shifted_maxwellian(v, n, m, T, v_drift):
    """
    Shifted Maxwellian (beam).

    f(v) = n * sqrt(m / (2πkT)) * exp(-m*(v - v_drift)² / (2kT))

    Parameters:
    -----------
    v : array
        Velocity [m/s]
    n, m, T : float
        Density, mass, temperature
    v_drift : float
        Drift velocity [m/s]

    Returns:
    --------
    f : array
        Distribution function [s/m^4]
    """
    v_th = np.sqrt(2 * k * T / m)
    return n * np.sqrt(m / (2 * np.pi * k * T)) * np.exp(-(v - v_drift)**2 / v_th**2)


def bump_on_tail(v, n_bg, n_beam, m, T_bg, T_beam, v_beam):
    """
    Bump-on-tail: background Maxwellian + beam.

    f = f_background + f_beam

    Parameters:
    -----------
    v : array
        Velocity [m/s]
    n_bg, n_beam : float
        Background and beam densities
    m : float
        Mass
    T_bg, T_beam : float
        Background and beam temperatures
    v_beam : float
        Beam velocity

    Returns:
    --------
    f : array
        Distribution function [s/m^4]
    """
    f_bg = maxwellian_1d(v, n_bg, m, T_bg)
    f_beam = shifted_maxwellian(v, n_beam, m, T_beam, v_beam)
    return f_bg + f_beam


def plot_1d_distributions():
    """Plot 1D velocity distributions."""

    # Parameters
    n = 1e20  # m^-3
    m = m_e
    T = 1e4  # K (~ 1 eV)
    v_th = np.sqrt(2 * k * T / m)

    # Velocity array
    v = np.linspace(-5*v_th, 5*v_th, 500)

    # Different distributions
    f_max = maxwellian_1d(v, n, m, T)
    f_kappa_2 = kappa_distribution_1d(v, n, m, T, kappa=2)
    f_kappa_5 = kappa_distribution_1d(v, n, m, T, kappa=5)
    f_kappa_10 = kappa_distribution_1d(v, n, m, T, kappa=10)
    f_shifted = shifted_maxwellian(v, n, m, T, v_drift=2*v_th)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Linear scale
    ax1.plot(v / v_th, f_max * v_th, 'b-', linewidth=2.5, label='Maxwellian')
    ax1.plot(v / v_th, f_kappa_2 * v_th, 'r-', linewidth=2, label='κ = 2')
    ax1.plot(v / v_th, f_kappa_5 * v_th, 'g-', linewidth=2, label='κ = 5')
    ax1.plot(v / v_th, f_kappa_10 * v_th, 'm-', linewidth=2, label='κ = 10')

    ax1.set_xlabel('v / v_th', fontsize=12)
    ax1.set_ylabel('f(v) × v_th [normalized]', fontsize=12)
    ax1.set_title('Maxwellian vs Kappa Distributions', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-5, 5])

    # Log scale (shows superthermal tails)
    ax2.semilogy(v / v_th, f_max * v_th, 'b-', linewidth=2.5, label='Maxwellian')
    ax2.semilogy(v / v_th, f_kappa_2 * v_th, 'r-', linewidth=2, label='κ = 2')
    ax2.semilogy(v / v_th, f_kappa_5 * v_th, 'g-', linewidth=2, label='κ = 5')
    ax2.semilogy(v / v_th, f_kappa_10 * v_th, 'm-', linewidth=2, label='κ = 10')

    ax2.set_xlabel('v / v_th', fontsize=12)
    ax2.set_ylabel('f(v) × v_th [log scale]', fontsize=12)
    ax2.set_title('Superthermal Tails (Log Scale)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([1e-6, 1])

    plt.tight_layout()
    plt.savefig('distribution_1d_kappa.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_bump_on_tail():
    """Plot bump-on-tail distribution."""

    # Parameters
    n_bg = 1e20
    n_beam = 0.1 * n_bg  # 10% beam
    m = m_e
    T_bg = 1e4  # K
    T_beam = 0.5 * T_bg  # Beam is colder
    v_beam = 3 * np.sqrt(2 * k * T_bg / m)  # Beam at 3 v_th

    v_th_bg = np.sqrt(2 * k * T_bg / m)

    # Velocity array
    v = np.linspace(-2*v_th_bg, 6*v_th_bg, 1000)

    # Distributions
    f_bg = maxwellian_1d(v, n_bg, m, T_bg)
    f_beam = shifted_maxwellian(v, n_beam, m, T_beam, v_beam)
    f_total = bump_on_tail(v, n_bg, n_beam, m, T_bg, T_beam, v_beam)

    # df/dv (slope)
    dfdv = np.gradient(f_total, v)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Distribution
    ax1.plot(v / v_th_bg, f_bg * v_th_bg, 'b--', linewidth=2,
             label='Background', alpha=0.7)
    ax1.plot(v / v_th_bg, f_beam * v_th_bg, 'r--', linewidth=2,
             label='Beam', alpha=0.7)
    ax1.plot(v / v_th_bg, f_total * v_th_bg, 'k-', linewidth=2.5,
             label='Total (bump-on-tail)')

    ax1.fill_between(v / v_th_bg, 0, f_total * v_th_bg, alpha=0.2, color='green',
                     label='Unstable region')

    ax1.set_xlabel('v / v_th', fontsize=12)
    ax1.set_ylabel('f(v) × v_th', fontsize=12)
    ax1.set_title('Bump-on-Tail Distribution', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Slope df/dv
    ax2.plot(v / v_th_bg, dfdv * v_th_bg**2, 'b-', linewidth=2.5)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2, label='df/dv = 0')

    # Highlight positive slope region (unstable)
    positive_slope = dfdv > 0
    ax2.fill_between(v / v_th_bg, 0, dfdv * v_th_bg**2,
                     where=positive_slope, alpha=0.3, color='red',
                     label='Positive slope (unstable)')

    ax2.set_xlabel('v / v_th', fontsize=12)
    ax2.set_ylabel('df/dv × v_th²', fontsize=12)
    ax2.set_title('Slope: Positive Slope → Instability', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bump_on_tail_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_2d_biMaxwellian():
    """Plot 2D bi-Maxwellian distribution."""

    # Parameters
    n = 1e20
    m = m_e
    T_perp = 2e4  # K (perpendicular hotter)
    T_para = 1e4  # K

    v_th_perp = np.sqrt(2 * k * T_perp / m)
    v_th_para = np.sqrt(2 * k * T_para / m)

    # 2D velocity grid
    v_perp = np.linspace(-4*v_th_perp, 4*v_th_perp, 200)
    v_para = np.linspace(-4*v_th_para, 4*v_th_para, 200)
    V_perp, V_para = np.meshgrid(v_perp, v_para)

    # Bi-Maxwellian
    f_bi = bi_maxwellian(V_perp, V_para, n, m, T_perp, T_para)

    # Isotropic Maxwellian for comparison (T = average)
    T_avg = (2 * T_perp + T_para) / 3
    v_th_avg = np.sqrt(2 * k * T_avg / m)
    f_iso = bi_maxwellian(V_perp, V_para, n, m, T_avg, T_avg)

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Bi-Maxwellian contours
    levels = np.logspace(np.log10(f_bi.max()) - 6, np.log10(f_bi.max()), 15)
    cs1 = ax1.contourf(V_para / v_th_para, V_perp / v_th_perp, f_bi,
                       levels=levels, cmap='hot', norm=plt.matplotlib.colors.LogNorm())
    ax1.contour(V_para / v_th_para, V_perp / v_th_perp, f_bi,
                levels=levels[::2], colors='white', linewidths=0.5, alpha=0.5)

    ax1.set_xlabel('v_∥ / v_th,∥', fontsize=12)
    ax1.set_ylabel('v_⊥ / v_th,⊥', fontsize=12)
    ax1.set_title(f'Bi-Maxwellian\n(T_⊥ = {T_perp/1e4:.1f}×10⁴ K, T_∥ = {T_para/1e4:.1f}×10⁴ K)',
                  fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    plt.colorbar(cs1, ax=ax1, label='f(v_⊥, v_∥)')

    # Plot 2: Isotropic Maxwellian
    cs2 = ax2.contourf(V_para / v_th_avg, V_perp / v_th_avg, f_iso,
                       levels=levels, cmap='hot', norm=plt.matplotlib.colors.LogNorm())
    ax2.contour(V_para / v_th_avg, V_perp / v_th_avg, f_iso,
                levels=levels[::2], colors='white', linewidths=0.5, alpha=0.5)

    ax2.set_xlabel('v_∥ / v_th', fontsize=12)
    ax2.set_ylabel('v_⊥ / v_th', fontsize=12)
    ax2.set_title(f'Isotropic Maxwellian\n(T = {T_avg/1e4:.1f}×10⁴ K)',
                  fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    plt.colorbar(cs2, ax=ax2, label='f(v_⊥, v_∥)')

    # Plot 3: 1D cuts
    idx_para_0 = np.argmin(np.abs(v_para))
    idx_perp_0 = np.argmin(np.abs(v_perp))

    f_cut_para = f_bi[:, idx_perp_0]  # v_perp = 0 cut
    f_cut_perp = f_bi[idx_para_0, :]  # v_para = 0 cut

    ax3.semilogy(v_para / v_th_para, f_cut_para, 'b-', linewidth=2.5,
                 label='f(v_∥, v_⊥=0)')
    ax3.semilogy(v_perp / v_th_perp, f_cut_perp, 'r-', linewidth=2.5,
                 label='f(v_∥=0, v_⊥)')

    ax3.set_xlabel('v / v_th', fontsize=12)
    ax3.set_ylabel('f(v)', fontsize=12)
    ax3.set_title('1D Cuts Through Distribution', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bi_maxwellian_2d.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_3d_speed_distribution():
    """Plot 3D speed distributions."""

    # Parameters
    n = 1e20
    m = m_e
    T = 1e4  # K
    v_th = np.sqrt(2 * k * T / m)

    # Speed array (only positive)
    v = np.linspace(0, 5*v_th, 500)

    # 3D speed distributions
    f_maxwell = maxwellian_3d(v, n, m, T)
    f_kappa_2 = 4 * np.pi * v**2 * kappa_distribution_1d(v, n, m, T, kappa=2)
    f_kappa_5 = 4 * np.pi * v**2 * kappa_distribution_1d(v, n, m, T, kappa=5)

    # Most probable, mean, and RMS speeds for Maxwellian
    v_mp = np.sqrt(2 * k * T / m)  # Most probable
    v_mean = np.sqrt(8 * k * T / (np.pi * m))  # Mean
    v_rms = np.sqrt(3 * k * T / m)  # RMS

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Linear scale
    ax1.plot(v / v_th, f_maxwell, 'b-', linewidth=2.5, label='Maxwellian')
    ax1.plot(v / v_th, f_kappa_2, 'r-', linewidth=2, label='κ = 2')
    ax1.plot(v / v_th, f_kappa_5, 'g-', linewidth=2, label='κ = 5')

    # Mark characteristic speeds
    ax1.axvline(v_mp / v_th, color='cyan', linestyle='--', linewidth=2,
                label=f'v_mp = {v_mp/v_th:.2f} v_th')
    ax1.axvline(v_mean / v_th, color='orange', linestyle='--', linewidth=2,
                label=f'v_mean = {v_mean/v_th:.2f} v_th')
    ax1.axvline(v_rms / v_th, color='magenta', linestyle='--', linewidth=2,
                label=f'v_rms = {v_rms/v_th:.2f} v_th')

    ax1.set_xlabel('v / v_th', fontsize=12)
    ax1.set_ylabel('4πv² f(v)', fontsize=12)
    ax1.set_title('3D Speed Distribution', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Log scale
    ax2.semilogy(v / v_th, f_maxwell, 'b-', linewidth=2.5, label='Maxwellian')
    ax2.semilogy(v / v_th, f_kappa_2, 'r-', linewidth=2, label='κ = 2')
    ax2.semilogy(v / v_th, f_kappa_5, 'g-', linewidth=2, label='κ = 5')

    ax2.set_xlabel('v / v_th', fontsize=12)
    ax2.set_ylabel('4πv² f(v) [log]', fontsize=12)
    ax2.set_title('Speed Distribution (Log Scale)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([1e-2, f_maxwell.max() * 2])

    plt.tight_layout()
    plt.savefig('speed_distribution_3d.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("VELOCITY DISTRIBUTION FUNCTIONS IN PLASMAS")
    print("="*80 + "\n")

    print("Generating plots...")
    print("  1. 1D distributions (Maxwellian vs Kappa)...")
    plot_1d_distributions()

    print("  2. Bump-on-tail distribution...")
    plot_bump_on_tail()

    print("  3. 2D bi-Maxwellian distribution...")
    plot_2d_biMaxwellian()

    print("  4. 3D speed distributions...")
    plot_3d_speed_distribution()

    print("\nDone! Generated 4 figures:")
    print("  - distribution_1d_kappa.png")
    print("  - bump_on_tail_distribution.png")
    print("  - bi_maxwellian_2d.png")
    print("  - speed_distribution_3d.png")
