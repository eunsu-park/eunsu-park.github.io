#!/usr/bin/env python3
"""
Collision Frequencies and Transport in Plasmas

This script calculates and visualizes collision frequencies, Coulomb logarithm,
Spitzer resistivity, and mean free paths for various plasma conditions.

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, m_p, epsilon_0, k, c


def coulomb_logarithm(n_e, T_e, Z=1):
    """
    Calculate the Coulomb logarithm.

    Parameters:
    -----------
    n_e : float or array
        Electron density [m^-3]
    T_e : float or array
        Electron temperature [eV]
    Z : int
        Ion charge number

    Returns:
    --------
    float or array : Coulomb logarithm (dimensionless)
    """
    # Two regimes based on temperature
    # High temperature (T_e > 10 eV): classical formula
    # Low temperature: quantum effects important

    ln_Lambda = np.where(
        T_e > 10,
        24 - np.log(np.sqrt(n_e * 1e-6) / T_e),  # High T
        23 - np.log(np.sqrt(n_e * 1e-6) * Z / T_e**1.5)  # Low T
    )

    # Enforce lower bound
    ln_Lambda = np.maximum(ln_Lambda, 2.0)

    return ln_Lambda


def electron_electron_collision_freq(n_e, T_e):
    """
    Electron-electron collision frequency.

    Parameters:
    -----------
    n_e : float or array
        Electron density [m^-3]
    T_e : float or array
        Electron temperature [eV]

    Returns:
    --------
    float or array : ν_ee [Hz]
    """
    T_J = T_e * e
    ln_Lambda = coulomb_logarithm(n_e, T_e)

    # Formula from NRL Plasma Formulary
    nu_ee = (n_e * e**4 * ln_Lambda) / (
        12 * np.pi**1.5 * epsilon_0**2 * m_e**0.5 * T_J**1.5
    )

    return nu_ee


def electron_ion_collision_freq(n_e, T_e, Z=1):
    """
    Electron-ion collision frequency (momentum transfer).

    Parameters:
    -----------
    n_e : float or array
        Electron density [m^-3]
    T_e : float or array
        Electron temperature [eV]
    Z : int
        Ion charge number

    Returns:
    --------
    float or array : ν_ei [Hz]
    """
    T_J = T_e * e
    ln_Lambda = coulomb_logarithm(n_e, T_e, Z)

    # Spitzer formula
    nu_ei = (Z * n_e * e**4 * ln_Lambda) / (
        12 * np.pi**1.5 * epsilon_0**2 * m_e**0.5 * T_J**1.5
    )

    return nu_ei


def ion_ion_collision_freq(n_i, T_i, A=1, Z=1):
    """
    Ion-ion collision frequency.

    Parameters:
    -----------
    n_i : float or array
        Ion density [m^-3]
    T_i : float or array
        Ion temperature [eV]
    A : float
        Ion mass number
    Z : int
        Ion charge number

    Returns:
    --------
    float or array : ν_ii [Hz]
    """
    T_J = T_i * e
    m_i = A * m_p
    ln_Lambda = coulomb_logarithm(n_i, T_i, Z)

    nu_ii = (Z**4 * n_i * e**4 * ln_Lambda) / (
        12 * np.pi**1.5 * epsilon_0**2 * m_i**0.5 * T_J**1.5
    )

    return nu_ii


def spitzer_resistivity(n_e, T_e, Z=1):
    """
    Spitzer resistivity (classical).

    Parameters:
    -----------
    n_e : float or array
        Electron density [m^-3]
    T_e : float or array
        Electron temperature [eV]
    Z : int
        Ion charge number

    Returns:
    --------
    float or array : η [Ω·m]
    """
    T_J = T_e * e
    ln_Lambda = coulomb_logarithm(n_e, T_e, Z)

    # Spitzer formula with numerical factor
    eta = (Z * m_e * e**2 * ln_Lambda) / (
        12 * np.pi**1.5 * epsilon_0**2 * T_J**1.5
    )

    return eta


def mean_free_path(n_e, T_e):
    """
    Electron mean free path.

    Parameters:
    -----------
    n_e : float or array
        Electron density [m^-3]
    T_e : float or array
        Electron temperature [eV]

    Returns:
    --------
    float or array : λ_mfp [m]
    """
    T_J = T_e * e
    v_th = np.sqrt(2 * T_J / m_e)
    nu_ei = electron_ion_collision_freq(n_e, T_e)

    return v_th / nu_ei


def plot_collision_frequencies():
    """Plot collision frequencies as function of temperature."""

    n_e = 1e20  # m^-3 (tokamak)
    T_range = np.logspace(0, 5, 200)  # 1 eV to 100 keV

    # Calculate frequencies
    nu_ee = electron_electron_collision_freq(n_e, T_range)
    nu_ei = electron_ion_collision_freq(n_e, T_range)
    nu_ii = ion_ion_collision_freq(n_e, T_range)

    # Plasma frequency for reference
    omega_pe = np.sqrt(n_e * e**2 / (epsilon_0 * m_e))
    f_pe = omega_pe / (2 * np.pi)

    # Gyrofrequency (assume B = 5 T)
    B = 5.0  # Tesla
    omega_ce = e * B / m_e
    f_ce = omega_ce / (2 * np.pi)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.loglog(T_range, nu_ee, 'b-', linewidth=2.5, label='ν_ee (e-e collision)')
    ax.loglog(T_range, nu_ei, 'r-', linewidth=2.5, label='ν_ei (e-i collision)')
    ax.loglog(T_range, nu_ii, 'g-', linewidth=2.5, label='ν_ii (i-i collision)')

    # Reference frequencies
    ax.axhline(f_pe, color='purple', linestyle='--', linewidth=2,
               label=f'f_pe = {f_pe:.2e} Hz')
    ax.axhline(f_ce, color='orange', linestyle='--', linewidth=2,
               label=f'f_ce = {f_ce:.2e} Hz (B=5T)')

    ax.set_xlabel('Electron Temperature [eV]', fontsize=13)
    ax.set_ylabel('Collision Frequency [Hz]', fontsize=13)
    ax.set_title(f'Collision Frequencies vs Temperature\n(n_e = {n_e:.1e} m⁻³)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # Add regime annotations
    ax.text(2, 1e6, 'Collisional\nRegime', fontsize=11, color='red',
            ha='left', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.text(1e4, 1e6, 'Collisionless\nRegime', fontsize=11, color='blue',
            ha='left', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig('collision_frequencies.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_coulomb_logarithm():
    """Plot Coulomb logarithm for various conditions."""

    # Temperature range
    T_range = np.logspace(-1, 5, 100)  # 0.1 eV to 100 keV

    # Different densities
    densities = [1e14, 1e16, 1e18, 1e20, 1e22]  # m^-3

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(densities)))

    for n_e, color in zip(densities, colors):
        ln_Lambda = coulomb_logarithm(n_e * np.ones_like(T_range), T_range)
        ax.semilogx(T_range, ln_Lambda, linewidth=2.5, color=color,
                    label=f'n_e = {n_e:.1e} m⁻³')

    ax.set_xlabel('Temperature [eV]', fontsize=13)
    ax.set_ylabel('Coulomb Logarithm ln Λ', fontsize=13)
    ax.set_title('Coulomb Logarithm vs Temperature and Density',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 30])

    plt.tight_layout()
    plt.savefig('coulomb_logarithm.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_spitzer_resistivity():
    """Plot Spitzer resistivity vs temperature."""

    # Temperature range
    T_range = np.logspace(0, 5, 200)  # 1 eV to 100 keV

    # Different densities
    densities = [1e18, 1e19, 1e20, 1e21]  # m^-3

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(densities)))

    for n_e, color in zip(densities, colors):
        eta = spitzer_resistivity(n_e * np.ones_like(T_range), T_range)
        ax.loglog(T_range, eta, linewidth=2.5, color=color,
                  label=f'n_e = {n_e:.1e} m⁻³')

    # Copper resistivity for reference
    eta_copper = 1.7e-8  # Ω·m at room temperature
    ax.axhline(eta_copper, color='brown', linestyle='--', linewidth=2,
               label='Copper (room temp)')

    ax.set_xlabel('Temperature [eV]', fontsize=13)
    ax.set_ylabel('Resistivity η [Ω·m]', fontsize=13)
    ax.set_title('Spitzer Resistivity vs Temperature',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('spitzer_resistivity.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_mean_free_path():
    """Plot mean free path vs temperature."""

    # Temperature range
    T_range = np.logspace(0, 5, 200)  # 1 eV to 100 keV

    # Different densities
    densities = [1e18, 1e19, 1e20, 1e21]  # m^-3

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.cool(np.linspace(0.1, 0.9, len(densities)))

    for n_e, color in zip(densities, colors):
        mfp = mean_free_path(n_e * np.ones_like(T_range), T_range)
        ax.loglog(T_range, mfp, linewidth=2.5, color=color,
                  label=f'n_e = {n_e:.1e} m⁻³')

    ax.set_xlabel('Temperature [eV]', fontsize=13)
    ax.set_ylabel('Mean Free Path λ_mfp [m]', fontsize=13)
    ax.set_title('Electron Mean Free Path vs Temperature',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Add reference lines
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='1 m')
    ax.axhline(1e-3, color='gray', linestyle=':', alpha=0.5, label='1 mm')

    plt.tight_layout()
    plt.savefig('mean_free_path.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_timescale_comparison():
    """Compare collision times with other plasma timescales."""

    n_e = 1e20  # m^-3
    B = 5.0  # Tesla
    T_range = np.logspace(0, 5, 200)  # 1 eV to 100 keV

    # Collision time
    nu_ei = electron_ion_collision_freq(n_e, T_range)
    tau_coll = 1 / nu_ei

    # Plasma period
    omega_pe = np.sqrt(n_e * e**2 / (epsilon_0 * m_e))
    tau_pe = 2 * np.pi / omega_pe

    # Gyroperiod
    omega_ce = e * B / m_e
    tau_ce = 2 * np.pi / omega_ce

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.loglog(T_range, tau_coll, 'r-', linewidth=3, label='τ_coll = 1/ν_ei')
    ax.axhline(tau_pe, color='blue', linestyle='--', linewidth=2,
               label=f'τ_pe = 2π/ω_pe = {tau_pe:.2e} s')
    ax.axhline(tau_ce, color='green', linestyle='--', linewidth=2,
               label=f'τ_ce = 2π/ω_ce = {tau_ce:.2e} s (B=5T)')

    # Shaded regions
    ax.fill_between(T_range, 1e-15, tau_pe, alpha=0.2, color='blue',
                    label='τ < τ_pe (plasma oscillation)')
    ax.fill_between(T_range, tau_pe, tau_ce, alpha=0.2, color='green',
                    label='τ_pe < τ < τ_ce (gyration)')

    ax.set_xlabel('Electron Temperature [eV]', fontsize=13)
    ax.set_ylabel('Time [s]', fontsize=13)
    ax.set_title(f'Plasma Timescales Comparison\n(n_e = {n_e:.1e} m⁻³, B = {B} T)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e-15, 1e0])

    # Add annotations
    ax.text(10, 1e-6, 'Collisional', fontsize=12, color='red', fontweight='bold')
    ax.text(1e4, 1e-6, 'Collisionless', fontsize=12, color='blue', fontweight='bold')

    plt.tight_layout()
    plt.savefig('timescale_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_example_calculations():
    """Print example calculations for different plasmas."""

    print("\n" + "="*90)
    print("COLLISION FREQUENCY CALCULATIONS FOR VARIOUS PLASMAS")
    print("="*90 + "\n")

    plasmas = [
        ('Tokamak Core', 1e20, 10e3, 5.0),
        ('Solar Corona', 1e14, 100, 1e-4),
        ('Ionosphere', 1e11, 0.1, 50e-6),
        ('Neon Sign', 1e16, 2, 0.0),
    ]

    for name, n_e, T_e, B in plasmas:
        print(f"{name}:")
        print(f"  n_e = {n_e:.2e} m^-3, T_e = {T_e:.2e} eV, B = {B} T")

        ln_Lambda = coulomb_logarithm(n_e, T_e)
        nu_ei = electron_ion_collision_freq(n_e, T_e)
        eta = spitzer_resistivity(n_e, T_e)
        mfp = mean_free_path(n_e, T_e)

        omega_pe = np.sqrt(n_e * e**2 / (epsilon_0 * m_e))
        tau_pe = 1 / omega_pe

        print(f"  Coulomb logarithm: {ln_Lambda:.2f}")
        print(f"  Collision freq ν_ei: {nu_ei:.3e} Hz")
        print(f"  Collision time τ_coll: {1/nu_ei:.3e} s")
        print(f"  Plasma period τ_pe: {tau_pe:.3e} s")
        print(f"  Collisionality ν/ω_pe: {nu_ei * tau_pe:.3e}")
        print(f"  Spitzer resistivity: {eta:.3e} Ω·m")
        print(f"  Mean free path: {mfp:.3e} m")

        if B > 0:
            omega_ce = e * B / m_e
            tau_ce = 1 / omega_ce
            print(f"  Gyroperiod τ_ce: {tau_ce:.3e} s")
            print(f"  ν_ei/ω_ce: {nu_ei / omega_ce:.3e}")

        print()


if __name__ == '__main__':
    print("\n" + "="*90)
    print("COLLISION FREQUENCIES AND TRANSPORT IN PLASMAS")
    print("="*90)

    # Print example calculations
    print_example_calculations()

    # Generate plots
    print("Generating plots...")
    print("  1. Collision frequencies vs temperature...")
    plot_collision_frequencies()

    print("  2. Coulomb logarithm...")
    plot_coulomb_logarithm()

    print("  3. Spitzer resistivity...")
    plot_spitzer_resistivity()

    print("  4. Mean free path...")
    plot_mean_free_path()

    print("  5. Timescale comparison...")
    plot_timescale_comparison()

    print("\nDone! Generated 5 figures:")
    print("  - collision_frequencies.png")
    print("  - coulomb_logarithm.png")
    print("  - spitzer_resistivity.png")
    print("  - mean_free_path.png")
    print("  - timescale_comparison.png")
