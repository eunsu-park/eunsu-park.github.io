#!/usr/bin/env python3
"""
Two-Fluid vs MHD Wave Dispersion

This script compares single-fluid MHD wave dispersion with two-fluid
corrections for parallel propagation in a magnetized plasma.

Key Physics:
- MHD Alfvén wave: ω = k·vA (linear dispersion)
- Two-fluid corrections at k·di ~ 1 (ion skin depth)
- Ion cyclotron wave and whistler branches
- Kinetic Alfvén wave at finite k_perp

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Physical constants
EPS0 = 8.854187817e-12  # F/m
QE = 1.602176634e-19    # C
ME = 9.10938356e-31     # kg
MP = 1.672621898e-27    # kg
C = 2.99792458e8        # m/s
MU0 = 4 * np.pi * 1e-7  # H/m

def compute_plasma_parameters(ne, B0, mi=MP):
    """
    Compute characteristic plasma parameters.

    Parameters:
    -----------
    ne : float
        Electron density [m^-3]
    B0 : float
        Magnetic field [T]
    mi : float
        Ion mass [kg]

    Returns:
    --------
    dict : Plasma parameters
    """
    # Alfvén speed
    vA = B0 / np.sqrt(MU0 * ne * mi)

    # Ion skin depth
    di = C / np.sqrt(ne * QE**2 / (mi * EPS0))

    # Cyclotron frequencies
    omega_ci = QE * B0 / mi
    omega_ce = QE * B0 / ME

    # Plasma frequencies
    omega_pi = np.sqrt(ne * QE**2 / (mi * EPS0))
    omega_pe = np.sqrt(ne * QE**2 / (ME * EPS0))

    # Ion Larmor radius (thermal)
    # Assume Ti = 1 keV for estimate
    Ti = 1000 * QE  # J
    vti = np.sqrt(2 * Ti / mi)
    rho_i = vti / omega_ci

    return {
        'vA': vA,
        'di': di,
        'rho_i': rho_i,
        'omega_ci': omega_ci,
        'omega_ce': omega_ce,
        'omega_pi': omega_pi,
        'omega_pe': omega_pe,
        'ne': ne,
        'B0': B0
    }

def mhd_alfven_dispersion(k, params):
    """
    MHD Alfvén wave: ω = k·vA.

    Parameters:
    -----------
    k : array
        Wavenumber [rad/m]
    params : dict
        Plasma parameters

    Returns:
    --------
    omega : array
        Angular frequency [rad/s]
    """
    return k * params['vA']

def two_fluid_parallel_dispersion(k, params):
    """
    Two-fluid dispersion for parallel propagation.

    Includes ion cyclotron and whistler branches.

    Approximate dispersion:
    ω² = k²vA² * (1 + k²di²) / (1 + k²di² + vA²/c²)

    Parameters:
    -----------
    k : array
        Wavenumber [rad/m]
    params : dict
        Plasma parameters

    Returns:
    --------
    omega : array
        Angular frequency [rad/s]
    """
    vA = params['vA']
    di = params['di']
    omega_ci = params['omega_ci']

    # Two-fluid correction
    k_di = k * di

    # Low frequency branch (ion cyclotron)
    # ω ≈ k·vA / sqrt(1 + k²di²)
    omega_ic = k * vA / np.sqrt(1 + k_di**2)

    return omega_ic

def whistler_branch(k, params):
    """
    High-frequency whistler branch in two-fluid theory.

    ω ≈ (k²vA²)/(ωci·di·k) for k·di >> 1

    Parameters:
    -----------
    k : array
        Wavenumber [rad/m]
    params : dict
        Plasma parameters

    Returns:
    --------
    omega : array
        Angular frequency [rad/s]
    """
    vA = params['vA']
    di = params['di']
    omega_ci = params['omega_ci']
    omega_ce = params['omega_ce']

    # Whistler approximation: ω ≈ k²vA²/(ωci) for k·di >> 1
    # More accurate: ω ≈ k²c²/(ωpe) * (ωce/ω)
    # Simple form:
    omega_w = (k * vA)**2 / (omega_ci + 1e-10)  # Avoid division by zero

    # Cap at electron cyclotron frequency
    omega_w = np.minimum(omega_w, 0.5 * omega_ce)

    return omega_w

def kinetic_alfven_dispersion(k_perp, k_parallel, params):
    """
    Kinetic Alfvén wave dispersion with finite k_perp.

    ω = k_∥·vA * sqrt(1 + k_perp²·ρi²)

    Parameters:
    -----------
    k_perp : array
        Perpendicular wavenumber [rad/m]
    k_parallel : float
        Parallel wavenumber [rad/m]
    params : dict
        Plasma parameters

    Returns:
    --------
    omega : array
        Angular frequency [rad/s]
    """
    vA = params['vA']
    rho_i = params['rho_i']

    # KAW dispersion
    omega = k_parallel * vA * np.sqrt(1 + (k_perp * rho_i)**2)

    return omega

def plot_two_fluid_waves():
    """
    Create comprehensive comparison of MHD vs two-fluid waves.
    """
    # Typical tokamak parameters
    ne = 1e19  # m^-3
    B0 = 2.0   # T
    mi = 2 * MP  # Deuterium

    params = compute_plasma_parameters(ne, B0, mi)

    print("=" * 70)
    print("Two-Fluid vs MHD Wave Comparison")
    print("=" * 70)
    print(f"Electron density: {params['ne']:.2e} m^-3")
    print(f"Magnetic field: {params['B0']:.2f} T")
    print(f"Alfvén speed: {params['vA']/1e6:.3f} × 10^6 m/s")
    print(f"Ion skin depth: {params['di']*100:.2f} cm")
    print(f"Ion Larmor radius (1 keV): {params['rho_i']*1000:.2f} mm")
    print(f"Ion cyclotron frequency: {params['omega_ci']/(2*np.pi)/1e6:.2f} MHz")
    print("=" * 70)

    # Create figure
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Wavenumber array
    k_min = 1e0
    k_max = 100 / params['di']
    k = np.logspace(np.log10(k_min), np.log10(k_max), 1000)

    # Compute dispersions
    omega_mhd = mhd_alfven_dispersion(k, params)
    omega_2f = two_fluid_parallel_dispersion(k, params)
    omega_whistler = whistler_branch(k, params)

    # Normalize
    k_di = k * params['di']
    omega_norm_mhd = omega_mhd / params['omega_ci']
    omega_norm_2f = omega_2f / params['omega_ci']
    omega_norm_w = omega_whistler / params['omega_ci']

    # Plot 1: Dispersion relation (full range)
    ax1 = fig.add_subplot(gs[0, :])

    ax1.loglog(k_di, omega_norm_mhd, 'r--', linewidth=2.5, label='MHD Alfvén')
    ax1.loglog(k_di, omega_norm_2f, 'b-', linewidth=2.5, label='Two-fluid (IC branch)')
    ax1.loglog(k_di, omega_norm_w, 'g-', linewidth=2.5, label='Whistler branch')

    # Mark transition region
    ax1.axvline(x=1, color='purple', linestyle=':', linewidth=2,
                label=r'$k \cdot d_i = 1$')
    ax1.axhline(y=1, color='orange', linestyle=':', linewidth=2,
                label=r'$\omega = \Omega_{ci}$')

    ax1.set_xlabel(r'$k \cdot d_i$', fontsize=12)
    ax1.set_ylabel(r'$\omega / \Omega_{ci}$', fontsize=12)
    ax1.set_title('Parallel Propagation: MHD vs Two-Fluid',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.set_xlim([1e-2, 1e2])
    ax1.set_ylim([1e-2, 1e3])

    # Plot 2: Phase velocity
    ax2 = fig.add_subplot(gs[1, 0])

    vph_mhd = omega_mhd / k
    vph_2f = omega_2f / k

    ax2.semilogx(k_di, vph_mhd / params['vA'], 'r--', linewidth=2, label='MHD')
    ax2.semilogx(k_di, vph_2f / params['vA'], 'b-', linewidth=2, label='Two-fluid')

    ax2.axvline(x=1, color='purple', linestyle=':', linewidth=2)
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1)

    ax2.set_xlabel(r'$k \cdot d_i$', fontsize=12)
    ax2.set_ylabel(r'$v_{ph} / v_A$', fontsize=12)
    ax2.set_title('Phase Velocity', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim([1e-2, 1e2])

    # Plot 3: Group velocity
    ax3 = fig.add_subplot(gs[1, 1])

    vg_2f = np.gradient(omega_2f, k)

    ax3.semilogx(k_di, params['vA'] * np.ones_like(k) / params['vA'],
                'r--', linewidth=2, label='MHD')
    ax3.semilogx(k_di, vg_2f / params['vA'], 'b-', linewidth=2, label='Two-fluid')

    ax3.axvline(x=1, color='purple', linestyle=':', linewidth=2)
    ax3.axhline(y=1, color='gray', linestyle='--', linewidth=1)

    ax3.set_xlabel(r'$k \cdot d_i$', fontsize=12)
    ax3.set_ylabel(r'$v_g / v_A$', fontsize=12)
    ax3.set_title('Group Velocity', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xlim([1e-2, 1e2])

    # Plot 4: Dispersion relation (linear scale, low k)
    ax4 = fig.add_subplot(gs[2, 0])

    k_low = np.linspace(0, 5 / params['di'], 500)
    omega_mhd_low = mhd_alfven_dispersion(k_low, params)
    omega_2f_low = two_fluid_parallel_dispersion(k_low, params)

    ax4.plot(k_low * params['di'], omega_mhd_low / params['omega_ci'],
            'r--', linewidth=2, label='MHD')
    ax4.plot(k_low * params['di'], omega_2f_low / params['omega_ci'],
            'b-', linewidth=2, label='Two-fluid')

    ax4.axvline(x=1, color='purple', linestyle=':', linewidth=2,
                label=r'$k \cdot d_i = 1$')

    ax4.set_xlabel(r'$k \cdot d_i$', fontsize=12)
    ax4.set_ylabel(r'$\omega / \Omega_{ci}$', fontsize=12)
    ax4.set_title('Low-k Regime (Linear Scale)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    # Plot 5: Kinetic Alfvén wave (perpendicular k)
    ax5 = fig.add_subplot(gs[2, 1])

    k_parallel_fixed = 1.0 / params['di']  # Fixed k_∥
    k_perp = np.linspace(0, 10 / params['rho_i'], 500)

    omega_kaw = kinetic_alfven_dispersion(k_perp, k_parallel_fixed, params)
    omega_mhd_perp = k_parallel_fixed * params['vA'] * np.ones_like(k_perp)

    ax5.plot(k_perp * params['rho_i'], omega_kaw / params['omega_ci'],
            'b-', linewidth=2, label='KAW')
    ax5.plot(k_perp * params['rho_i'], omega_mhd_perp / params['omega_ci'],
            'r--', linewidth=2, label='MHD (k_perp = 0)')

    ax5.axvline(x=1, color='green', linestyle=':', linewidth=2,
                label=r'$k_\perp \cdot \rho_i = 1$')

    ax5.set_xlabel(r'$k_\perp \cdot \rho_i$', fontsize=12)
    ax5.set_ylabel(r'$\omega / \Omega_{ci}$', fontsize=12)
    ax5.set_title(f'Kinetic Alfvén Wave (k∥·di = {k_parallel_fixed * params["di"]:.1f})',
                  fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10)

    plt.suptitle('Two-Fluid Corrections to MHD Wave Dispersion',
                 fontsize=16, fontweight='bold', y=0.997)

    plt.savefig('two_fluid_waves.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'two_fluid_waves.png'")
    print("\nKey findings:")
    print(f"  - MHD valid for k·di << 1")
    print(f"  - Two-fluid effects important at k·di ~ 1")
    print(f"  - KAW effects at k_perp·ρi ~ 1")

    plt.show()

if __name__ == "__main__":
    plot_two_fluid_waves()
