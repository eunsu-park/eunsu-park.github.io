#!/usr/bin/env python3
"""
Langmuir Wave Dispersion Relation

This script computes and visualizes the electrostatic Langmuir wave dispersion
relation in both cold and warm (kinetic) plasmas. It demonstrates the Bohm-Gross
dispersion relation and shows where Landau damping becomes important.

Key Physics:
- Cold plasma: ω = ωpe (no dependence on k)
- Warm plasma (Bohm-Gross): ω² = ωpe² + 3k²vth²
- Landau damping becomes significant when k*λD ~ 1

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
KB = 1.380649e-23       # J/K

def compute_plasma_parameters(ne, Te):
    """
    Compute fundamental plasma parameters.

    Parameters:
    -----------
    ne : float
        Electron density [m^-3]
    Te : float
        Electron temperature [eV]

    Returns:
    --------
    dict : Dictionary containing plasma parameters
    """
    # Convert temperature to SI units
    Te_joule = Te * QE

    # Electron plasma frequency
    omega_pe = np.sqrt(ne * QE**2 / (ME * EPS0))
    f_pe = omega_pe / (2 * np.pi)

    # Electron thermal velocity
    vth = np.sqrt(2 * Te_joule / ME)

    # Debye length
    lambda_D = np.sqrt(EPS0 * Te_joule / (ne * QE**2))

    return {
        'omega_pe': omega_pe,
        'f_pe': f_pe,
        'vth': vth,
        'lambda_D': lambda_D,
        'ne': ne,
        'Te': Te
    }

def bohm_gross_dispersion(k, params):
    """
    Compute Bohm-Gross dispersion relation for Langmuir waves.

    ω² = ωpe² + 3k²vth²

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
    omega_pe = params['omega_pe']
    vth = params['vth']

    omega_sq = omega_pe**2 + 3 * k**2 * vth**2
    return np.sqrt(omega_sq)

def cold_plasma_dispersion(k, params):
    """
    Cold plasma dispersion: ω = ωpe (constant).

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
    omega_pe = params['omega_pe']
    return omega_pe * np.ones_like(k)

def phase_velocity(omega, k):
    """
    Compute phase velocity vph = ω/k.

    Parameters:
    -----------
    omega : array
        Angular frequency [rad/s]
    k : array
        Wavenumber [rad/m]

    Returns:
    --------
    vph : array
        Phase velocity [m/s]
    """
    return omega / k

def group_velocity(omega, k):
    """
    Compute group velocity vg = dω/dk numerically.

    Parameters:
    -----------
    omega : array
        Angular frequency [rad/s]
    k : array
        Wavenumber [rad/m]

    Returns:
    --------
    vg : array
        Group velocity [m/s]
    """
    # Use central differences for interior points
    vg = np.gradient(omega, k)
    return vg

def plot_langmuir_dispersion():
    """
    Create comprehensive visualization of Langmuir wave dispersion.
    """
    # Define plasma parameters for a typical laboratory plasma
    ne = 1e18  # m^-3 (10^12 cm^-3)
    Te = 10.0  # eV

    params = compute_plasma_parameters(ne, Te)

    # Print plasma parameters
    print("=" * 60)
    print("Plasma Parameters")
    print("=" * 60)
    print(f"Electron density: {params['ne']:.2e} m^-3")
    print(f"Electron temperature: {params['Te']:.1f} eV")
    print(f"Electron plasma frequency: {params['f_pe']/1e9:.3f} GHz")
    print(f"Thermal velocity: {params['vth']/1e6:.3f} × 10^6 m/s")
    print(f"Debye length: {params['lambda_D']*1e6:.3f} μm")
    print("=" * 60)

    # Create wavenumber array
    # Range from small k to k ~ 10/λD
    k_min = 1e3  # rad/m
    k_max = 10 / params['lambda_D']
    k = np.linspace(k_min, k_max, 1000)

    # Compute dispersion relations
    omega_warm = bohm_gross_dispersion(k, params)
    omega_cold = cold_plasma_dispersion(k, params)

    # Compute velocities
    vph_warm = phase_velocity(omega_warm, k)
    vg_warm = group_velocity(omega_warm, k)

    # Identify Landau damping region (k*λD ~ 0.5 to 1.5)
    k_lambda_D = k * params['lambda_D']
    landau_mask = (k_lambda_D >= 0.5) & (k_lambda_D <= 1.5)

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Dispersion relation ω(k)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(k * params['lambda_D'], omega_warm / params['omega_pe'],
             'b-', linewidth=2, label='Warm plasma (Bohm-Gross)')
    ax1.plot(k * params['lambda_D'], omega_cold / params['omega_pe'],
             'r--', linewidth=2, label='Cold plasma')

    # Highlight Landau damping region
    ax1.axvspan(0.5, 1.5, alpha=0.2, color='yellow',
                label='Strong Landau damping (k·λD ~ 1)')

    ax1.set_xlabel(r'$k \lambda_D$', fontsize=12)
    ax1.set_ylabel(r'$\omega / \omega_{pe}$', fontsize=12)
    ax1.set_title('Langmuir Wave Dispersion Relation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, 10])

    # Plot 2: Phase velocity
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(k * params['lambda_D'], vph_warm / params['vth'],
             'b-', linewidth=2, label='Warm plasma')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Cold plasma (∞)')

    # Highlight Landau damping region
    ax2.axvspan(0.5, 1.5, alpha=0.2, color='yellow')

    ax2.set_xlabel(r'$k \lambda_D$', fontsize=12)
    ax2.set_ylabel(r'$v_{ph} / v_{th}$', fontsize=12)
    ax2.set_title('Phase Velocity', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, 10])
    ax2.set_ylim([0, 20])

    # Plot 3: Group velocity
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(k * params['lambda_D'], vg_warm / params['vth'],
             'b-', linewidth=2, label='Warm plasma')
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Cold plasma')

    # Highlight Landau damping region
    ax3.axvspan(0.5, 1.5, alpha=0.2, color='yellow')

    ax3.set_xlabel(r'$k \lambda_D$', fontsize=12)
    ax3.set_ylabel(r'$v_g / v_{th}$', fontsize=12)
    ax3.set_title('Group Velocity', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xlim([0, 10])

    # Plot 4: Comparison in frequency space
    ax4 = fig.add_subplot(gs[2, 0])
    f_warm = omega_warm / (2 * np.pi * params['f_pe'])
    f_cold = omega_cold / (2 * np.pi * params['f_pe'])

    ax4.plot(k * params['lambda_D'], f_warm,
             'b-', linewidth=2, label='Warm plasma')
    ax4.plot(k * params['lambda_D'], f_cold,
             'r--', linewidth=2, label='Cold plasma')

    # Highlight Landau damping region
    ax4.axvspan(0.5, 1.5, alpha=0.2, color='yellow')

    ax4.set_xlabel(r'$k \lambda_D$', fontsize=12)
    ax4.set_ylabel(r'$f / f_{pe}$', fontsize=12)
    ax4.set_title('Frequency Normalization', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_xlim([0, 10])

    # Plot 5: Relative correction
    ax5 = fig.add_subplot(gs[2, 1])
    correction = (omega_warm - omega_cold) / omega_cold * 100

    ax5.plot(k * params['lambda_D'], correction,
             'g-', linewidth=2)

    # Highlight Landau damping region
    ax5.axvspan(0.5, 1.5, alpha=0.2, color='yellow')

    ax5.set_xlabel(r'$k \lambda_D$', fontsize=12)
    ax5.set_ylabel('Correction (%)', fontsize=12)
    ax5.set_title('Kinetic Correction to Cold Plasma Frequency',
                  fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 10])

    plt.suptitle('Langmuir Wave Characteristics in Warm vs Cold Plasma',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('langmuir_wave_dispersion.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'langmuir_wave_dispersion.png'")

    plt.show()

if __name__ == "__main__":
    plot_langmuir_dispersion()
