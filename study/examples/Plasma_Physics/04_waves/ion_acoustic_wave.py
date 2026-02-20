#!/usr/bin/env python3
"""
Ion Acoustic Wave Dispersion and Damping

This script computes and visualizes ion acoustic wave properties including
dispersion relation, damping rates, and the effect of electron-to-ion
temperature ratio.

Key Physics:
- Sound speed: cs = sqrt(kTe/mi)
- Dispersion: ω = k*cs / sqrt(1 + k²λD²)
- Landau damping depends strongly on Te/Ti ratio
- Weakly damped only when Te >> Ti

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import erf

# Physical constants
EPS0 = 8.854187817e-12  # F/m
QE = 1.602176634e-19    # C
ME = 9.10938356e-31     # kg
MP = 1.672621898e-27    # kg (proton mass)
KB = 1.380649e-23       # J/K

def compute_sound_speed(Te, Ti, mi):
    """
    Compute ion acoustic sound speed.

    cs = sqrt((kTe + 3kTi) / mi) ≈ sqrt(kTe/mi) for Te >> Ti

    Parameters:
    -----------
    Te : float
        Electron temperature [eV]
    Ti : float
        Ion temperature [eV]
    mi : float
        Ion mass [kg]

    Returns:
    --------
    cs : float
        Sound speed [m/s]
    """
    Te_joule = Te * QE
    Ti_joule = Ti * QE

    # Full expression including ion pressure
    cs = np.sqrt((Te_joule + 3 * Ti_joule) / mi)

    return cs

def ion_acoustic_dispersion(k, ne, Te, Ti, mi):
    """
    Compute ion acoustic wave dispersion relation.

    ω = k*cs / sqrt(1 + k²λD²)

    Parameters:
    -----------
    k : array
        Wavenumber [rad/m]
    ne : float
        Electron density [m^-3]
    Te : float
        Electron temperature [eV]
    Ti : float
        Ion temperature [eV]
    mi : float
        Ion mass [kg]

    Returns:
    --------
    omega : array
        Angular frequency [rad/s]
    """
    # Compute sound speed
    cs = compute_sound_speed(Te, Ti, mi)

    # Debye length (using electron temperature)
    Te_joule = Te * QE
    lambda_D = np.sqrt(EPS0 * Te_joule / (ne * QE**2))

    # Dispersion relation
    omega = k * cs / np.sqrt(1 + k**2 * lambda_D**2)

    return omega

def ion_landau_damping(k, ne, Te, Ti, mi):
    """
    Compute ion Landau damping rate for ion acoustic waves.

    This is an approximate formula valid for Te/Ti >> 1.

    γ/ω ≈ -sqrt(π/8) * (me/mi)^(1/2) * (1 + Te/Ti)^(-3/2) * exp(-Te/(2Ti) - 3/2)

    Parameters:
    -----------
    k : array
        Wavenumber [rad/m]
    ne : float
        Electron density [m^-3]
    Te : float
        Electron temperature [eV]
    Ti : float
        Ion temperature [eV]
    mi : float
        Ion mass [kg]

    Returns:
    --------
    gamma : array
        Damping rate [rad/s]
    """
    # Compute frequency
    omega = ion_acoustic_dispersion(k, ne, Te, Ti, mi)

    # Temperature ratio
    tau = Te / Ti

    # Approximate damping rate (valid for tau >> 1)
    # γ/ω ≈ -sqrt(π/8) * (1/tau)^(3/2) * exp(-1/(2*tau) - 3/2)
    if tau > 1:
        gamma_over_omega = -np.sqrt(np.pi / 8) * tau**(-3/2) * np.exp(-tau/2 - 3/2)
    else:
        # Strong damping when Te ~ Ti
        gamma_over_omega = -1.0

    gamma = gamma_over_omega * omega

    return gamma

def plot_ion_acoustic_waves():
    """
    Create comprehensive visualization of ion acoustic wave properties.
    """
    # Plasma parameters
    ne = 1e18  # m^-3
    mi = MP    # Hydrogen ions

    # Different temperature ratios to explore
    tau_values = [1, 3, 10, 100]  # Te/Ti
    colors = ['red', 'orange', 'green', 'blue']

    # Create wavenumber array
    Te_ref = 10.0  # eV
    lambda_D_ref = np.sqrt(EPS0 * Te_ref * QE / (ne * QE**2))
    k = np.linspace(1e2, 10 / lambda_D_ref, 1000)

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Store data for different tau values
    omega_data = []
    gamma_data = []
    cs_data = []

    print("=" * 70)
    print("Ion Acoustic Wave Parameters")
    print("=" * 70)
    print(f"Electron density: {ne:.2e} m^-3")
    print(f"Ion mass: {mi/MP:.1f} × proton mass")
    print("-" * 70)

    for tau, color in zip(tau_values, colors):
        Te = 10.0  # eV
        Ti = Te / tau

        # Compute dispersion
        omega = ion_acoustic_dispersion(k, ne, Te, Ti, mi)
        gamma = ion_landau_damping(k, ne, Te, Ti, mi)
        cs = compute_sound_speed(Te, Ti, mi)

        omega_data.append(omega)
        gamma_data.append(gamma)
        cs_data.append(cs)

        print(f"Te/Ti = {tau:6.1f}: Te = {Te:5.1f} eV, Ti = {Ti:5.2f} eV, "
              f"cs = {cs/1e3:6.2f} km/s")

    print("=" * 70)

    # Plot 1: Dispersion relation ω(k)
    ax1 = fig.add_subplot(gs[0, :])
    for i, (tau, color) in enumerate(zip(tau_values, colors)):
        k_lambda = k * lambda_D_ref
        omega = omega_data[i]
        cs = cs_data[i]

        ax1.plot(k_lambda, omega / (k * cs),
                 color=color, linewidth=2, label=f'Te/Ti = {tau}')

    # Plot long wavelength limit (should approach 1)
    ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1,
                label='Long wavelength limit')

    ax1.set_xlabel(r'$k \lambda_D$', fontsize=12)
    ax1.set_ylabel(r'$\omega / (k c_s)$', fontsize=12)
    ax1.set_title('Ion Acoustic Wave Dispersion (Normalized)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_xlim([0, 10])
    ax1.set_ylim([0, 1.1])

    # Plot 2: Absolute frequency
    ax2 = fig.add_subplot(gs[1, 0])
    for i, (tau, color) in enumerate(zip(tau_values, colors)):
        k_lambda = k * lambda_D_ref
        omega = omega_data[i]

        ax2.plot(k_lambda, omega / (2 * np.pi * 1e9),
                 color=color, linewidth=2, label=f'Te/Ti = {tau}')

    ax2.set_xlabel(r'$k \lambda_D$', fontsize=12)
    ax2.set_ylabel('Frequency (GHz)', fontsize=12)
    ax2.set_title('Absolute Frequency', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, 10])

    # Plot 3: Phase velocity
    ax3 = fig.add_subplot(gs[1, 1])
    for i, (tau, color) in enumerate(zip(tau_values, colors)):
        k_lambda = k * lambda_D_ref
        omega = omega_data[i]
        cs = cs_data[i]
        vph = omega / k

        ax3.plot(k_lambda, vph / cs,
                 color=color, linewidth=2, label=f'Te/Ti = {tau}')

    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1,
                label='Sound speed')

    ax3.set_xlabel(r'$k \lambda_D$', fontsize=12)
    ax3.set_ylabel(r'$v_{ph} / c_s$', fontsize=12)
    ax3.set_title('Phase Velocity', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xlim([0, 10])

    # Plot 4: Damping rate (absolute)
    ax4 = fig.add_subplot(gs[2, 0])
    for i, (tau, color) in enumerate(zip(tau_values, colors)):
        if tau >= 3:  # Only plot for cases with weak damping
            k_lambda = k * lambda_D_ref
            gamma = gamma_data[i]

            ax4.plot(k_lambda, -gamma / (2 * np.pi * 1e9),
                     color=color, linewidth=2, label=f'Te/Ti = {tau}')

    ax4.set_xlabel(r'$k \lambda_D$', fontsize=12)
    ax4.set_ylabel('Damping Rate (GHz)', fontsize=12)
    ax4.set_title('Ion Landau Damping Rate (Te/Ti ≥ 3)',
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_xlim([0, 10])
    ax4.set_yscale('log')

    # Plot 5: Damping rate normalized
    ax5 = fig.add_subplot(gs[2, 1])
    for i, (tau, color) in enumerate(zip(tau_values, colors)):
        if tau >= 3:  # Only plot for cases with weak damping
            k_lambda = k * lambda_D_ref
            omega = omega_data[i]
            gamma = gamma_data[i]

            ax5.plot(k_lambda, -gamma / omega,
                     color=color, linewidth=2, label=f'Te/Ti = {tau}')

    ax5.set_xlabel(r'$k \lambda_D$', fontsize=12)
    ax5.set_ylabel(r'$|\gamma| / \omega$', fontsize=12)
    ax5.set_title('Normalized Damping Rate (Te/Ti ≥ 3)',
                  fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10)
    ax5.set_xlim([0, 10])
    ax5.set_yscale('log')

    plt.suptitle('Ion Acoustic Wave: Effect of Temperature Ratio Te/Ti',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('ion_acoustic_wave.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'ion_acoustic_wave.png'")

    plt.show()

if __name__ == "__main__":
    plot_ion_acoustic_waves()
