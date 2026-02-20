#!/usr/bin/env python3
"""
Plasma Parameters Calculator and Visualization

This script computes fundamental plasma parameters for various plasma regimes
and visualizes the parameter space on a density-temperature diagram.

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, m_p, epsilon_0, k, c, mu_0

# Physical constants
m_i = m_p  # Assume proton mass for ions


def compute_plasma_parameters(n_e, T_e, B=0.0, Z=1):
    """
    Compute fundamental plasma parameters.

    Parameters:
    -----------
    n_e : float
        Electron density [m^-3]
    T_e : float
        Electron temperature [eV]
    B : float
        Magnetic field strength [Tesla]
    Z : int
        Ion charge number

    Returns:
    --------
    dict : Dictionary containing all plasma parameters
    """
    # Convert temperature to Joules
    T_e_J = T_e * e

    # Debye length [m]
    lambda_D = np.sqrt(epsilon_0 * T_e_J / (n_e * e**2))

    # Plasma frequency [rad/s]
    omega_pe = np.sqrt(n_e * e**2 / (epsilon_0 * m_e))
    omega_pi = np.sqrt(n_e * Z**2 * e**2 / (epsilon_0 * m_i))

    # Thermal velocity [m/s]
    v_th_e = np.sqrt(2 * T_e_J / m_e)
    v_th_i = np.sqrt(2 * T_e_J / m_i)

    # Magnetic field parameters (if B > 0)
    if B > 0:
        omega_ce = e * B / m_e  # Electron gyrofrequency [rad/s]
        omega_ci = Z * e * B / m_i  # Ion gyrofrequency [rad/s]
        r_Le = v_th_e / omega_ce  # Electron Larmor radius [m]
        r_Li = v_th_i / omega_ci  # Ion Larmor radius [m]

        # Plasma beta
        p_thermal = n_e * k * T_e * e  # Thermal pressure [Pa]
        p_magnetic = B**2 / (2 * mu_0)  # Magnetic pressure [Pa]
        beta = p_thermal / p_magnetic if p_magnetic > 0 else np.inf
    else:
        omega_ce = omega_ci = 0.0
        r_Le = r_Li = np.inf
        beta = np.inf

    # Coulomb logarithm (using simplified formula)
    if T_e > 10:  # eV
        ln_Lambda = 24 - np.log(np.sqrt(n_e * 1e-6) / T_e)
    else:
        ln_Lambda = 23 - np.log(np.sqrt(n_e * 1e-6) * Z / T_e**1.5)
    ln_Lambda = max(ln_Lambda, 2.0)  # Lower bound

    # Collision frequency (Spitzer, electron-ion) [Hz]
    nu_ei = (n_e * e**4 * ln_Lambda) / (
        12 * np.pi**1.5 * epsilon_0**2 * m_e**0.5 * (T_e_J)**1.5
    ) if T_e > 0 else np.inf

    # Mean free path [m]
    mfp = v_th_e / nu_ei if nu_ei > 0 else np.inf

    return {
        'lambda_D': lambda_D,
        'omega_pe': omega_pe,
        'omega_pi': omega_pi,
        'omega_ce': omega_ce,
        'omega_ci': omega_ci,
        'v_th_e': v_th_e,
        'v_th_i': v_th_i,
        'r_Le': r_Le,
        'r_Li': r_Li,
        'beta': beta,
        'ln_Lambda': ln_Lambda,
        'nu_ei': nu_ei,
        'mfp': mfp
    }


def print_parameters_table():
    """Print a comparison table of parameters for different plasma regimes."""

    # Define plasma regimes
    plasmas = [
        ('Tokamak Core', 1e20, 10e3, 5.0),
        ('Solar Wind', 5e6, 10, 5e-9),
        ('Ionosphere', 1e11, 0.1, 50e-6),
        ('Lightning', 1e24, 3, 0.0),
        ('Neon Sign', 1e16, 2, 0.0)
    ]

    print("=" * 100)
    print(f"{'Plasma Type':<15} {'n_e[m^-3]':<12} {'T_e[eV]':<10} {'λ_D[m]':<12} "
          f"{'f_pe[Hz]':<12} {'r_L[m]':<12} {'β':<10}")
    print("=" * 100)

    for name, n_e, T_e, B in plasmas:
        params = compute_plasma_parameters(n_e, T_e, B)
        f_pe = params['omega_pe'] / (2 * np.pi)
        r_L = params['r_Le']

        print(f"{name:<15} {n_e:<12.2e} {T_e:<10.2e} {params['lambda_D']:<12.2e} "
              f"{f_pe:<12.2e} {r_L:<12.2e} {params['beta']:<10.2e}")

    print("=" * 100)


def plot_parameter_space():
    """Plot plasma parameter space diagram showing different regimes."""

    # Create density and temperature arrays
    n_e = np.logspace(4, 28, 100)  # m^-3
    T_e = np.logspace(-2, 5, 100)  # eV

    N, T = np.meshgrid(n_e, T_e)

    # Compute Debye length
    T_J = T * e
    lambda_D = np.sqrt(epsilon_0 * T_J / (N * e**2))

    # Compute plasma parameter
    n_D = N * (4/3) * np.pi * lambda_D**3

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Debye length contours
    cs1 = ax1.contourf(np.log10(N), np.log10(T), np.log10(lambda_D),
                       levels=20, cmap='viridis')
    ax1.contour(np.log10(N), np.log10(T), np.log10(lambda_D),
                levels=[-9, -6, -3, 0, 3], colors='white', linewidths=1.5)

    # Mark different plasma regimes
    regimes = [
        ('Tokamak\nCore', 20, 4, 'red'),
        ('Solar\nWind', 6.7, 1, 'yellow'),
        ('Ionosphere', 11, -1, 'cyan'),
        ('Lightning', 24, 0.5, 'orange'),
        ('Neon\nSign', 16, 0.3, 'magenta')
    ]

    for name, log_n, log_T, color in regimes:
        ax1.plot(log_n, log_T, 'o', color=color, markersize=10,
                markeredgecolor='white', markeredgewidth=2)
        ax1.text(log_n, log_T + 0.5, name, color=color, fontsize=9,
                ha='center', fontweight='bold')

    ax1.set_xlabel('log₁₀(n_e [m⁻³])', fontsize=12)
    ax1.set_ylabel('log₁₀(T_e [eV])', fontsize=12)
    ax1.set_title('Debye Length in Parameter Space', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    cbar1 = plt.colorbar(cs1, ax=ax1)
    cbar1.set_label('log₁₀(λ_D [m])', fontsize=11)

    # Plot 2: Plasma parameter contours
    cs2 = ax2.contourf(np.log10(N), np.log10(T), np.log10(n_D),
                       levels=20, cmap='plasma')
    ax2.contour(np.log10(N), np.log10(T), np.log10(n_D),
                levels=[0, 3, 6, 9], colors='white', linewidths=1.5)

    # Mark same regimes
    for name, log_n, log_T, color in regimes:
        ax2.plot(log_n, log_T, 'o', color=color, markersize=10,
                markeredgecolor='white', markeredgewidth=2)
        ax2.text(log_n, log_T + 0.5, name, color=color, fontsize=9,
                ha='center', fontweight='bold')

    # Add diagonal line for n_D = 1 (plasma validity boundary)
    n_line = np.logspace(4, 28, 50)
    T_line = (n_line * (4/3) * np.pi * e**6) / (epsilon_0**3 * k**3)
    valid_idx = (T_line > 1e-2) & (T_line < 1e5)
    ax2.plot(np.log10(n_line[valid_idx]), np.log10(T_line[valid_idx] / e),
             'r--', linewidth=2, label='n_D = 1 (boundary)')

    ax2.set_xlabel('log₁₀(n_e [m⁻³])', fontsize=12)
    ax2.set_ylabel('log₁₀(T_e [eV])', fontsize=12)
    ax2.set_title('Plasma Parameter (n_D) Space', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=10)

    cbar2 = plt.colorbar(cs2, ax=ax2)
    cbar2.set_label('log₁₀(n_D)', fontsize=11)

    plt.tight_layout()
    plt.savefig('plasma_parameter_space.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_frequency_comparison():
    """Compare characteristic frequencies for tokamak plasma."""

    n_e = 1e20  # m^-3
    T_range = np.logspace(0, 4, 100)  # 1 eV to 10 keV
    B = 5.0  # Tesla

    frequencies = {
        'ω_pe': [],
        'ω_ce': [],
        'ω_ci': [],
        'ν_ei': []
    }

    for T_e in T_range:
        params = compute_plasma_parameters(n_e, T_e, B)
        frequencies['ω_pe'].append(params['omega_pe'])
        frequencies['ω_ce'].append(params['omega_ce'])
        frequencies['ω_ci'].append(params['omega_ci'])
        frequencies['ν_ei'].append(params['nu_ei'])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.loglog(T_range, np.array(frequencies['ω_pe']) / (2*np.pi),
              'b-', linewidth=2, label='f_pe (plasma frequency)')
    ax.loglog(T_range, np.array(frequencies['ω_ce']) / (2*np.pi),
              'r-', linewidth=2, label='f_ce (electron gyro)')
    ax.loglog(T_range, np.array(frequencies['ω_ci']) / (2*np.pi),
              'g-', linewidth=2, label='f_ci (ion gyro)')
    ax.loglog(T_range, frequencies['ν_ei'],
              'm--', linewidth=2, label='ν_ei (collision freq)')

    ax.set_xlabel('Electron Temperature [eV]', fontsize=13)
    ax.set_ylabel('Frequency [Hz]', fontsize=13)
    ax.set_title(f'Characteristic Frequencies\n(n_e = {n_e:.1e} m⁻³, B = {B} T)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=11)

    # Add regions
    ax.axhspan(1e6, 1e9, alpha=0.1, color='yellow', label='Radio waves')
    ax.axhspan(1e9, 1e12, alpha=0.1, color='cyan', label='Microwaves')

    plt.tight_layout()
    plt.savefig('plasma_frequencies.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("\n" + "="*100)
    print("PLASMA PARAMETERS CALCULATOR")
    print("="*100 + "\n")

    # Print comparison table
    print_parameters_table()

    # Example calculation for tokamak
    print("\n\nDetailed calculation for Tokamak Core:")
    print("-" * 60)
    n_e = 1e20  # m^-3
    T_e = 10e3  # eV
    B = 5.0  # Tesla

    params = compute_plasma_parameters(n_e, T_e, B)

    print(f"Density: {n_e:.2e} m^-3")
    print(f"Temperature: {T_e:.2e} eV")
    print(f"Magnetic field: {B} T\n")

    print(f"Debye length: {params['lambda_D']:.4e} m")
    print(f"Plasma frequency (e): {params['omega_pe']/(2*np.pi):.4e} Hz")
    print(f"Plasma frequency (i): {params['omega_pi']/(2*np.pi):.4e} Hz")
    print(f"Gyrofrequency (e): {params['omega_ce']/(2*np.pi):.4e} Hz")
    print(f"Gyrofrequency (i): {params['omega_ci']/(2*np.pi):.4e} Hz")
    print(f"Thermal velocity (e): {params['v_th_e']:.4e} m/s")
    print(f"Thermal velocity (i): {params['v_th_i']:.4e} m/s")
    print(f"Larmor radius (e): {params['r_Le']:.4e} m")
    print(f"Larmor radius (i): {params['r_Li']:.4e} m")
    print(f"Plasma beta: {params['beta']:.4e}")
    print(f"Coulomb logarithm: {params['ln_Lambda']:.2f}")
    print(f"Collision frequency: {params['nu_ei']:.4e} Hz")
    print(f"Mean free path: {params['mfp']:.4e} m")

    # Generate plots
    print("\n\nGenerating parameter space diagrams...")
    plot_parameter_space()

    print("Generating frequency comparison plot...")
    plot_frequency_comparison()

    print("\nDone! Check plasma_parameter_space.png and plasma_frequencies.png")
