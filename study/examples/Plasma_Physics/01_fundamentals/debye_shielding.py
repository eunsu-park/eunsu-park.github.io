#!/usr/bin/env python3
"""
Debye Shielding Visualization

This script demonstrates the shielding of a test charge in plasma by solving
the Poisson equation with Boltzmann electrons. Shows both linearized analytical
solution and numerical nonlinear solution.

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, epsilon_0, k
from scipy.integrate import odeint


def debye_length(n0, T_eV):
    """
    Calculate Debye length.

    Parameters:
    -----------
    n0 : float
        Background density [m^-3]
    T_eV : float
        Temperature [eV]

    Returns:
    --------
    float : Debye length [m]
    """
    T_J = T_eV * e
    return np.sqrt(epsilon_0 * T_J / (n0 * e**2))


def bare_coulomb_potential(r, q):
    """
    Bare Coulomb potential.

    Parameters:
    -----------
    r : array
        Radial distance [m]
    q : float
        Test charge [C]

    Returns:
    --------
    array : Potential [V]
    """
    # Avoid singularity at r=0
    r_safe = np.where(r > 0, r, 1e-10)
    return q / (4 * np.pi * epsilon_0 * r_safe)


def debye_shielded_potential(r, q, lambda_D):
    """
    Debye-shielded potential (analytical, linearized).

    Parameters:
    -----------
    r : array
        Radial distance [m]
    q : float
        Test charge [C]
    lambda_D : float
        Debye length [m]

    Returns:
    --------
    array : Potential [V]
    """
    r_safe = np.where(r > 0, r, 1e-10)
    return (q / (4 * np.pi * epsilon_0 * r_safe)) * np.exp(-r_safe / lambda_D)


def poisson_1d_spherical(y, r, n0, T_eV, q):
    """
    1D spherical Poisson equation with Boltzmann electrons.

    d²φ/dr² + (2/r)dφ/dr = (e*n0/ε0)(exp(eφ/kT) - 1)

    Convert to system of ODEs:
    y[0] = φ
    y[1] = dφ/dr

    Parameters:
    -----------
    y : array
        [φ, dφ/dr]
    r : float
        Radial coordinate [m]
    n0 : float
        Background density [m^-3]
    T_eV : float
        Temperature [eV]
    q : float
        Test charge [C]

    Returns:
    --------
    array : [dφ/dr, d²φ/dr²]
    """
    phi, dphi_dr = y
    T_J = T_eV * e

    # Avoid singularity at r=0
    if r < 1e-12:
        r = 1e-12

    # Poisson equation
    d2phi_dr2 = (e * n0 / epsilon_0) * (np.exp(e * phi / T_J) - 1) - (2 / r) * dphi_dr

    return [dphi_dr, d2phi_dr2]


def solve_poisson_numerical(r_array, n0, T_eV, q):
    """
    Solve Poisson equation numerically with boundary conditions.

    Parameters:
    -----------
    r_array : array
        Radial grid [m]
    n0 : float
        Background density [m^-3]
    T_eV : float
        Temperature [eV]
    q : float
        Test charge [C]

    Returns:
    --------
    array : Potential φ(r) [V]
    """
    # Boundary condition at small r (approximately bare Coulomb)
    r0 = r_array[0]
    phi0 = q / (4 * np.pi * epsilon_0 * r0)
    dphi0 = -q / (4 * np.pi * epsilon_0 * r0**2)

    # Initial condition
    y0 = [phi0, dphi0]

    # Solve ODE
    solution = odeint(poisson_1d_spherical, y0, r_array, args=(n0, T_eV, q))

    return solution[:, 0]


def plot_shielding_comparison():
    """Plot comparison of bare Coulomb, linearized Debye, and numerical solution."""

    # Parameters
    n0 = 1e16  # m^-3
    T_eV = 2.0  # eV
    q = e  # Test charge (one elementary charge)

    lambda_D = debye_length(n0, T_eV)

    # Radial grid
    r = np.logspace(-6, -2, 500)  # 1 μm to 1 cm

    # Compute potentials
    phi_bare = bare_coulomb_potential(r, q)
    phi_debye = debye_shielded_potential(r, q, lambda_D)
    phi_numerical = solve_poisson_numerical(r, n0, T_eV, q)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Linear scale
    ax1.plot(r * 1e3, phi_bare, 'k--', linewidth=2, label='Bare Coulomb', alpha=0.7)
    ax1.plot(r * 1e3, phi_debye, 'b-', linewidth=2, label='Debye (linearized)')
    ax1.plot(r * 1e3, phi_numerical, 'r:', linewidth=2.5, label='Numerical (nonlinear)')
    ax1.axvline(lambda_D * 1e3, color='gray', linestyle='--', alpha=0.5,
                label=f'λ_D = {lambda_D*1e3:.3f} mm')

    ax1.set_xlabel('Distance r [mm]', fontsize=12)
    ax1.set_ylabel('Potential φ(r) [V]', fontsize=12)
    ax1.set_title('Debye Shielding of Test Charge', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 5])
    ax1.set_ylim([0, 20])

    # Plot 2: Log-log scale
    ax2.loglog(r * 1e3, phi_bare, 'k--', linewidth=2, label='Bare Coulomb', alpha=0.7)
    ax2.loglog(r * 1e3, phi_debye, 'b-', linewidth=2, label='Debye (linearized)')
    ax2.loglog(r * 1e3, phi_numerical, 'r:', linewidth=2.5, label='Numerical (nonlinear)')
    ax2.axvline(lambda_D * 1e3, color='gray', linestyle='--', alpha=0.5,
                label=f'λ_D = {lambda_D*1e3:.3f} mm')

    ax2.set_xlabel('Distance r [mm]', fontsize=12)
    ax2.set_ylabel('Potential φ(r) [V]', fontsize=12)
    ax2.set_title('Debye Shielding (Log Scale)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('debye_shielding_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_temperature_dependence():
    """Show how shielding changes with temperature."""

    # Parameters
    n0 = 1e16  # m^-3
    q = e
    temperatures = [0.5, 1.0, 2.0, 5.0]  # eV

    # Radial grid
    r = np.linspace(0.001e-3, 5e-3, 200)  # mm scale

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(temperatures)))

    for T_eV, color in zip(temperatures, colors):
        lambda_D = debye_length(n0, T_eV)
        phi_debye = debye_shielded_potential(r, q, lambda_D)

        ax.plot(r * 1e3, phi_debye, linewidth=2.5, color=color,
                label=f'T = {T_eV} eV, λ_D = {lambda_D*1e3:.3f} mm')
        ax.axvline(lambda_D * 1e3, color=color, linestyle=':', alpha=0.5)

    # Bare Coulomb for reference
    phi_bare = bare_coulomb_potential(r, q)
    ax.plot(r * 1e3, phi_bare, 'k--', linewidth=2, label='Bare Coulomb', alpha=0.5)

    ax.set_xlabel('Distance r [mm]', fontsize=12)
    ax.set_ylabel('Potential φ(r) [V]', fontsize=12)
    ax.set_title(f'Temperature Dependence of Debye Shielding\n(n₀ = {n0:.1e} m⁻³)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 30])

    plt.tight_layout()
    plt.savefig('debye_temperature_dependence.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_density_response():
    """Visualize electron density response around test charge."""

    # Parameters
    n0 = 1e16  # m^-3
    T_eV = 2.0  # eV
    q = e

    lambda_D = debye_length(n0, T_eV)
    T_J = T_eV * e

    # Radial grid
    r = np.linspace(0.001e-3, 10e-3, 500)  # mm scale

    # Compute potential
    phi_debye = debye_shielded_potential(r, q, lambda_D)

    # Electron density from Boltzmann relation
    n_e = n0 * np.exp(e * phi_debye / T_J)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Potential
    ax1.plot(r * 1e3, phi_debye, 'b-', linewidth=2.5)
    ax1.axvline(lambda_D * 1e3, color='red', linestyle='--', linewidth=2,
                label=f'λ_D = {lambda_D*1e3:.3f} mm')
    ax1.axhline(0, color='k', linestyle='-', linewidth=0.8)

    ax1.set_xlabel('Distance r [mm]', fontsize=12)
    ax1.set_ylabel('Potential φ(r) [V]', fontsize=12)
    ax1.set_title('Electrostatic Potential', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Density perturbation
    delta_n = n_e - n0
    ax2.plot(r * 1e3, delta_n / n0 * 100, 'r-', linewidth=2.5)
    ax2.axvline(lambda_D * 1e3, color='red', linestyle='--', linewidth=2,
                label=f'λ_D = {lambda_D*1e3:.3f} mm')
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.8)

    ax2.set_xlabel('Distance r [mm]', fontsize=12)
    ax2.set_ylabel('Density Perturbation Δn/n₀ [%]', fontsize=12)
    ax2.set_title('Electron Density Response', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('debye_density_response.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_2d_potential():
    """2D visualization of shielded potential."""

    # Parameters
    n0 = 1e16  # m^-3
    T_eV = 2.0  # eV
    q = e

    lambda_D = debye_length(n0, T_eV)

    # Create 2D grid
    x = np.linspace(-5e-3, 5e-3, 200)
    y = np.linspace(-5e-3, 5e-3, 200)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Compute potentials
    phi_bare = bare_coulomb_potential(R, q)
    phi_debye = debye_shielded_potential(R, q, lambda_D)

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Bare Coulomb
    levels = np.linspace(0, 20, 30)
    cs1 = ax1.contourf(X * 1e3, Y * 1e3, phi_bare, levels=levels, cmap='hot')
    ax1.contour(X * 1e3, Y * 1e3, phi_bare, levels=[1, 5, 10, 15],
                colors='white', linewidths=1, alpha=0.5)
    ax1.set_xlabel('x [mm]', fontsize=12)
    ax1.set_ylabel('y [mm]', fontsize=12)
    ax1.set_title('Bare Coulomb Potential', fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    plt.colorbar(cs1, ax=ax1, label='φ [V]')

    # Plot 2: Debye shielded
    cs2 = ax2.contourf(X * 1e3, Y * 1e3, phi_debye, levels=levels, cmap='hot')
    ax2.contour(X * 1e3, Y * 1e3, phi_debye, levels=[1, 5, 10, 15],
                colors='white', linewidths=1, alpha=0.5)
    circle = plt.Circle((0, 0), lambda_D * 1e3, color='cyan', fill=False,
                        linewidth=2, linestyle='--', label='λ_D')
    ax2.add_patch(circle)
    ax2.set_xlabel('x [mm]', fontsize=12)
    ax2.set_ylabel('y [mm]', fontsize=12)
    ax2.set_title('Debye-Shielded Potential', fontsize=13, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.legend(loc='upper right', fontsize=10)
    plt.colorbar(cs2, ax=ax2, label='φ [V]')

    # Plot 3: Difference
    diff = phi_bare - phi_debye
    cs3 = ax3.contourf(X * 1e3, Y * 1e3, diff, levels=20, cmap='coolwarm')
    ax3.contour(X * 1e3, Y * 1e3, diff, levels=10, colors='black',
                linewidths=0.5, alpha=0.3)
    circle = plt.Circle((0, 0), lambda_D * 1e3, color='green', fill=False,
                        linewidth=2, linestyle='--', label='λ_D')
    ax3.add_patch(circle)
    ax3.set_xlabel('x [mm]', fontsize=12)
    ax3.set_ylabel('y [mm]', fontsize=12)
    ax3.set_title('Shielding Effect (Difference)', fontsize=13, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.legend(loc='upper right', fontsize=10)
    plt.colorbar(cs3, ax=ax3, label='Δφ [V]')

    plt.tight_layout()
    plt.savefig('debye_2d_potential.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("DEBYE SHIELDING VISUALIZATION")
    print("="*80 + "\n")

    # Example parameters
    n0 = 1e16  # m^-3
    T_eV = 2.0  # eV
    q = e

    lambda_D = debye_length(n0, T_eV)

    print(f"Plasma parameters:")
    print(f"  Density: {n0:.2e} m^-3")
    print(f"  Temperature: {T_eV} eV")
    print(f"  Test charge: {q/e:.1f} e")
    print(f"\nDebye length: {lambda_D:.4e} m = {lambda_D*1e3:.4f} mm\n")

    print("Generating plots...")
    print("  1. Shielding comparison (linear vs numerical)...")
    plot_shielding_comparison()

    print("  2. Temperature dependence...")
    plot_temperature_dependence()

    print("  3. Density response...")
    plot_density_response()

    print("  4. 2D potential visualization...")
    plot_2d_potential()

    print("\nDone! Generated 4 figures:")
    print("  - debye_shielding_comparison.png")
    print("  - debye_temperature_dependence.png")
    print("  - debye_density_response.png")
    print("  - debye_2d_potential.png")
