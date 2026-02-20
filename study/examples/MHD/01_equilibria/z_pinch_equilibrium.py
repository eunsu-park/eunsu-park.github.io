#!/usr/bin/env python3
"""
Z-Pinch Equilibrium Analysis

This script solves the Z-pinch equilibrium equation:
    p'(r) = -J_z(r) * B_θ(r)

for a given current profile and verifies the Bennett relation:
    I² = 8π N k T / μ₀

The Z-pinch is a cylindrical plasma configuration where an axial current
creates an azimuthal magnetic field that confines the plasma.

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, simpson
from scipy.constants import mu_0, k as k_B, e

# Physical constants
MU_0 = mu_0  # Permeability of free space


def current_density_profile(r, r_max, I_total, profile_type='uniform'):
    """
    Define current density profile J_z(r).

    Parameters
    ----------
    r : array_like
        Radial positions
    r_max : float
        Maximum radius of the pinch
    I_total : float
        Total current (A)
    profile_type : str
        Type of profile: 'uniform', 'parabolic', or 'skin'

    Returns
    -------
    J_z : ndarray
        Current density at each radius
    """
    if profile_type == 'uniform':
        # Uniform current density
        J_0 = I_total / (np.pi * r_max**2)
        J_z = np.where(r <= r_max, J_0, 0.0)

    elif profile_type == 'parabolic':
        # Parabolic profile: J ~ (1 - (r/r_max)²)
        J_0 = 2 * I_total / (np.pi * r_max**2)
        J_z = np.where(r <= r_max, J_0 * (1 - (r/r_max)**2), 0.0)

    elif profile_type == 'skin':
        # Skin current: J ~ exp(-(r/δ)²)
        delta = r_max / 3  # Skin depth
        J_0 = I_total / (np.pi * delta**2 * (1 - np.exp(-r_max**2/delta**2)))
        J_z = J_0 * np.exp(-(r/delta)**2)

    else:
        raise ValueError(f"Unknown profile type: {profile_type}")

    return J_z


def azimuthal_field(r, r_max, I_total, profile_type='uniform'):
    """
    Compute azimuthal magnetic field B_θ(r) from Ampere's law.

    B_θ(r) = μ₀ I(r) / (2π r)
    where I(r) is the current enclosed within radius r.

    Parameters
    ----------
    r : array_like
        Radial positions
    r_max : float
        Maximum radius
    I_total : float
        Total current
    profile_type : str
        Current profile type

    Returns
    -------
    B_theta : ndarray
        Azimuthal magnetic field
    """
    r = np.asarray(r)
    B_theta = np.zeros_like(r)

    for i, ri in enumerate(r):
        if ri < 1e-10:
            B_theta[i] = 0.0
        else:
            # Integrate current density to get enclosed current
            r_int = np.linspace(0, ri, 100)
            J_int = current_density_profile(r_int, r_max, I_total, profile_type)
            I_enclosed = 2 * np.pi * simpson(r_int * J_int, x=r_int)
            B_theta[i] = MU_0 * I_enclosed / (2 * np.pi * ri)

    return B_theta


def solve_pressure_balance(r, r_max, I_total, p_edge, profile_type='uniform'):
    """
    Solve pressure balance equation: dp/dr = -J_z * B_θ

    Parameters
    ----------
    r : array_like
        Radial grid
    r_max : float
        Pinch radius
    I_total : float
        Total current
    p_edge : float
        Pressure at edge
    profile_type : str
        Current profile type

    Returns
    -------
    pressure : ndarray
        Pressure profile
    """
    J_z = current_density_profile(r, r_max, I_total, profile_type)
    B_theta = azimuthal_field(r, r_max, I_total, profile_type)

    # Integrate from edge inward
    pressure = np.zeros_like(r)
    pressure[-1] = p_edge

    for i in range(len(r)-2, -1, -1):
        dr = r[i+1] - r[i]
        # dp/dr = -J_z * B_θ, integrate backward
        pressure[i] = pressure[i+1] - J_z[i] * B_theta[i] * dr

    return pressure


def safety_factor(r, r_max, I_total, profile_type='uniform'):
    """
    Compute safety factor q(r) = r * B_z / (R * B_θ)

    For a Z-pinch with small B_z (from external field),
    we assume B_z = constant for this calculation.

    Parameters
    ----------
    r : array_like
        Radial positions
    r_max : float
        Pinch radius
    I_total : float
        Total current
    profile_type : str
        Current profile type

    Returns
    -------
    q : ndarray
        Safety factor
    """
    B_z = 0.1  # Assume small axial field (Tesla)
    B_theta = azimuthal_field(r, r_max, I_total, profile_type)
    R_major = 1.0  # Assume major radius = 1 m for calculation

    # Avoid division by zero
    q = np.where(np.abs(B_theta) > 1e-10,
                 r * B_z / (R_major * B_theta),
                 np.inf)
    return q


def bennett_relation(I_total, n_avg, T_avg):
    """
    Verify Bennett relation: I² = 8π N k T / μ₀

    where N is the line density (particles per unit length).

    Parameters
    ----------
    I_total : float
        Total current (A)
    n_avg : float
        Average particle density (m⁻³)
    T_avg : float
        Average temperature (eV)

    Returns
    -------
    dict
        Bennett relation verification results
    """
    # Convert temperature to Joules
    T_J = T_avg * e

    # For a pinch of radius a, N ~ n * π * a²
    # We'll compute the expected current from Bennett relation
    a = 0.01  # Assume 1 cm radius
    N_line = n_avg * np.pi * a**2

    I_bennett = np.sqrt(8 * np.pi * N_line * k_B * T_J / MU_0)

    return {
        'I_actual': I_total,
        'I_bennett': I_bennett,
        'ratio': I_total / I_bennett,
        'N_line': N_line
    }


def plot_equilibrium(r, pressure, B_theta, J_z, q, profile_type):
    """
    Plot Z-pinch equilibrium profiles.

    Parameters
    ----------
    r : ndarray
        Radial grid
    pressure : ndarray
        Pressure profile
    B_theta : ndarray
        Azimuthal field
    J_z : ndarray
        Current density
    q : ndarray
        Safety factor
    profile_type : str
        Current profile type
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Pressure
    axes[0, 0].plot(r * 100, pressure / 1e3, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Radius (cm)', fontsize=12)
    axes[0, 0].set_ylabel('Pressure (kPa)', fontsize=12)
    axes[0, 0].set_title('Pressure Profile', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Azimuthal field
    axes[0, 1].plot(r * 100, B_theta * 1e3, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Radius (cm)', fontsize=12)
    axes[0, 1].set_ylabel('B_θ (mT)', fontsize=12)
    axes[0, 1].set_title('Azimuthal Magnetic Field', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Current density
    axes[1, 0].plot(r * 100, J_z / 1e6, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Radius (cm)', fontsize=12)
    axes[1, 0].set_ylabel('J_z (MA/m²)', fontsize=12)
    axes[1, 0].set_title('Current Density', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Safety factor (plot only finite values)
    q_plot = np.where(np.isfinite(q) & (q < 100), q, np.nan)
    axes[1, 1].plot(r * 100, q_plot, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Radius (cm)', fontsize=12)
    axes[1, 1].set_ylabel('Safety factor q', fontsize=12)
    axes[1, 1].set_title('Safety Factor', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Z-Pinch Equilibrium ({profile_type.capitalize()} Current Profile)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'z_pinch_{profile_type}.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    # Parameters
    r_max = 0.01  # Pinch radius: 1 cm
    I_total = 100e3  # Total current: 100 kA
    p_edge = 100.0  # Edge pressure: 100 Pa
    profile_type = 'parabolic'  # 'uniform', 'parabolic', or 'skin'

    # Plasma parameters for Bennett relation
    n_avg = 1e20  # Average density: 10²⁰ m⁻³
    T_avg = 100.0  # Average temperature: 100 eV

    # Create radial grid
    r = np.linspace(0, r_max, 200)

    # Compute profiles
    print(f"Computing Z-pinch equilibrium with {profile_type} current profile...")
    J_z = current_density_profile(r, r_max, I_total, profile_type)
    B_theta = azimuthal_field(r, r_max, I_total, profile_type)
    pressure = solve_pressure_balance(r, r_max, I_total, p_edge, profile_type)
    q = safety_factor(r, r_max, I_total, profile_type)

    # Print key results
    print(f"\nKey Parameters:")
    print(f"  Pinch radius: {r_max*100:.2f} cm")
    print(f"  Total current: {I_total/1e3:.1f} kA")
    print(f"  Peak current density: {np.max(J_z)/1e6:.2f} MA/m²")
    print(f"  Peak B_θ: {np.max(B_theta)*1e3:.2f} mT")
    print(f"  Peak pressure: {np.max(pressure)/1e3:.2f} kPa")
    print(f"  Central pressure: {pressure[0]/1e3:.2f} kPa")

    # Bennett relation
    bennett = bennett_relation(I_total, n_avg, T_avg)
    print(f"\nBennett Relation Verification:")
    print(f"  Actual current: {bennett['I_actual']/1e3:.1f} kA")
    print(f"  Bennett current: {bennett['I_bennett']/1e3:.1f} kA")
    print(f"  Ratio I_actual/I_bennett: {bennett['ratio']:.3f}")
    print(f"  Line density N: {bennett['N_line']:.2e} m⁻¹")

    if np.abs(bennett['ratio'] - 1.0) < 0.2:
        print("  ✓ Bennett relation approximately satisfied")
    else:
        print("  ✗ Bennett relation not satisfied (adjust n or T)")

    # Plot results
    plot_equilibrium(r, pressure, B_theta, J_z, q, profile_type)
    print(f"\nPlot saved as 'z_pinch_{profile_type}.png'")


if __name__ == '__main__':
    main()
