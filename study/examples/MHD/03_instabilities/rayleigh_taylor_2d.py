#!/usr'bin/env python3
"""
Rayleigh-Taylor Instability in MHD

Analyzes the 2D MHD Rayleigh-Taylor instability with magnetic field.

Setup: Heavy fluid (ρ₂) over light fluid (ρ₁) with transverse B field.
Growth rate: γ² = gk - (k·B)²/(μ₀ρ)

Shows magnetic stabilization of short wavelengths.

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt


def dispersion_relation(k, g, rho_1, rho_2, B, theta):
    """
    Compute growth rate for RT instability with magnetic field.

    γ² = A * g * k - (k·B)²/(μ₀ρ_avg)

    where A = (ρ₂ - ρ₁)/(ρ₂ + ρ₁) is the Atwood number.

    Parameters
    ----------
    k : float or array_like
        Wave number (m⁻¹)
    g : float
        Gravitational acceleration (m/s²)
    rho_1, rho_2 : float
        Fluid densities (kg/m³), rho_2 > rho_1
    B : float
        Magnetic field strength (T)
    theta : float
        Angle between k and B (radians)

    Returns
    -------
    gamma : float or ndarray
        Growth rate (s⁻¹)
    """
    mu_0 = 4 * np.pi * 1e-7

    # Atwood number
    A = (rho_2 - rho_1) / (rho_2 + rho_1)

    # Average density
    rho_avg = (rho_1 + rho_2) / 2

    # Gravitational term (destabilizing)
    term_grav = A * g * np.abs(k)

    # Magnetic stabilization
    k_parallel = np.abs(k) * np.cos(theta)  # Component parallel to B
    term_mag = (B * k_parallel)**2 / (mu_0 * rho_avg)

    # Growth rate squared
    gamma_sq = term_grav - term_mag

    # Return growth rate (zero if stable)
    gamma = np.where(gamma_sq > 0, np.sqrt(gamma_sq), 0.0)

    return gamma


def critical_wavelength(g, rho_1, rho_2, B, theta):
    """
    Compute critical wavelength for stabilization.

    At λ_c, growth rate goes to zero: γ(k_c) = 0

    Parameters
    ----------
    g : float
        Gravity
    rho_1, rho_2 : float
        Densities
    B : float
        Magnetic field
    theta : float
        Angle

    Returns
    -------
    lambda_c : float
        Critical wavelength (m)
    """
    mu_0 = 4 * np.pi * 1e-7
    A = (rho_2 - rho_1) / (rho_2 + rho_1)
    rho_avg = (rho_1 + rho_2) / 2

    if np.abs(np.cos(theta)) < 1e-10:
        # Perpendicular: no stabilization
        return np.inf

    # k_c from γ²(k_c) = 0
    k_c = A * g * mu_0 * rho_avg / (B * np.cos(theta))**2

    lambda_c = 2 * np.pi / k_c if k_c > 0 else np.inf

    return lambda_c


def plot_growth_vs_wavenumber(g, rho_1, rho_2, B_values, theta=0):
    """
    Plot growth rate vs wavenumber for different B fields.

    Parameters
    ----------
    g : float
        Gravity
    rho_1, rho_2 : float
        Densities
    B_values : array_like
        Magnetic field strengths
    theta : float
        Angle (default 0 = parallel)
    """
    k = np.logspace(-2, 3, 200)  # Wave numbers from 0.01 to 1000 m⁻¹

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(B_values)))

    for i, B in enumerate(B_values):
        gamma = dispersion_relation(k, g, rho_1, rho_2, B, theta)

        label = f'B = {B:.2f} T'
        if B > 0:
            lambda_c = critical_wavelength(g, rho_1, rho_2, B, theta)
            if np.isfinite(lambda_c):
                label += f' (λ_c={lambda_c*100:.1f} cm)'

        ax.loglog(k, gamma, color=colors[i], linewidth=2.5, label=label)

    # No field case
    gamma_0 = dispersion_relation(k, g, rho_1, rho_2, 0, theta)
    ax.loglog(k, gamma_0, 'k--', linewidth=2, label='B = 0 (no field)')

    ax.set_xlabel('Wave number k (m⁻¹)', fontsize=13)
    ax.set_ylabel('Growth rate γ (s⁻¹)', fontsize=13)
    ax.set_title('Rayleigh-Taylor Growth Rate: Magnetic Stabilization',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Add annotations
    ax.text(0.05, 0.95,
           f'ρ₁ = {rho_1:.1e} kg/m³\nρ₂ = {rho_2:.1e} kg/m³\ng = {g:.1f} m/s²\nθ = {np.degrees(theta):.0f}°',
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig('rt_growth_vs_k.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_growth_vs_angle(g, rho_1, rho_2, B, k_values):
    """
    Plot growth rate vs angle between k and B.

    Parameters
    ----------
    g : float
        Gravity
    rho_1, rho_2 : float
        Densities
    B : float
        Magnetic field
    k_values : array_like
        Wave numbers to plot
    """
    theta = np.linspace(0, np.pi, 200)  # 0 to 180 degrees

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.plasma(np.linspace(0, 1, len(k_values)))

    for i, k in enumerate(k_values):
        gamma = np.array([dispersion_relation(k, g, rho_1, rho_2, B, t)
                         for t in theta])

        lambda_val = 2 * np.pi / k
        ax.plot(np.degrees(theta), gamma, color=colors[i], linewidth=2.5,
               label=f'λ = {lambda_val*100:.1f} cm')

    ax.set_xlabel('Angle θ between k and B (degrees)', fontsize=13)
    ax.set_ylabel('Growth rate γ (s⁻¹)', fontsize=13)
    ax.set_title('RT Growth Rate vs Field Orientation',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 180])

    # Mark key angles
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='θ=0° (parallel)')
    ax.axvline(x=90, color='b', linestyle='--', alpha=0.5, label='θ=90° (perpendicular)')

    plt.tight_layout()
    plt.savefig('rt_growth_vs_angle.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_2d_mode_structure(k_x, k_z, g, rho_1, rho_2, B, t_max=0.1):
    """
    Visualize 2D mode structure evolution.

    Parameters
    ----------
    k_x, k_z : float
        Wave vector components
    g : float
        Gravity
    rho_1, rho_2 : float
        Densities
    B : float
        Magnetic field (in x-direction)
    t_max : float
        Maximum time
    """
    # Grid
    x = np.linspace(0, 2*np.pi/k_x, 100)
    z = np.linspace(-0.05, 0.05, 100)
    X, Z = np.meshgrid(x, z)

    # Wave vector and angle
    k = np.sqrt(k_x**2 + k_z**2)
    theta = np.arctan2(k_z, k_x)

    # Growth rate
    gamma = dispersion_relation(k, g, rho_1, rho_2, B, theta)

    # Time snapshots
    times = np.linspace(0, t_max, 4)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, t in enumerate(times):
        ax = axes[i]

        # Displacement: ξ ~ exp(γt) * sin(k·x)
        amplitude = np.exp(gamma * t)
        xi = amplitude * np.sin(k_x * X + k_z * Z)

        # Plot
        im = ax.contourf(X*100, Z*100, xi, levels=20, cmap='RdBu_r')
        ax.contour(X*100, Z*100, xi, levels=10, colors='black',
                  linewidths=0.5, alpha=0.3)

        ax.set_xlabel('x (cm)', fontsize=11)
        ax.set_ylabel('z (cm)', fontsize=11)
        ax.set_title(f't = {t*1e3:.2f} ms, amplitude = {amplitude:.2f}',
                    fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Displacement ξ')

    plt.suptitle(f'RT Instability Evolution (γ = {gamma:.2f} s⁻¹, B = {B:.2f} T)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('rt_mode_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_stability_diagram(g, rho_1, rho_2):
    """
    Create stability diagram in (B, λ) space.

    Parameters
    ----------
    g : float
        Gravity
    rho_1, rho_2 : float
        Densities
    """
    # Grid
    B_arr = np.linspace(0, 0.5, 100)
    lambda_arr = np.logspace(-3, 0, 100)  # 1 mm to 1 m

    B_grid, lambda_grid = np.meshgrid(B_arr, lambda_arr)
    k_grid = 2 * np.pi / lambda_grid

    # Growth rate (parallel field)
    gamma_grid = np.zeros_like(B_grid)

    for i in range(len(B_arr)):
        for j in range(len(lambda_arr)):
            gamma_grid[j, i] = dispersion_relation(k_grid[j, i], g, rho_1, rho_2,
                                                   B_grid[j, i], theta=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filled contours
    im = ax.contourf(B_grid, lambda_grid*100, gamma_grid,
                     levels=20, cmap='hot_r')

    # Stability boundary (γ = 0)
    cs = ax.contour(B_grid, lambda_grid*100, gamma_grid,
                   levels=[0], colors='blue', linewidths=3)
    ax.clabel(cs, inline=True, fontsize=11, fmt='γ=0 (stable)')

    ax.set_xlabel('Magnetic field B (T)', fontsize=13)
    ax.set_ylabel('Wavelength λ (cm)', fontsize=13)
    ax.set_yscale('log')
    ax.set_title('RT Stability Diagram',
                 fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Growth rate γ (s⁻¹)', fontsize=12)

    # Add text regions
    ax.text(0.4, 50, 'UNSTABLE', fontsize=16, fontweight='bold',
           color='white', ha='center',
           bbox=dict(boxstyle='round', facecolor='red', alpha=0.6))
    ax.text(0.4, 0.5, 'STABLE', fontsize=16, fontweight='bold',
           color='black', ha='center',
           bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.6))

    plt.tight_layout()
    plt.savefig('rt_stability_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    # Parameters
    g = 10.0  # m/s² (gravity or equivalent acceleration)
    rho_1 = 0.1  # kg/m³ (light fluid, top initially)
    rho_2 = 1.0  # kg/m³ (heavy fluid, bottom)

    print("Rayleigh-Taylor Instability with Magnetic Field")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Light fluid density ρ₁: {rho_1:.2f} kg/m³")
    print(f"  Heavy fluid density ρ₂: {rho_2:.2f} kg/m³")
    print(f"  Atwood number A: {(rho_2-rho_1)/(rho_2+rho_1):.3f}")
    print(f"  Acceleration g: {g:.1f} m/s²")
    print()

    # Test different field strengths
    B_values = [0, 0.05, 0.1, 0.2, 0.3]

    print("Growth rates vs wavenumber:")
    plot_growth_vs_wavenumber(g, rho_1, rho_2, B_values, theta=0)
    print("  Saved as 'rt_growth_vs_k.png'")

    # Growth vs angle
    B_test = 0.2  # T
    k_test = [10, 50, 100, 200]  # m⁻¹

    print("\nGrowth rate vs field orientation:")
    plot_growth_vs_angle(g, rho_1, rho_2, B_test, k_test)
    print("  Saved as 'rt_growth_vs_angle.png'")

    # 2D mode evolution
    print("\n2D mode structure evolution:")
    k_x, k_z = 100, 0  # Parallel to field
    plot_2d_mode_structure(k_x, k_z, g, rho_1, rho_2, B=0.1, t_max=0.05)
    print("  Saved as 'rt_mode_evolution.png'")

    # Stability diagram
    print("\nStability diagram in (B, λ) space:")
    plot_stability_diagram(g, rho_1, rho_2)
    print("  Saved as 'rt_stability_diagram.png'")

    # Critical wavelength examples
    print("\nCritical wavelengths (parallel field):")
    for B in [0.05, 0.1, 0.2, 0.3]:
        lambda_c = critical_wavelength(g, rho_1, rho_2, B, theta=0)
        if np.isfinite(lambda_c):
            print(f"  B = {B:.2f} T: λ_c = {lambda_c*100:.2f} cm")


if __name__ == '__main__':
    main()
