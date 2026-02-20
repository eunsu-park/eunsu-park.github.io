#!/usr/bin/env python3
"""
Diamagnetic Drift Visualization

This script visualizes the diamagnetic drift, which is a fluid effect
(not a real particle drift) arising from pressure gradients.

Key Physics:
- Diamagnetic current: J* = B × ∇p / B²
- Ion and electron drifts in opposite directions
- NOT a real particle drift - fluid velocity only
- Compare with E×B drift (same for both species)

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Physical constants
QE = 1.602176634e-19    # C
ME = 9.10938356e-31     # kg
MP = 1.672621898e-27    # kg

def density_profile(x, n0, Ln):
    """
    Exponential density profile.

    n(x) = n0 * exp(-x/Ln)

    Parameters:
    -----------
    x : array
        Position [m]
    n0 : float
        Central density [m^-3]
    Ln : float
        Density scale length [m]

    Returns:
    --------
    n : array
        Density [m^-3]
    """
    return n0 * np.exp(-x / Ln)

def pressure_gradient(x, n0, T, Ln):
    """
    Pressure gradient for exponential profile.

    p = n·kT
    ∇p = -kT·n0/Ln·exp(-x/Ln)

    Parameters:
    -----------
    x : array
        Position [m]
    n0 : float
        Central density [m^-3]
    T : float
        Temperature [eV]
    Ln : float
        Scale length [m]

    Returns:
    --------
    grad_p : array
        Pressure gradient [Pa/m]
    """
    T_joule = T * QE
    n = density_profile(x, n0, Ln)
    grad_p = -T_joule * n / Ln
    return grad_p

def diamagnetic_drift_velocity(grad_p, B0, charge_sign=1):
    """
    Compute diamagnetic drift velocity.

    v* = (B × ∇p) / (qnB²)

    For electrons: charge_sign = -1
    For ions: charge_sign = +1

    Parameters:
    -----------
    grad_p : array
        Pressure gradient [Pa/m]
    B0 : float
        Magnetic field [T]
    charge_sign : int
        +1 for ions, -1 for electrons

    Returns:
    --------
    v_star : array
        Diamagnetic drift velocity [m/s]
    """
    # For gradient in x direction, B in z direction
    # v* is in y direction
    v_star = -charge_sign * grad_p / (B0**2)

    return v_star

def visualize_gyro_orbits(x_center, n, B0, T, m, charge_sign, num_particles=5):
    """
    Visualize gyro orbits at different x positions to show mechanism.

    Parameters:
    -----------
    x_center : array
        Center positions
    n : array
        Density at each position
    B0 : float
        Magnetic field
    T : float
        Temperature [eV]
    m : float
        Particle mass
    charge_sign : int
        ±1
    num_particles : int
        Number of particles to show

    Returns:
    --------
    x_orbits, y_orbits : lists of orbit coordinates
    """
    T_joule = T * QE
    vth = np.sqrt(2 * T_joule / m)
    omega_c = abs(charge_sign * QE * B0 / m)
    rho = vth / omega_c

    x_orbits = []
    y_orbits = []

    # Sample particles at different x positions
    x_samples = np.linspace(x_center.min(), x_center.max(), num_particles)

    for x_c in x_samples:
        # Gyro orbit
        theta = np.linspace(0, 2 * np.pi, 100)

        if charge_sign > 0:
            # Ion gyrates clockwise (looking along B)
            x_orbit = x_c + rho * np.cos(theta)
            y_orbit = rho * np.sin(theta)
        else:
            # Electron gyrates counterclockwise
            x_orbit = x_c - rho * np.cos(theta)
            y_orbit = rho * np.sin(theta)

        x_orbits.append(x_orbit)
        y_orbits.append(y_orbit)

    return x_orbits, y_orbits, rho

def plot_diamagnetic_drift():
    """
    Create comprehensive visualization of diamagnetic drift.
    """
    # Plasma parameters
    n0 = 1e19  # m^-3
    Te = 10.0  # eV
    Ti = 10.0  # eV
    B0 = 1.0   # T
    Ln = 0.05  # 5 cm scale length
    mi = MP

    print("=" * 70)
    print("Diamagnetic Drift Parameters")
    print("=" * 70)
    print(f"Central density: {n0:.2e} m^-3")
    print(f"Electron temperature: {Te:.1f} eV")
    print(f"Ion temperature: {Ti:.1f} eV")
    print(f"Magnetic field: {B0:.2f} T")
    print(f"Density scale length: {Ln*100:.1f} cm")
    print("=" * 70)

    # Position array
    x = np.linspace(0, 0.2, 1000)  # 0 to 20 cm

    # Density and pressure profiles
    ne = density_profile(x, n0, Ln)
    ni = ne  # Quasineutrality

    grad_pe = pressure_gradient(x, n0, Te, Ln)
    grad_pi = pressure_gradient(x, n0, Ti, Ln)

    # Diamagnetic velocities (in y-direction)
    # For gradient in -x direction: ∇p points in -x
    # B in +z direction
    # B × ∇p points in ±y direction

    v_star_e = diamagnetic_drift_velocity(grad_pe, B0, charge_sign=-1)
    v_star_i = diamagnetic_drift_velocity(grad_pi, B0, charge_sign=+1)

    # Diamagnetic current
    J_star = QE * (ni * v_star_i - ne * v_star_e)

    # For comparison: E×B drift (same for both species)
    # Assume some electric field
    E_field = 1000  # V/m (arbitrary)
    v_ExB = E_field / B0

    # Create figure
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Plot 1: Density profile
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x * 100, ne / 1e19, 'b-', linewidth=2)
    ax1.set_xlabel('Position x (cm)', fontsize=11)
    ax1.set_ylabel(r'Density ($10^{19}$ m$^{-3}$)', fontsize=11)
    ax1.set_title('Density Profile (∇n points in -x)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=Ln * 100, color='r', linestyle='--', linewidth=1,
                label=f'Scale length = {Ln*100:.1f} cm')
    ax1.legend(fontsize=9)

    # Plot 2: Pressure gradient
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x * 100, grad_pe / 1e6, 'b-', linewidth=2, label='Electron')
    ax2.plot(x * 100, grad_pi / 1e6, 'r-', linewidth=2, label='Ion')
    ax2.set_xlabel('Position x (cm)', fontsize=11)
    ax2.set_ylabel(r'∇p (MPa/m)', fontsize=11)
    ax2.set_title('Pressure Gradient', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # Plot 3: Diamagnetic drift velocities
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(x * 100, v_star_e / 1e3, 'b-', linewidth=2.5, label='Electron v*e (fluid)')
    ax3.plot(x * 100, v_star_i / 1e3, 'r-', linewidth=2.5, label='Ion v*i (fluid)')
    ax3.axhline(y=v_ExB / 1e3, color='g', linestyle='--', linewidth=2,
                label=f'E×B drift = {v_ExB/1e3:.1f} km/s (both species)')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    ax3.set_xlabel('Position x (cm)', fontsize=11)
    ax3.set_ylabel('Drift Velocity (km/s)', fontsize=11)
    ax3.set_title('Diamagnetic Drift vs E×B Drift', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, loc='upper right')

    # Add text annotation
    ax3.text(0.02, 0.95, 'Diamagnetic drifts are OPPOSITE\nE×B drifts are SAME',
            transform=ax3.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Plot 4: Diamagnetic current
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(x * 100, J_star / 1e6, 'purple', linewidth=2.5)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Position x (cm)', fontsize=11)
    ax4.set_ylabel(r'Current Density (MA/m$^2$)', fontsize=11)
    ax4.set_title('Diamagnetic Current J* = (B × ∇p)/B²',
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Mechanism illustration (gyro orbits)
    ax5 = fig.add_subplot(gs[2, 1])

    # Show more particles on one side than the other
    x_left = 0.05  # High density
    x_right = 0.15  # Low density

    n_left = density_profile(x_left, n0, Ln)
    n_right = density_profile(x_right, n0, Ln)

    num_left = int(20 * n_left / n0)
    num_right = int(20 * n_right / n0)

    # Electron Larmor radius
    Te_joule = Te * QE
    vth_e = np.sqrt(2 * Te_joule / ME)
    omega_ce = QE * B0 / ME
    rho_e = vth_e / omega_ce

    # Plot electron gyro orbits
    y_positions_left = np.linspace(-0.01, 0.01, num_left)
    y_positions_right = np.linspace(-0.01, 0.01, num_right)

    for y_pos in y_positions_left:
        theta = np.linspace(0, 2 * np.pi, 50)
        x_orbit = x_left - rho_e * np.cos(theta)
        y_orbit = y_pos + rho_e * np.sin(theta)
        ax5.plot(x_orbit * 100, y_orbit * 100, 'b-', alpha=0.3, linewidth=0.5)

    for y_pos in y_positions_right:
        theta = np.linspace(0, 2 * np.pi, 50)
        x_orbit = x_right - rho_e * np.cos(theta)
        y_orbit = y_pos + rho_e * np.sin(theta)
        ax5.plot(x_orbit * 100, y_orbit * 100, 'b-', alpha=0.3, linewidth=0.5)

    # Mark regions
    ax5.axvline(x=x_left * 100, color='red', linestyle='--', linewidth=2,
                label='High density')
    ax5.axvline(x=x_right * 100, color='blue', linestyle='--', linewidth=2,
                label='Low density')

    # Arrow showing net drift
    ax5.annotate('Net drift →', xy=(0.5, 0.95), xytext=(0.2, 0.95),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=11, fontweight='bold')

    ax5.set_xlabel('x (cm)', fontsize=11)
    ax5.set_ylabel('y (cm)', fontsize=11)
    ax5.set_title('Mechanism: More Orbits on Left → Net Drift',
                  fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.set_aspect('equal')

    # Plot 6: Summary comparison table
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')

    # Create comparison table
    comparison_data = [
        ['Property', 'Diamagnetic Drift', 'E×B Drift'],
        ['Direction', 'Opposite for e⁻ and ions', 'Same for both'],
        ['Real drift?', 'NO (fluid only)', 'YES (particle drift)'],
        ['Formula', 'v* = (B × ∇p)/(qnB²)', 'v = E×B/B²'],
        ['Current', 'J* = (B × ∇p)/B²', 'J = 0 (both drift same)'],
        ['Origin', 'Pressure gradient', 'Electric field'],
    ]

    # Draw table
    table = ax6.table(cellText=comparison_data, cellLoc='left',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(comparison_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax6.set_title('Comparison: Diamagnetic vs E×B Drift',
                  fontsize=12, fontweight='bold', pad=20)

    plt.suptitle('Diamagnetic Drift: A Fluid Effect',
                 fontsize=16, fontweight='bold', y=0.998)

    plt.savefig('diamagnetic_drift.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'diamagnetic_drift.png'")
    print("\nKey insight:")
    print("  Diamagnetic drift is NOT a real particle drift!")
    print("  It's a fluid velocity arising from density gradients.")
    print("  Individual particles don't drift diamagnetically,")
    print("  but the fluid (average) velocity does.")

    plt.show()

if __name__ == "__main__":
    plot_diamagnetic_drift()
