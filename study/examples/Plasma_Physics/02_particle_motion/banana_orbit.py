#!/usr/bin/env python3
"""
Banana Orbit Simulation in Tokamak

This script simulates banana orbits in a simplified tokamak geometry.
Demonstrates the difference between passing and trapped particles in
toroidal magnetic field configuration.

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e


def boris_push(x, v, q, m, B, dt):
    """Boris algorithm for particle pushing (no E field)."""
    t = (q * dt / (2 * m)) * B
    t_mag2 = np.dot(t, t)
    s = 2 * t / (1 + t_mag2)

    v_prime = v + np.cross(v, t)
    v_new = v + np.cross(v_prime, s)
    x_new = x + v_new * dt

    return x_new, v_new


def tokamak_field(R, Z, phi, R0, B0, epsilon):
    """
    Simplified tokamak magnetic field using large aspect ratio approximation.

    B_toroidal = B0 * R0 / R  (1/R dependence)
    B_poloidal ≈ ε * B0  (simplified, constant)

    Parameters:
    -----------
    R, Z, phi : float
        Cylindrical coordinates [m, m, rad]
    R0 : float
        Major radius [m]
    B0 : float
        Magnetic field at R0 [T]
    epsilon : float
        Inverse aspect ratio (a/R0), determines poloidal field strength

    Returns:
    --------
    B : array (3,)
        Magnetic field in cylindrical coordinates [B_R, B_Z, B_phi]
    """
    # Toroidal field (dominant)
    B_phi = B0 * R0 / R

    # Poloidal field (simplified as constant, pointing in Z direction)
    # In reality, B_poloidal varies with position
    B_R = 0.0
    B_Z = epsilon * B0

    return np.array([B_R, B_Z, B_phi])


def cylindrical_to_cartesian(R, Z, phi):
    """Convert cylindrical to Cartesian coordinates."""
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = Z
    return np.array([x, y, z])


def cartesian_to_cylindrical(x, y, z):
    """Convert Cartesian to cylindrical coordinates."""
    R = np.sqrt(x**2 + y**2)
    Z = z
    phi = np.arctan2(y, x)
    return R, Z, phi


def cyl_field_to_cart(B_cyl, phi):
    """
    Convert magnetic field from cylindrical to Cartesian coordinates.

    B_cyl = [B_R, B_Z, B_phi]
    """
    B_R, B_Z, B_phi = B_cyl

    B_x = B_R * np.cos(phi) - B_phi * np.sin(phi)
    B_y = B_R * np.sin(phi) + B_phi * np.cos(phi)
    B_z = B_Z

    return np.array([B_x, B_y, B_z])


def simulate_tokamak_orbit(q, m, R0, B0, epsilon, v0_cyl, x0_cyl, dt, n_steps):
    """
    Simulate particle orbit in tokamak.

    Parameters:
    -----------
    q, m : float
        Charge and mass
    R0, B0 : float
        Tokamak major radius and field
    epsilon : float
        Inverse aspect ratio
    v0_cyl : array (3,)
        Initial velocity in cylindrical coords [v_R, v_Z, v_phi]
    x0_cyl : array (3,)
        Initial position in cylindrical coords [R, Z, phi]
    dt : float
        Timestep
    n_steps : int
        Number of steps

    Returns:
    --------
    trajectory : dict
    """
    # Initialize in Cartesian coordinates
    x = np.zeros((n_steps, 3))
    v = np.zeros((n_steps, 3))
    t = np.zeros(n_steps)

    R_traj = np.zeros(n_steps)
    Z_traj = np.zeros(n_steps)
    phi_traj = np.zeros(n_steps)

    # Initial conditions
    R0_pos, Z0_pos, phi0 = x0_cyl
    x[0] = cylindrical_to_cartesian(R0_pos, Z0_pos, phi0)

    # Convert initial velocity to Cartesian
    v_R, v_Z, v_phi = v0_cyl
    v[0] = np.array([
        v_R * np.cos(phi0) - v_phi * np.sin(phi0),
        v_R * np.sin(phi0) + v_phi * np.cos(phi0),
        v_Z
    ])

    R_traj[0] = R0_pos
    Z_traj[0] = Z0_pos
    phi_traj[0] = phi0

    for i in range(n_steps - 1):
        # Get current cylindrical coordinates
        R, Z, phi = cartesian_to_cylindrical(x[i, 0], x[i, 1], x[i, 2])

        # Get magnetic field in cylindrical coords
        B_cyl = tokamak_field(R, Z, phi, R0, B0, epsilon)

        # Convert to Cartesian
        B_cart = cyl_field_to_cart(B_cyl, phi)

        # Push particle
        x[i+1], v[i+1] = boris_push(x[i], v[i], q, m, B_cart, dt)
        t[i+1] = t[i] + dt

        # Store cylindrical coordinates
        R_traj[i+1], Z_traj[i+1], phi_traj[i+1] = cartesian_to_cylindrical(
            x[i+1, 0], x[i+1, 1], x[i+1, 2]
        )

    return {
        'x': x[:, 0], 'y': x[:, 1], 'z': x[:, 2],
        'vx': v[:, 0], 'vy': v[:, 1], 'vz': v[:, 2],
        'R': R_traj, 'Z': Z_traj, 'phi': phi_traj,
        't': t
    }


def plot_banana_orbit():
    """Demonstrate banana orbit (trapped particle)."""

    # Tokamak parameters
    R0 = 1.0     # Major radius [m]
    B0 = 2.0     # Magnetic field [T]
    epsilon = 0.3  # Inverse aspect ratio (a/R0)
    a = epsilon * R0  # Minor radius

    print(f"\nTokamak Parameters:")
    print(f"  Major radius R0 = {R0} m")
    print(f"  Minor radius a = {a} m")
    print(f"  Magnetic field B0 = {B0} T")
    print(f"  Inverse aspect ratio ε = {epsilon}")

    # Particle parameters
    q = -e
    m = m_e

    # Trapped particle: start at outer midplane with velocity mostly poloidal
    R_start = R0 + 0.5 * a  # Start at outer part of torus
    Z_start = 0.0
    phi_start = 0.0

    # Velocity: small parallel, large perpendicular (trapped condition)
    v_thermal = 1e7  # m/s
    v_parallel = 0.3 * v_thermal  # Small parallel velocity
    v_perp = np.sqrt(v_thermal**2 - v_parallel**2)

    # Initial velocity in cylindrical coords
    v0_cyl = np.array([0.0, v_perp, v_parallel])
    x0_cyl = np.array([R_start, Z_start, phi_start])

    # Time parameters
    omega_c = abs(q) * B0 / m
    T_c = 2 * np.pi / omega_c
    dt = T_c / 50
    n_steps = 5000

    # Simulate
    traj = simulate_tokamak_orbit(q, m, R0, B0, epsilon, v0_cyl, x0_cyl, dt, n_steps)

    # Create figure
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(traj['x'], traj['y'], traj['z'], 'b-', linewidth=1, alpha=0.7)
    ax1.plot([traj['x'][0]], [traj['y'][0]], [traj['z'][0]],
             'go', markersize=10, label='Start')

    # Draw torus outline
    theta_tor = np.linspace(0, 2*np.pi, 100)
    for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
        x_tor = (R0 + a * np.cos(theta_tor)) * np.cos(phi)
        y_tor = (R0 + a * np.cos(theta_tor)) * np.sin(phi)
        z_tor = a * np.sin(theta_tor)
        ax1.plot(x_tor, y_tor, z_tor, 'k--', alpha=0.3, linewidth=1)

    ax1.set_xlabel('x [m]', fontsize=11)
    ax1.set_ylabel('y [m]', fontsize=11)
    ax1.set_zlabel('z [m]', fontsize=11)
    ax1.set_title('3D Banana Orbit', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)

    # Plot 2: Poloidal cross-section (R-Z plane)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(traj['R'], traj['Z'], 'b-', linewidth=2, alpha=0.7)
    ax2.plot(traj['R'][0], traj['Z'][0], 'go', markersize=10, label='Start')

    # Draw plasma boundary (circular cross-section)
    theta = np.linspace(0, 2*np.pi, 100)
    R_boundary = R0 + a * np.cos(theta)
    Z_boundary = a * np.sin(theta)
    ax2.plot(R_boundary, Z_boundary, 'k--', linewidth=2, label='Plasma boundary')
    ax2.plot(R0, 0, 'r*', markersize=15, label='Magnetic axis')

    ax2.set_xlabel('R [m]', fontsize=12)
    ax2.set_ylabel('Z [m]', fontsize=12)
    ax2.set_title('Poloidal Cross-Section (Banana Shape)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Plot 3: Toroidal angle evolution
    ax3 = fig.add_subplot(2, 3, 3)
    # Unwrap phi to avoid discontinuities
    phi_unwrapped = np.unwrap(traj['phi'])
    ax3.plot(traj['t'] * 1e6, phi_unwrapped * 180/np.pi, 'b-', linewidth=2)

    ax3.set_xlabel('Time [μs]', fontsize=12)
    ax3.set_ylabel('Toroidal Angle φ [degrees]', fontsize=12)
    ax3.set_title('Toroidal Motion', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: R vs time (shows banana tips)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(traj['t'] * 1e6, traj['R'], 'b-', linewidth=2)
    ax4.axhline(R0 + a, color='red', linestyle='--', linewidth=2, label='Outer boundary')
    ax4.axhline(R0 - a, color='red', linestyle='--', linewidth=2, label='Inner boundary')
    ax4.axhline(R0, color='green', linestyle=':', linewidth=2, label='Magnetic axis')

    ax4.set_xlabel('Time [μs]', fontsize=12)
    ax4.set_ylabel('Major Radius R [m]', fontsize=12)
    ax4.set_title('Radial Excursion', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Z vs time (poloidal oscillation)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(traj['t'] * 1e6, traj['Z'], 'b-', linewidth=2)
    ax5.axhline(a, color='red', linestyle='--', linewidth=2, label='Upper boundary')
    ax5.axhline(-a, color='red', linestyle='--', linewidth=2, label='Lower boundary')

    ax5.set_xlabel('Time [μs]', fontsize=12)
    ax5.set_ylabel('Z [m]', fontsize=12)
    ax5.set_title('Vertical Motion', fontsize=13, fontweight='bold')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Banana width
    ax6 = fig.add_subplot(2, 3, 6)
    # Calculate banana width from R excursion
    R_max = np.max(traj['R'])
    R_min = np.min(traj['R'])
    banana_width = R_max - R_min

    ax6.hist(traj['R'], bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax6.axvline(R_max, color='red', linestyle='--', linewidth=2,
                label=f'Width = {banana_width*100:.2f} cm')
    ax6.axvline(R_min, color='red', linestyle='--', linewidth=2)

    ax6.set_xlabel('R [m]', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.set_title('Banana Width Distribution', fontsize=13, fontweight='bold')
    ax6.legend(loc='best', fontsize=10)
    ax6.grid(True, alpha=0.3)

    print(f"\nBanana Orbit Characteristics:")
    print(f"  Banana width: {banana_width*100:.2f} cm")
    print(f"  R range: [{R_min:.3f}, {R_max:.3f}] m")
    print(f"  Z range: [{np.min(traj['Z']):.3f}, {np.max(traj['Z']):.3f}] m")

    plt.tight_layout()
    plt.savefig('banana_orbit.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_passing_vs_trapped():
    """Compare passing and trapped orbits."""

    # Tokamak parameters
    R0 = 1.0
    B0 = 2.0
    epsilon = 0.3
    a = epsilon * R0

    # Particle parameters
    q = -e
    m = m_e

    # Initial position
    R_start = R0 + 0.5 * a
    Z_start = 0.0
    phi_start = 0.0

    v_thermal = 1e7  # m/s

    # Two cases: trapped vs passing
    # Trapped: small v_parallel (< critical)
    # Passing: large v_parallel (> critical)

    cases = [
        ('Trapped', 0.2 * v_thermal),   # Small parallel velocity
        ('Passing', 0.8 * v_thermal),   # Large parallel velocity
    ]

    omega_c = abs(q) * B0 / m
    T_c = 2 * np.pi / omega_c
    dt = T_c / 50
    n_steps = 5000

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['blue', 'red']

    for i, (label, v_para) in enumerate(cases):
        v_perp = np.sqrt(v_thermal**2 - v_para**2)
        v0_cyl = np.array([0.0, v_perp, v_para])
        x0_cyl = np.array([R_start, Z_start, phi_start])

        traj = simulate_tokamak_orbit(q, m, R0, B0, epsilon, v0_cyl, x0_cyl, dt, n_steps)

        # Plot in R-Z plane
        ax = axes[0]
        ax.plot(traj['R'], traj['Z'], color=colors[i], linewidth=2,
                alpha=0.7, label=f'{label} (v_∥/v_th = {v_para/v_thermal:.1f})')

    # Draw plasma boundary
    ax = axes[0]
    theta = np.linspace(0, 2*np.pi, 100)
    R_boundary = R0 + a * np.cos(theta)
    Z_boundary = a * np.sin(theta)
    ax.plot(R_boundary, Z_boundary, 'k--', linewidth=2, label='Plasma boundary')
    ax.plot(R0, 0, 'g*', markersize=15, label='Magnetic axis')

    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title('Passing vs Trapped Orbits', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Second plot: phi vs time
    ax = axes[1]
    for i, (label, v_para) in enumerate(cases):
        v_perp = np.sqrt(v_thermal**2 - v_para**2)
        v0_cyl = np.array([0.0, v_perp, v_para])
        x0_cyl = np.array([R_start, Z_start, phi_start])

        traj = simulate_tokamak_orbit(q, m, R0, B0, epsilon, v0_cyl, x0_cyl, dt, n_steps)

        phi_unwrapped = np.unwrap(traj['phi'])
        ax.plot(traj['t'] * 1e6, phi_unwrapped, color=colors[i],
                linewidth=2, label=label)

    ax.set_xlabel('Time [μs]', fontsize=12)
    ax.set_ylabel('Toroidal Angle φ [rad]', fontsize=12)
    ax.set_title('Toroidal Circulation', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('passing_vs_trapped.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("BANANA ORBIT SIMULATION IN TOKAMAK")
    print("="*80)

    print("\nPart 1: Banana Orbit (Trapped Particle)")
    print("-" * 80)
    plot_banana_orbit()

    print("\nPart 2: Passing vs Trapped Comparison")
    print("-" * 80)
    plot_passing_vs_trapped()

    print("\nKey Points:")
    print("  - Trapped particles: small v_∥, banana-shaped orbits")
    print("  - Passing particles: large v_∥, circulate around torus")
    print("  - Banana width depends on v_∥ and radial position")

    print("\nDone! Generated 2 figures:")
    print("  - banana_orbit.png")
    print("  - passing_vs_trapped.png")
