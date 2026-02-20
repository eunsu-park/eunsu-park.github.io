#!/usr/bin/env python3
"""
Magnetic Mirror Simulation

This script simulates particle motion in a magnetic mirror/bottle configuration.
Demonstrates trapping, bounce motion, loss cone, and conservation of the
magnetic moment (adiabatic invariant).

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e


def boris_push(x, v, q, m, E, B, dt):
    """Boris algorithm for particle pushing."""
    v_minus = v + (q * E / m) * (dt / 2)

    t = (q * dt / (2 * m)) * B
    t_mag2 = np.dot(t, t)
    s = 2 * t / (1 + t_mag2)

    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    v_new = v_plus + (q * E / m) * (dt / 2)
    x_new = x + v_new * dt

    return x_new, v_new


def magnetic_mirror_field(z, B0, L):
    """
    Magnetic mirror field: B_z(z) = B0 * (1 + (z/L)²)

    Parameters:
    -----------
    z : float or array
        Position along z [m]
    B0 : float
        Minimum field at center [T]
    L : float
        Mirror scale length [m]

    Returns:
    --------
    B : array (3,) or (N, 3)
        Magnetic field [T]
    """
    B_z = B0 * (1 + (z / L)**2)

    if np.isscalar(z):
        return np.array([0, 0, B_z])
    else:
        return np.column_stack([np.zeros_like(z), np.zeros_like(z), B_z])


def simulate_mirror(q, m, B0, L, v0, x0, dt, n_steps, max_z=None):
    """
    Simulate particle in magnetic mirror.

    Parameters:
    -----------
    q, m : float
        Charge and mass
    B0, L : float
        Mirror parameters
    v0, x0 : array
        Initial velocity and position
    dt : float
        Timestep
    n_steps : int
        Maximum steps
    max_z : float
        Maximum |z| before particle is considered lost

    Returns:
    --------
    trajectory : dict
    """
    x = np.zeros((n_steps, 3))
    v = np.zeros((n_steps, 3))
    t = np.zeros(n_steps)
    B_field = np.zeros((n_steps, 3))
    mu = np.zeros(n_steps)  # Magnetic moment

    x[0] = x0
    v[0] = v0

    E = np.array([0, 0, 0])
    lost = False
    n_actual = n_steps

    for i in range(n_steps - 1):
        B = magnetic_mirror_field(x[i, 2], B0, L)
        B_field[i] = B
        B_mag = np.linalg.norm(B)

        # Calculate magnetic moment μ = m*v_perp²/(2*B)
        v_para = np.dot(v[i], B) / B_mag
        v_perp = np.sqrt(np.dot(v[i], v[i]) - v_para**2)
        mu[i] = m * v_perp**2 / (2 * B_mag)

        x[i+1], v[i+1] = boris_push(x[i], v[i], q, m, E, B, dt)
        t[i+1] = t[i] + dt

        # Check if particle is lost
        if max_z is not None and abs(x[i+1, 2]) > max_z:
            lost = True
            n_actual = i + 1
            break

    if not lost:
        B = magnetic_mirror_field(x[-1, 2], B0, L)
        B_field[-1] = B
        B_mag = np.linalg.norm(B)
        v_para = np.dot(v[-1], B) / B_mag
        v_perp = np.sqrt(np.dot(v[-1], v[-1]) - v_para**2)
        mu[-1] = m * v_perp**2 / (2 * B_mag)

    return {
        'x': x[:n_actual, 0], 'y': x[:n_actual, 1], 'z': x[:n_actual, 2],
        'vx': v[:n_actual, 0], 'vy': v[:n_actual, 1], 'vz': v[:n_actual, 2],
        't': t[:n_actual], 'B': B_field[:n_actual],
        'mu': mu[:n_actual], 'lost': lost
    }


def plot_trapped_vs_lost():
    """Demonstrate trapped particles vs lost particles."""

    # Parameters
    B0 = 0.1  # Tesla (minimum field at center)
    L = 0.5   # meters
    B_mirror = B0 * (1 + 1)**2  # Field at z = ±L

    mirror_ratio = B_mirror / B0

    q = -e
    m = m_e

    # Initial position
    x0 = np.array([0, 0, 0])

    # Test different pitch angles
    v_total = 1e7  # m/s (total speed)

    # Loss cone angle: sin²(θ_lc) = B0 / B_mirror
    sin_theta_lc = np.sqrt(B0 / B_mirror)
    theta_lc = np.arcsin(sin_theta_lc) * 180 / np.pi

    print(f"\nMagnetic Mirror Configuration:")
    print(f"  B0 (center) = {B0} T")
    print(f"  B_mirror (at z=±L) = {B_mirror} T")
    print(f"  Mirror ratio R = {mirror_ratio:.2f}")
    print(f"  Loss cone angle θ_lc = {theta_lc:.2f}°")

    # Pitch angles to test
    pitch_angles = [20, 40, 50, 60, 70, 80]  # degrees

    omega_c = abs(q) * B0 / m
    T_c = 2 * np.pi / omega_c
    dt = T_c / 100
    n_steps = 5000
    max_z = 2 * L

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(pitch_angles)))

    # Storage for loss cone plot
    v_perp_list = []
    v_para_list = []
    lost_list = []

    for pitch_angle, color in zip(pitch_angles, colors):
        theta_rad = pitch_angle * np.pi / 180
        v_perp = v_total * np.sin(theta_rad)
        v_para = v_total * np.cos(theta_rad)

        v0 = np.array([v_perp, 0, v_para])

        traj = simulate_mirror(q, m, B0, L, v0, x0, dt, n_steps, max_z)

        v_perp_list.append(v_perp)
        v_para_list.append(v_para)
        lost_list.append(traj['lost'])

        # Plot z vs t
        label = f"{pitch_angle}° {'(lost)' if traj['lost'] else '(trapped)'}"
        linestyle = '--' if traj['lost'] else '-'
        ax = axes[0, 0]
        ax.plot(traj['t'] * 1e6, traj['z'] * 1e2, linestyle=linestyle,
                linewidth=2, color=color, label=label)

    # Finish first plot
    ax = axes[0, 0]
    ax.axhline(L * 1e2, color='red', linestyle=':', linewidth=2, label='Mirror location')
    ax.axhline(-L * 1e2, color='red', linestyle=':', linewidth=2)
    ax.set_xlabel('Time [μs]', fontsize=12)
    ax.set_ylabel('z Position [cm]', fontsize=12)
    ax.set_title('Trapped vs Lost Particles', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Loss cone in velocity space
    ax = axes[0, 1]

    # Plot particles
    for v_perp, v_para, lost in zip(v_perp_list, v_para_list, lost_list):
        marker = 'x' if lost else 'o'
        color = 'red' if lost else 'blue'
        size = 100
        ax.scatter(v_para / 1e6, v_perp / 1e6, marker=marker, s=size,
                  color=color, edgecolors='black', linewidth=1.5)

    # Draw loss cone boundary
    v_para_cone = np.linspace(0, v_total / 1e6, 100)
    v_perp_cone = v_para_cone * np.tan(theta_lc * np.pi / 180)
    ax.plot(v_para_cone, v_perp_cone, 'g--', linewidth=3,
            label=f'Loss cone θ_lc = {theta_lc:.1f}°')
    ax.plot(v_para_cone, -v_perp_cone, 'g--', linewidth=3)

    # Draw total velocity circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot((v_total / 1e6) * np.cos(theta), (v_total / 1e6) * np.sin(theta),
            'k:', linewidth=2, alpha=0.5, label='|v| = const')

    ax.set_xlabel('v_∥ [10⁶ m/s]', fontsize=12)
    ax.set_ylabel('v_⊥ [10⁶ m/s]', fontsize=12)
    ax.set_title('Loss Cone in Velocity Space', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 3: Magnetic field profile
    ax = axes[1, 0]
    z_profile = np.linspace(-1.5*L, 1.5*L, 200)
    B_profile = B0 * (1 + (z_profile / L)**2)

    ax.plot(z_profile * 1e2, B_profile, 'b-', linewidth=3)
    ax.axvline(L * 1e2, color='red', linestyle='--', linewidth=2, label='Mirror points')
    ax.axvline(-L * 1e2, color='red', linestyle='--', linewidth=2)
    ax.axhline(B0, color='green', linestyle=':', linewidth=2, label=f'B0 = {B0} T')
    ax.axhline(B_mirror, color='orange', linestyle=':', linewidth=2,
               label=f'B_mirror = {B_mirror} T')

    ax.set_xlabel('z Position [cm]', fontsize=12)
    ax.set_ylabel('|B| [T]', fontsize=12)
    ax.set_title('Magnetic Field Profile', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Example trapped particle bounce
    theta_trapped = 60  # degrees (well above loss cone)
    v_perp = v_total * np.sin(theta_trapped * np.pi / 180)
    v_para = v_total * np.cos(theta_trapped * np.pi / 180)
    v0 = np.array([v_perp, 0, v_para])

    traj = simulate_mirror(q, m, B0, L, v0, x0, dt, n_steps, max_z)

    ax = axes[1, 1]
    # 2D projection
    ax.plot(traj['z'] * 1e2, traj['x'] * 1e2, 'b-', linewidth=1, alpha=0.7)
    ax.plot(traj['z'][0] * 1e2, traj['x'][0] * 1e2, 'go', markersize=10, label='Start')
    ax.axvline(L * 1e2, color='red', linestyle='--', linewidth=2, label='Mirror')
    ax.axvline(-L * 1e2, color='red', linestyle='--', linewidth=2)

    ax.set_xlabel('z [cm]', fontsize=12)
    ax.set_ylabel('x [cm]', fontsize=12)
    ax.set_title(f'Trapped Particle (θ = {theta_trapped}°)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('magnetic_mirror_trapped_lost.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_adiabatic_invariant():
    """Verify conservation of magnetic moment (adiabatic invariant)."""

    # Parameters
    B0 = 0.1  # Tesla
    L = 0.5   # meters

    q = -e
    m = m_e

    # Trapped particle
    v_total = 1e7  # m/s
    theta = 60 * np.pi / 180  # Pitch angle
    v_perp = v_total * np.sin(theta)
    v_para = v_total * np.cos(theta)

    v0 = np.array([v_perp, 0, v_para])
    x0 = np.array([0, 0, 0])

    omega_c = abs(q) * B0 / m
    T_c = 2 * np.pi / omega_c
    dt = T_c / 100
    n_steps = 5000

    traj = simulate_mirror(q, m, B0, L, v0, x0, dt, n_steps)

    # Calculate B magnitude along trajectory
    B_mag = np.linalg.norm(traj['B'], axis=1)

    # Calculate v_perp and v_para
    v_perp_traj = np.zeros(len(traj['t']))
    v_para_traj = np.zeros(len(traj['t']))

    for i in range(len(traj['t'])):
        v_vec = np.array([traj['vx'][i], traj['vy'][i], traj['vz'][i]])
        B_vec = traj['B'][i]
        B_norm = np.linalg.norm(B_vec)

        v_para_traj[i] = np.dot(v_vec, B_vec) / B_norm
        v_perp_traj[i] = np.sqrt(np.dot(v_vec, v_vec) - v_para_traj[i]**2)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Bounce trajectory
    ax = axes[0, 0]
    ax.plot(traj['t'] * 1e6, traj['z'] * 1e2, 'b-', linewidth=2)
    ax.axhline(L * 1e2, color='red', linestyle='--', linewidth=2, label='Mirror points')
    ax.axhline(-L * 1e2, color='red', linestyle='--', linewidth=2)

    ax.set_xlabel('Time [μs]', fontsize=12)
    ax.set_ylabel('z Position [cm]', fontsize=12)
    ax.set_title('Bounce Motion', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Magnetic moment conservation
    ax = axes[0, 1]
    mu_normalized = traj['mu'] / traj['mu'][0]
    ax.plot(traj['t'] * 1e6, mu_normalized, 'b-', linewidth=2)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect conservation')

    ax.set_xlabel('Time [μs]', fontsize=12)
    ax.set_ylabel('μ / μ₀', fontsize=12)
    ax.set_title('Magnetic Moment Conservation', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.995, 1.005])

    # Print conservation statistics
    mu_error = np.abs(traj['mu'] - traj['mu'][0]) / traj['mu'][0]
    print(f"\nMagnetic Moment Conservation:")
    print(f"  Maximum error: {np.max(mu_error) * 100:.4f}%")
    print(f"  Mean error: {np.mean(mu_error) * 100:.4f}%")

    # Plot 3: v_perp vs B (should follow v_perp² ∝ B)
    ax = axes[1, 0]
    ax.plot(B_mag, v_perp_traj / 1e6, 'b.', markersize=2, alpha=0.5, label='Simulation')

    # Theoretical: v_perp² = 2μB/m, so v_perp = sqrt(2μB/m)
    B_theory = np.linspace(B_mag.min(), B_mag.max(), 100)
    mu_avg = np.mean(traj['mu'])
    v_perp_theory = np.sqrt(2 * mu_avg * B_theory / m)
    ax.plot(B_theory, v_perp_theory / 1e6, 'r-', linewidth=3, label='Theory: v_⊥ ∝ √B')

    ax.set_xlabel('|B| [T]', fontsize=12)
    ax.set_ylabel('v_⊥ [10⁶ m/s]', fontsize=12)
    ax.set_title('v_⊥ vs B (Adiabatic Relation)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: v_para vs z (reverses at mirror points)
    ax = axes[1, 1]
    ax.plot(traj['z'] * 1e2, v_para_traj / 1e6, 'b-', linewidth=2)
    ax.axvline(L * 1e2, color='red', linestyle='--', linewidth=2, label='Mirror points')
    ax.axvline(-L * 1e2, color='red', linestyle='--', linewidth=2)
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)

    ax.set_xlabel('z Position [cm]', fontsize=12)
    ax.set_ylabel('v_∥ [10⁶ m/s]', fontsize=12)
    ax.set_title('Parallel Velocity (Reverses at Mirror)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('magnetic_mirror_adiabatic_invariant.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("MAGNETIC MIRROR SIMULATION")
    print("="*80)

    print("\nPart 1: Trapped vs Lost Particles")
    print("-" * 80)
    plot_trapped_vs_lost()

    print("\nPart 2: Adiabatic Invariant (Magnetic Moment Conservation)")
    print("-" * 80)
    plot_adiabatic_invariant()

    print("\nDone! Generated 2 figures:")
    print("  - magnetic_mirror_trapped_lost.png")
    print("  - magnetic_mirror_adiabatic_invariant.png")
