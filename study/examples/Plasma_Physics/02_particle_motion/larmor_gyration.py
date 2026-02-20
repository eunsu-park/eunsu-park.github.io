#!/usr/bin/env python3
"""
Larmor Gyration Simulation

This script simulates charged particle gyration in a uniform magnetic field
using the Boris algorithm. Demonstrates circular and helical orbits for
electrons and protons, and verifies energy conservation.

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import e, m_e, m_p, c


def boris_push(x, v, q, m, B, dt):
    """
    Boris algorithm for particle pushing in electromagnetic fields.

    This is the standard algorithm used in PIC codes for its superior
    energy conservation properties.

    Parameters:
    -----------
    x : array (3,)
        Position [m]
    v : array (3,)
        Velocity [m/s]
    q : float
        Charge [C]
    m : float
        Mass [kg]
    B : array (3,)
        Magnetic field [T]
    dt : float
        Timestep [s]

    Returns:
    --------
    x_new, v_new : Updated position and velocity
    """
    # Half acceleration (no E field in this case)
    v_minus = v.copy()

    # Rotation
    t = (q * dt / (2 * m)) * B
    t_mag2 = np.dot(t, t)
    s = 2 * t / (1 + t_mag2)

    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    # Half acceleration again
    v_new = v_plus

    # Update position
    x_new = x + v_new * dt

    return x_new, v_new


def simulate_gyration(q, m, B, v0, x0, dt, n_steps):
    """
    Simulate particle gyration in uniform magnetic field.

    Parameters:
    -----------
    q : float
        Charge [C]
    m : float
        Mass [kg]
    B : array (3,)
        Magnetic field [T]
    v0 : array (3,)
        Initial velocity [m/s]
    x0 : array (3,)
        Initial position [m]
    dt : float
        Timestep [s]
    n_steps : int
        Number of steps

    Returns:
    --------
    trajectory : dict with 'x', 'y', 'z', 'vx', 'vy', 'vz', 't'
    """
    # Initialize arrays
    x = np.zeros((n_steps, 3))
    v = np.zeros((n_steps, 3))
    t = np.zeros(n_steps)

    x[0] = x0
    v[0] = v0

    # Time integration
    for i in range(n_steps - 1):
        x[i+1], v[i+1] = boris_push(x[i], v[i], q, m, B, dt)
        t[i+1] = t[i] + dt

    return {
        'x': x[:, 0], 'y': x[:, 1], 'z': x[:, 2],
        'vx': v[:, 0], 'vy': v[:, 1], 'vz': v[:, 2],
        't': t
    }


def theoretical_gyroradius(v_perp, q, m, B):
    """Calculate theoretical Larmor radius."""
    omega_c = abs(q) * B / m
    return v_perp / omega_c


def theoretical_gyrofrequency(q, m, B):
    """Calculate theoretical gyrofrequency."""
    return abs(q) * B / m


def plot_circular_orbit():
    """Plot circular orbit (no parallel velocity)."""

    # Parameters
    B0 = 1.0  # Tesla
    B = np.array([0, 0, B0])

    # Electron
    q_e = -e
    m_e_kg = m_e
    v0_e = np.array([1e6, 0, 0])  # 1000 km/s perpendicular
    x0 = np.array([0, 0, 0])

    # Proton
    q_p = e
    m_p_kg = m_p
    v0_p = np.array([1e5, 0, 0])  # 100 km/s perpendicular

    # Time parameters
    omega_ce = abs(q_e) * B0 / m_e_kg
    T_ce = 2 * np.pi / omega_ce
    dt = T_ce / 100
    n_steps = 300

    # Simulate
    traj_e = simulate_gyration(q_e, m_e_kg, B, v0_e, x0, dt, n_steps)
    traj_p = simulate_gyration(q_p, m_p_kg, B, v0_p, x0, dt, n_steps)

    # Theoretical values
    r_L_e = theoretical_gyroradius(np.linalg.norm(v0_e), q_e, m_e_kg, B0)
    r_L_p = theoretical_gyroradius(np.linalg.norm(v0_p), q_p, m_p_kg, B0)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Electron orbit
    ax1.plot(traj_e['x'] * 1e3, traj_e['y'] * 1e3, 'b-', linewidth=2, label='Electron')
    ax1.plot(traj_e['x'][0] * 1e3, traj_e['y'][0] * 1e3, 'go', markersize=10, label='Start')
    ax1.plot(0, 0, 'r+', markersize=15, markeredgewidth=3, label='Guiding center')

    # Draw circle with theoretical radius
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(r_L_e * 1e3 * np.cos(theta), r_L_e * 1e3 * np.sin(theta),
             'r--', alpha=0.5, linewidth=2, label=f'Theory: r_L = {r_L_e*1e3:.3f} mm')

    ax1.set_xlabel('x [mm]', fontsize=12)
    ax1.set_ylabel('y [mm]', fontsize=12)
    ax1.set_title('Electron Larmor Gyration (B = 1 T)', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.arrow(0.8 * r_L_e * 1e3, 0, 0, 0.3 * r_L_e * 1e3, head_width=0.1 * r_L_e * 1e3,
              head_length=0.05 * r_L_e * 1e3, fc='black', ec='black')
    ax1.text(0.85 * r_L_e * 1e3, 0.2 * r_L_e * 1e3, 'B', fontsize=14, fontweight='bold')

    # Proton orbit
    ax2.plot(traj_p['x'] * 1e3, traj_p['y'] * 1e3, 'r-', linewidth=2, label='Proton')
    ax2.plot(traj_p['x'][0] * 1e3, traj_p['y'][0] * 1e3, 'go', markersize=10, label='Start')
    ax2.plot(0, 0, 'b+', markersize=15, markeredgewidth=3, label='Guiding center')

    # Draw circle with theoretical radius
    ax2.plot(r_L_p * 1e3 * np.cos(theta), r_L_p * 1e3 * np.sin(theta),
             'b--', alpha=0.5, linewidth=2, label=f'Theory: r_L = {r_L_p*1e3:.1f} mm')

    ax2.set_xlabel('x [mm]', fontsize=12)
    ax2.set_ylabel('y [mm]', fontsize=12)
    ax2.set_title('Proton Larmor Gyration (B = 1 T)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.arrow(0.8 * r_L_p * 1e3, 0, 0, 0.3 * r_L_p * 1e3, head_width=2,
              head_length=1, fc='black', ec='black')
    ax2.text(0.85 * r_L_p * 1e3, 0.2 * r_L_p * 1e3, 'B', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('larmor_circular_orbit.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_helical_orbit():
    """Plot 3D helical orbit (with parallel velocity)."""

    # Parameters
    B0 = 1.0  # Tesla
    B = np.array([0, 0, B0])

    q_e = -e
    m_e_kg = m_e

    # Initial velocity with both perpendicular and parallel components
    v_perp = 1e6  # m/s
    v_para = 5e5  # m/s
    v0 = np.array([v_perp, 0, v_para])
    x0 = np.array([0, 0, 0])

    # Time parameters
    omega_ce = abs(q_e) * B0 / m_e_kg
    T_ce = 2 * np.pi / omega_ce
    dt = T_ce / 100
    n_steps = 500

    # Simulate
    traj = simulate_gyration(q_e, m_e_kg, B, v0, x0, dt, n_steps)

    # Theoretical values
    r_L = theoretical_gyroradius(v_perp, q_e, m_e_kg, B0)
    pitch = v_para * T_ce

    # 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(traj['x'] * 1e3, traj['y'] * 1e3, traj['z'] * 1e3,
            'b-', linewidth=2, label='Electron trajectory')
    ax.plot([traj['x'][0] * 1e3], [traj['y'][0] * 1e3], [traj['z'][0] * 1e3],
            'go', markersize=10, label='Start')
    ax.plot([0], [0], traj['z'] * 1e3, 'r--', alpha=0.5, linewidth=2,
            label='Guiding center line')

    ax.set_xlabel('x [mm]', fontsize=12)
    ax.set_ylabel('y [mm]', fontsize=12)
    ax.set_zlabel('z [mm]', fontsize=12)
    ax.set_title(f'Helical Orbit in Uniform B Field\n'
                 f'v_⊥ = {v_perp/1e3:.0f} km/s, v_∥ = {v_para/1e3:.0f} km/s',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)

    # Add annotations
    ax.text2D(0.05, 0.95, f'Larmor radius: {r_L*1e3:.3f} mm\n'
                          f'Pitch: {pitch*1e3:.2f} mm\n'
                          f'Gyroperiod: {T_ce*1e9:.2f} ns',
              transform=ax.transAxes, fontsize=11,
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('larmor_helical_orbit.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_energy_conservation():
    """Verify energy conservation with Boris algorithm."""

    # Parameters
    B0 = 1.0  # Tesla
    B = np.array([0, 0, B0])

    q_e = -e
    m_e_kg = m_e
    v0 = np.array([1e6, 5e5, 3e5])  # General velocity
    x0 = np.array([0, 0, 0])

    # Time parameters
    omega_ce = abs(q_e) * B0 / m_e_kg
    T_ce = 2 * np.pi / omega_ce

    # Test different timesteps
    dt_factors = [0.01, 0.05, 0.1, 0.2]
    n_periods = 20

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for dt_factor in dt_factors:
        dt = T_ce * dt_factor
        n_steps = int(n_periods * T_ce / dt)

        traj = simulate_gyration(q_e, m_e_kg, B, v0, x0, dt, n_steps)

        # Calculate kinetic energy
        v_squared = traj['vx']**2 + traj['vy']**2 + traj['vz']**2
        KE = 0.5 * m_e_kg * v_squared
        KE_relative = (KE - KE[0]) / KE[0]

        # Plot
        ax1.plot(traj['t'] / T_ce, v_squared / v_squared[0],
                linewidth=2, label=f'dt = {dt_factor:.2f} T_ce')
        ax2.semilogy(traj['t'] / T_ce, np.abs(KE_relative),
                     linewidth=2, label=f'dt = {dt_factor:.2f} T_ce')

    ax1.set_xlabel('Time [gyroperiods]', fontsize=12)
    ax1.set_ylabel('|v|² / |v₀|²', fontsize=12)
    ax1.set_title('Velocity Magnitude Conservation', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.999, 1.001])

    ax2.set_xlabel('Time [gyroperiods]', fontsize=12)
    ax2.set_ylabel('|ΔKE / KE₀|', fontsize=12)
    ax2.set_title('Relative Energy Error (Boris Algorithm)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('larmor_energy_conservation.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_gyrofrequency_verification():
    """Verify gyrofrequency matches theory."""

    # Parameters
    B0 = 1.0  # Tesla
    B = np.array([0, 0, B0])

    q_e = -e
    m_e_kg = m_e
    v0 = np.array([1e6, 0, 0])
    x0 = np.array([0, 0, 0])

    # Theoretical frequency
    omega_ce = abs(q_e) * B0 / m_e_kg
    f_ce = omega_ce / (2 * np.pi)
    T_ce = 1 / f_ce

    # Simulate for several periods
    dt = T_ce / 100
    n_steps = 500
    traj = simulate_gyration(q_e, m_e_kg, B, v0, x0, dt, n_steps)

    # Find peaks in x position (gyroperiods)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(traj['x'])

    if len(peaks) > 1:
        measured_periods = np.diff(traj['t'][peaks])
        T_measured = np.mean(measured_periods)
        f_measured = 1 / T_measured
    else:
        T_measured = np.nan
        f_measured = np.nan

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    # Position vs time
    ax1.plot(traj['t'] * 1e9, traj['x'] * 1e3, 'b-', linewidth=2, label='x(t)')
    ax1.plot(traj['t'] * 1e9, traj['y'] * 1e3, 'r-', linewidth=2, label='y(t)')
    if len(peaks) > 0:
        ax1.plot(traj['t'][peaks] * 1e9, traj['x'][peaks] * 1e3, 'go',
                markersize=8, label='Peaks')

    ax1.set_xlabel('Time [ns]', fontsize=12)
    ax1.set_ylabel('Position [mm]', fontsize=12)
    ax1.set_title('Position vs Time', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Velocity phase space
    ax2.plot(traj['vx'] / 1e6, traj['vy'] / 1e6, 'b-', linewidth=2)
    ax2.plot(traj['vx'][0] / 1e6, traj['vy'][0] / 1e6, 'go', markersize=10, label='Start')

    ax2.set_xlabel('v_x [10⁶ m/s]', fontsize=12)
    ax2.set_ylabel('v_y [10⁶ m/s]', fontsize=12)
    ax2.set_title('Velocity Phase Space', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Add text with frequencies
    if not np.isnan(f_measured):
        textstr = f'Theory: f_ce = {f_ce/1e9:.4f} GHz\n' \
                  f'Measured: f = {f_measured/1e9:.4f} GHz\n' \
                  f'Error: {abs(f_ce - f_measured)/f_ce * 100:.4f}%'
    else:
        textstr = f'Theory: f_ce = {f_ce/1e9:.4f} GHz'

    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('larmor_frequency_verification.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("LARMOR GYRATION SIMULATION (Boris Algorithm)")
    print("="*80 + "\n")

    # Print theoretical values
    B0 = 1.0
    print(f"Magnetic field: B = {B0} T\n")

    omega_ce = e * B0 / m_e
    omega_cp = e * B0 / m_p
    f_ce = omega_ce / (2 * np.pi)
    f_cp = omega_cp / (2 * np.pi)

    print("Electron:")
    print(f"  Gyrofrequency: {f_ce/1e9:.4f} GHz")
    print(f"  Gyroperiod: {1/f_ce*1e9:.4f} ns")

    v_perp = 1e6  # m/s
    r_Le = v_perp / omega_ce
    print(f"  Larmor radius (v_perp = {v_perp/1e6:.1f} Mm/s): {r_Le*1e3:.4f} mm\n")

    print("Proton:")
    print(f"  Gyrofrequency: {f_cp/1e6:.4f} MHz")
    print(f"  Gyroperiod: {1/f_cp*1e6:.4f} μs")

    v_perp = 1e5  # m/s
    r_Lp = v_perp / omega_cp
    print(f"  Larmor radius (v_perp = {v_perp/1e3:.1f} km/s): {r_Lp*1e3:.2f} mm\n")

    # Generate plots
    print("Generating plots...")
    print("  1. Circular orbits (electron and proton)...")
    plot_circular_orbit()

    print("  2. Helical orbit (3D)...")
    plot_helical_orbit()

    print("  3. Energy conservation test...")
    plot_energy_conservation()

    print("  4. Gyrofrequency verification...")
    plot_gyrofrequency_verification()

    print("\nDone! Generated 4 figures:")
    print("  - larmor_circular_orbit.png")
    print("  - larmor_helical_orbit.png")
    print("  - larmor_energy_conservation.png")
    print("  - larmor_frequency_verification.png")
