#!/usr/bin/env python3
"""
E×B Drift Simulation

This script demonstrates the E×B drift of charged particles in crossed
electric and magnetic fields. Shows that the drift is charge-independent
and verifies the drift velocity formula.

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, m_p


def boris_push(x, v, q, m, E, B, dt):
    """
    Boris algorithm for particle pushing in electromagnetic fields.

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
    E : array (3,)
        Electric field [V/m]
    B : array (3,)
        Magnetic field [T]
    dt : float
        Timestep [s]

    Returns:
    --------
    x_new, v_new : Updated position and velocity
    """
    # Half acceleration from E field
    v_minus = v + (q * E / m) * (dt / 2)

    # Rotation from B field
    t = (q * dt / (2 * m)) * B
    t_mag2 = np.dot(t, t)
    s = 2 * t / (1 + t_mag2)

    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    # Half acceleration from E field
    v_new = v_plus + (q * E / m) * (dt / 2)

    # Update position
    x_new = x + v_new * dt

    return x_new, v_new


def simulate_particle(q, m, E, B, v0, x0, dt, n_steps):
    """
    Simulate particle motion in E and B fields.

    Parameters:
    -----------
    q : float
        Charge [C]
    m : float
        Mass [kg]
    E : array (3,)
        Electric field [V/m]
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
        x[i+1], v[i+1] = boris_push(x[i], v[i], q, m, E, B, dt)
        t[i+1] = t[i] + dt

    return {
        'x': x[:, 0], 'y': x[:, 1], 'z': x[:, 2],
        'vx': v[:, 0], 'vy': v[:, 1], 'vz': v[:, 2],
        't': t
    }


def plot_exb_drift():
    """Demonstrate E×B drift for electron and ion."""

    # Fields configuration
    E0 = 1000  # V/m
    B0 = 0.1   # T
    E = np.array([E0, 0, 0])  # E in x direction
    B = np.array([0, 0, B0])  # B in z direction

    # Theoretical E×B drift velocity
    v_ExB = np.cross(E, B) / B0**2
    v_ExB_mag = np.linalg.norm(v_ExB)

    print(f"E = {E0} V/m (x direction)")
    print(f"B = {B0} T (z direction)")
    print(f"E×B drift velocity: {v_ExB} m/s")
    print(f"Magnitude: {v_ExB_mag} m/s = {v_ExB_mag/1e3:.1f} km/s\n")

    # Particle parameters
    q_e = -e
    q_p = e
    m_e_kg = m_e
    m_p_kg = m_p

    # Initial conditions (small perpendicular velocity)
    v0 = np.array([0, 1e4, 0])  # Small v_y
    x0_e = np.array([0, 0, 0])
    x0_p = np.array([0, 1, 0])  # Offset for visibility

    # Time parameters
    omega_ce = abs(q_e) * B0 / m_e_kg
    T_ce = 2 * np.pi / omega_ce
    dt = T_ce / 100
    n_steps = 2000

    # Simulate
    traj_e = simulate_particle(q_e, m_e_kg, E, B, v0, x0_e, dt, n_steps)
    traj_p = simulate_particle(q_p, m_p_kg, E, B, v0, x0_p, dt, n_steps)

    # Calculate drift velocities from simulation
    # Use linear fit to position vs time
    t_start = int(0.2 * n_steps)  # Skip initial transient
    t_end = n_steps

    drift_e_y = np.polyfit(traj_e['t'][t_start:t_end],
                           traj_e['y'][t_start:t_end], 1)[0]
    drift_p_y = np.polyfit(traj_p['t'][t_start:t_end],
                           traj_p['y'][t_start:t_end], 1)[0]

    print(f"Simulated drift velocities:")
    print(f"  Electron: {drift_e_y:.2f} m/s (y direction)")
    print(f"  Proton: {drift_p_y:.2f} m/s (y direction)")
    print(f"  Theory: {v_ExB[1]:.2f} m/s (y direction)")
    print(f"  Error (electron): {abs(drift_e_y - v_ExB[1])/v_ExB[1] * 100:.2f}%")
    print(f"  Error (proton): {abs(drift_p_y - v_ExB[1])/v_ExB[1] * 100:.2f}%\n")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Trajectories in x-y plane
    ax1 = axes[0, 0]
    ax1.plot(traj_e['x'], traj_e['y'], 'b-', linewidth=1.5, label='Electron', alpha=0.7)
    ax1.plot(traj_p['x'], traj_p['y'], 'r-', linewidth=1.5, label='Proton', alpha=0.7)
    ax1.plot(traj_e['x'][0], traj_e['y'][0], 'go', markersize=10, label='Start (e⁻)')
    ax1.plot(traj_p['x'][0], traj_p['y'][0], 'mo', markersize=10, label='Start (p⁺)')

    # Draw drift velocity arrow
    y_mid = (traj_e['y'][0] + traj_p['y'][0]) / 2
    ax1.arrow(0, y_mid, 0, 0.3, head_width=0.05, head_length=0.05,
              fc='green', ec='green', linewidth=2)
    ax1.text(0.1, y_mid + 0.15, 'v_ExB', fontsize=12, fontweight='bold', color='green')

    # Draw field vectors
    ax1.arrow(-0.3, 0.5, 0.2, 0, head_width=0.05, head_length=0.03,
              fc='red', ec='red', linewidth=2)
    ax1.text(-0.15, 0.6, 'E', fontsize=12, fontweight='bold', color='red')

    ax1.set_xlabel('x [m]', fontsize=12)
    ax1.set_ylabel('y [m]', fontsize=12)
    ax1.set_title('E×B Drift Trajectories', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: y position vs time
    ax2 = axes[0, 1]
    ax2.plot(traj_e['t'] * 1e6, traj_e['y'], 'b-', linewidth=2, label='Electron')
    ax2.plot(traj_p['t'] * 1e6, traj_p['y'], 'r-', linewidth=2, label='Proton')

    # Plot linear fit lines
    t_fit = traj_e['t'][t_start:t_end]
    ax2.plot(t_fit * 1e6, drift_e_y * t_fit + traj_e['y'][t_start],
             'b--', linewidth=2, alpha=0.5, label=f'Fit: {drift_e_y:.1f} m/s')
    ax2.plot(t_fit * 1e6, drift_p_y * t_fit + traj_p['y'][t_start],
             'r--', linewidth=2, alpha=0.5, label=f'Fit: {drift_p_y:.1f} m/s')

    ax2.set_xlabel('Time [μs]', fontsize=12)
    ax2.set_ylabel('y Position [m]', fontsize=12)
    ax2.set_title('Drift in y Direction (Both Species)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Velocity components (electron)
    ax3 = axes[1, 0]
    ax3.plot(traj_e['t'] * 1e6, traj_e['vx'] / 1e3, 'b-', linewidth=2, label='v_x')
    ax3.plot(traj_e['t'] * 1e6, traj_e['vy'] / 1e3, 'r-', linewidth=2, label='v_y')
    ax3.axhline(v_ExB[1] / 1e3, color='green', linestyle='--', linewidth=2,
                label=f'v_ExB = {v_ExB[1]/1e3:.1f} km/s')

    ax3.set_xlabel('Time [μs]', fontsize=12)
    ax3.set_ylabel('Velocity [km/s]', fontsize=12)
    ax3.set_title('Electron Velocity Components', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Phase space (v_x vs v_y)
    ax4 = axes[1, 1]
    ax4.plot(traj_e['vx'] / 1e3, traj_e['vy'] / 1e3, 'b-',
             linewidth=1, alpha=0.5, label='Electron')
    ax4.plot(traj_p['vx'] / 1e3, traj_p['vy'] / 1e3, 'r-',
             linewidth=1, alpha=0.5, label='Proton')
    ax4.plot(0, v_ExB[1] / 1e3, 'g*', markersize=20,
             label='E×B drift velocity')

    ax4.set_xlabel('v_x [km/s]', fontsize=12)
    ax4.set_ylabel('v_y [km/s]', fontsize=12)
    ax4.set_title('Velocity Phase Space', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exb_drift.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_parallel_acceleration():
    """Demonstrate acceleration along B when E is parallel to B."""

    # Fields configuration: E parallel to B
    E0 = 1000  # V/m
    B0 = 0.1   # T
    E = np.array([0, 0, E0])  # E in z direction
    B = np.array([0, 0, B0])  # B in z direction

    print(f"\nParallel E field case:")
    print(f"E = {E0} V/m (z direction, parallel to B)")
    print(f"B = {B0} T (z direction)")

    # Particle parameters
    q_e = -e
    q_p = e
    m_e_kg = m_e
    m_p_kg = m_p

    # Initial conditions
    v0 = np.array([1e5, 0, 0])  # Initial perpendicular velocity
    x0_e = np.array([0, 0, 0])
    x0_p = np.array([0, 0, 0])

    # Time parameters
    omega_ce = abs(q_e) * B0 / m_e_kg
    T_ce = 2 * np.pi / omega_ce
    dt = T_ce / 100
    n_steps = 1000

    # Simulate
    traj_e = simulate_particle(q_e, m_e_kg, E, B, v0, x0_e, dt, n_steps)
    traj_p = simulate_particle(q_p, m_p_kg, E, B, v0, x0_p, dt, n_steps)

    # Expected acceleration
    a_e = q_e * E0 / m_e_kg
    a_p = q_p * E0 / m_p_kg

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: 3D helical trajectory with acceleration
    from mpl_toolkits.mplot3d import Axes3D
    ax1 = plt.subplot(2, 2, 1, projection='3d')

    ax1.plot(traj_e['x'] * 1e3, traj_e['y'] * 1e3, traj_e['z'] * 1e3,
             'b-', linewidth=2, label='Electron')
    ax1.plot([traj_e['x'][0] * 1e3], [traj_e['y'][0] * 1e3], [traj_e['z'][0] * 1e3],
             'go', markersize=10, label='Start')

    ax1.set_xlabel('x [mm]', fontsize=11)
    ax1.set_ylabel('y [mm]', fontsize=11)
    ax1.set_zlabel('z [mm]', fontsize=11)
    ax1.set_title('Electron Trajectory (E ∥ B)', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)

    # Plot 2: Parallel velocity vs time
    ax2 = axes[0, 1]
    ax2.plot(traj_e['t'] * 1e6, traj_e['vz'] / 1e3, 'b-', linewidth=2, label='Electron')
    ax2.plot(traj_p['t'] * 1e6, traj_p['vz'] / 1e3, 'r-', linewidth=2, label='Proton')

    # Expected from kinematics: v = v0 + at
    ax2.plot(traj_e['t'] * 1e6, (v0[2] + a_e * traj_e['t']) / 1e3,
             'b--', linewidth=2, alpha=0.5, label='Theory (e⁻)')
    ax2.plot(traj_p['t'] * 1e6, (v0[2] + a_p * traj_p['t']) / 1e3,
             'r--', linewidth=2, alpha=0.5, label='Theory (p⁺)')

    ax2.set_xlabel('Time [μs]', fontsize=12)
    ax2.set_ylabel('v_z (parallel) [km/s]', fontsize=12)
    ax2.set_title('Parallel Acceleration', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Perpendicular velocity (should remain constant in magnitude)
    ax3 = axes[1, 0]
    v_perp_e = np.sqrt(traj_e['vx']**2 + traj_e['vy']**2)
    v_perp_p = np.sqrt(traj_p['vx']**2 + traj_p['vy']**2)

    ax3.plot(traj_e['t'] * 1e6, v_perp_e / 1e3, 'b-', linewidth=2, label='Electron')
    ax3.plot(traj_p['t'] * 1e6, v_perp_p / 1e3, 'r-', linewidth=2, label='Proton')
    ax3.axhline(np.linalg.norm(v0) / 1e3, color='green', linestyle='--',
                linewidth=2, label='Initial v_⊥')

    ax3.set_xlabel('Time [μs]', fontsize=12)
    ax3.set_ylabel('|v_⊥| [km/s]', fontsize=12)
    ax3.set_title('Perpendicular Velocity (Conserved)', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Total kinetic energy
    ax4 = axes[1, 1]
    KE_e = 0.5 * m_e_kg * (traj_e['vx']**2 + traj_e['vy']**2 + traj_e['vz']**2)
    KE_p = 0.5 * m_p_kg * (traj_p['vx']**2 + traj_p['vy']**2 + traj_p['vz']**2)

    ax4.plot(traj_e['t'] * 1e6, KE_e / e, 'b-', linewidth=2, label='Electron')
    ax4.plot(traj_p['t'] * 1e6, KE_p / e, 'r-', linewidth=2, label='Proton')

    ax4.set_xlabel('Time [μs]', fontsize=12)
    ax4.set_ylabel('Kinetic Energy [eV]', fontsize=12)
    ax4.set_title('Energy Gain from E Field', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('parallel_acceleration.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("E×B DRIFT SIMULATION")
    print("="*80 + "\n")

    print("Part 1: E×B Drift (E ⊥ B)")
    print("-" * 80)
    plot_exb_drift()

    print("\nPart 2: Parallel Acceleration (E ∥ B)")
    print("-" * 80)
    plot_parallel_acceleration()

    print("\nDone! Generated 2 figures:")
    print("  - exb_drift.png")
    print("  - parallel_acceleration.png")
