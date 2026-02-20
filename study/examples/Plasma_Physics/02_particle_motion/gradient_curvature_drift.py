#!/usr/bin/env python3
"""
Gradient and Curvature Drift Simulation

This script demonstrates grad-B drift and curvature drift in non-uniform
magnetic fields. Shows that these drifts are charge-dependent (opposite
directions for electrons and ions).

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, m_p


def boris_push(x, v, q, m, E, B, dt):
    """
    Boris algorithm for particle pushing.

    Parameters:
    -----------
    x, v : array (3,)
        Position [m] and velocity [m/s]
    q, m : float
        Charge [C] and mass [kg]
    E, B : array (3,)
        Electric [V/m] and magnetic [T] fields
    dt : float
        Timestep [s]

    Returns:
    --------
    x_new, v_new : Updated position and velocity
    """
    # Half acceleration
    v_minus = v + (q * E / m) * (dt / 2)

    # Rotation
    t = (q * dt / (2 * m)) * B
    t_mag2 = np.dot(t, t)
    s = 2 * t / (1 + t_mag2)

    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    # Half acceleration
    v_new = v_plus + (q * E / m) * (dt / 2)

    # Update position
    x_new = x + v_new * dt

    return x_new, v_new


def magnetic_field_gradient(x, B0, L):
    """
    Linearly increasing magnetic field in z direction.
    B = B0 * (1 + x/L) * ẑ

    Parameters:
    -----------
    x : array (3,)
        Position [m]
    B0 : float
        Reference field [T]
    L : float
        Gradient scale length [m]

    Returns:
    --------
    B : array (3,)
        Magnetic field [T]
    """
    B_magnitude = B0 * (1 + x[0] / L)
    return np.array([0, 0, B_magnitude])


def magnetic_field_dipole(x, B0, R0):
    """
    Simplified dipole-like field (2D approximation).

    In cylindrical coordinates centered on z-axis:
    B_r ≈ -B0 * (R0/r)³ * sin(θ)
    B_z ≈ B0 * (R0/r)³ * 2*cos(θ)

    Simplified for |z| << R:
    B_x ≈ -3 * B0 * R0² * x * z / r⁵
    B_y ≈ -3 * B0 * R0² * y * z / r⁵
    B_z ≈ B0 * R0² * (2*z² - x² - y²) / r⁵

    Parameters:
    -----------
    x : array (3,)
        Position [m]
    B0 : float
        Reference field at equator [T]
    R0 : float
        Reference radius [m]

    Returns:
    --------
    B : array (3,)
        Magnetic field [T]
    """
    r2 = x[0]**2 + x[1]**2 + x[2]**2
    r2 = max(r2, (0.01 * R0)**2)  # Avoid singularity
    r5 = r2**2.5

    R02 = R0**2

    Bx = -3 * B0 * R02 * x[0] * x[2] / r5
    By = -3 * B0 * R02 * x[1] * x[2] / r5
    Bz = B0 * R02 * (2*x[2]**2 - x[0]**2 - x[1]**2) / r5

    return np.array([Bx, By, Bz])


def simulate_grad_b_drift(q, m, B0, L, v0, x0, dt, n_steps):
    """Simulate particle in gradient-B field."""

    x = np.zeros((n_steps, 3))
    v = np.zeros((n_steps, 3))
    t = np.zeros(n_steps)
    B_field = np.zeros((n_steps, 3))

    x[0] = x0
    v[0] = v0

    E = np.array([0, 0, 0])  # No electric field

    for i in range(n_steps - 1):
        B = magnetic_field_gradient(x[i], B0, L)
        B_field[i] = B
        x[i+1], v[i+1] = boris_push(x[i], v[i], q, m, E, B, dt)
        t[i+1] = t[i] + dt

    B_field[-1] = magnetic_field_gradient(x[-1], B0, L)

    return {
        'x': x[:, 0], 'y': x[:, 1], 'z': x[:, 2],
        'vx': v[:, 0], 'vy': v[:, 1], 'vz': v[:, 2],
        't': t, 'B': B_field
    }


def simulate_dipole_drift(q, m, B0, R0, v0, x0, dt, n_steps):
    """Simulate particle in dipole-like field."""

    x = np.zeros((n_steps, 3))
    v = np.zeros((n_steps, 3))
    t = np.zeros(n_steps)

    x[0] = x0
    v[0] = v0

    E = np.array([0, 0, 0])

    for i in range(n_steps - 1):
        B = magnetic_field_dipole(x[i], B0, R0)
        x[i+1], v[i+1] = boris_push(x[i], v[i], q, m, E, B, dt)
        t[i+1] = t[i] + dt

    return {
        'x': x[:, 0], 'y': x[:, 1], 'z': x[:, 2],
        'vx': v[:, 0], 'vy': v[:, 1], 'vz': v[:, 2],
        't': t
    }


def plot_grad_b_drift():
    """Plot grad-B drift showing opposite drift for electron and ion."""

    # Parameters
    B0 = 1.0  # Tesla
    L = 0.1   # meters (gradient scale length)

    # Particles
    q_e = -e
    q_p = e
    m_e_kg = m_e
    m_p_kg = m_p

    # Initial conditions (perpendicular velocity)
    v_perp = 1e6  # m/s
    v0_e = np.array([0, v_perp, 0])
    v0_p = np.array([0, v_perp, 0])
    x0 = np.array([0, 0, 0])

    # Time parameters
    omega_ce = abs(q_e) * B0 / m_e_kg
    T_ce = 2 * np.pi / omega_ce
    dt = T_ce / 100
    n_steps = 2000

    # Simulate
    traj_e = simulate_grad_b_drift(q_e, m_e_kg, B0, L, v0_e, x0, dt, n_steps)
    traj_p = simulate_grad_b_drift(q_p, m_p_kg, B0, L, v0_p, x0, dt, n_steps)

    # Theoretical drift velocity (grad-B drift)
    # v_∇B = (m * v_perp²) / (2 * q * B²) * (B × ∇B) / B
    # For B = B0(1 + x/L)ẑ, ∇B = (B0/L)x̂
    # B × ∇B = B0²(1+x/L)/L * (ẑ × x̂) = B0²(1+x/L)/L * ŷ
    # At x=0: v_∇B ≈ (m * v_perp²) / (2 * q * B0 * L) * ŷ

    v_drift_e_theory = (m_e_kg * v_perp**2) / (2 * abs(q_e) * B0**2 * L)
    v_drift_p_theory = (m_p_kg * v_perp**2) / (2 * abs(q_p) * B0**2 * L)

    # Calculate drift from simulation
    t_start = int(0.3 * n_steps)
    drift_e_y = np.polyfit(traj_e['t'][t_start:], traj_e['y'][t_start:], 1)[0]
    drift_p_y = np.polyfit(traj_p['t'][t_start:], traj_p['y'][t_start:], 1)[0]

    print(f"\nGrad-B Drift:")
    print(f"  B = B0(1 + x/L)ẑ, B0 = {B0} T, L = {L} m")
    print(f"  v_perp = {v_perp/1e6:.1f} Mm/s")
    print(f"\nTheoretical drift velocities:")
    print(f"  Electron: {v_drift_e_theory:.2f} m/s (negative y)")
    print(f"  Proton: {v_drift_p_theory:.2f} m/s (positive y)")
    print(f"\nSimulated drift velocities:")
    print(f"  Electron: {drift_e_y:.2f} m/s")
    print(f"  Proton: {drift_p_y:.2f} m/s")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Trajectories in x-y plane
    ax1 = axes[0, 0]
    ax1.plot(traj_e['x'] * 1e3, traj_e['y'] * 1e3, 'b-',
             linewidth=1.5, alpha=0.7, label='Electron')
    ax1.plot(traj_p['x'] * 1e3, traj_p['y'] * 1e3, 'r-',
             linewidth=1.5, alpha=0.7, label='Proton')
    ax1.plot(0, 0, 'go', markersize=10, label='Start')

    # Show gradient direction
    ax1.arrow(20, -20, 10, 0, head_width=3, head_length=2,
              fc='purple', ec='purple', linewidth=2)
    ax1.text(35, -20, '∇B', fontsize=12, fontweight='bold', color='purple')

    ax1.set_xlabel('x [mm]', fontsize=12)
    ax1.set_ylabel('y [mm]', fontsize=12)
    ax1.set_title('Grad-B Drift Trajectories', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: y position vs time
    ax2 = axes[0, 1]
    ax2.plot(traj_e['t'] * 1e6, traj_e['y'] * 1e3, 'b-', linewidth=2, label='Electron (drifts down)')
    ax2.plot(traj_p['t'] * 1e6, traj_p['y'] * 1e3, 'r-', linewidth=2, label='Proton (drifts up)')

    ax2.set_xlabel('Time [μs]', fontsize=12)
    ax2.set_ylabel('y Position [mm]', fontsize=12)
    ax2.set_title('Opposite Drift Directions', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Magnetic field strength along trajectory
    ax3 = axes[1, 0]
    B_mag_e = np.linalg.norm(traj_e['B'], axis=1)
    ax3.plot(traj_e['x'] * 1e3, B_mag_e, 'b-', linewidth=2, label='|B| along trajectory')
    ax3.axhline(B0, color='gray', linestyle='--', linewidth=2, label=f'B0 = {B0} T')

    ax3.set_xlabel('x Position [mm]', fontsize=12)
    ax3.set_ylabel('|B| [T]', fontsize=12)
    ax3.set_title('Magnetic Field Strength', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Drift velocity comparison
    ax4 = axes[1, 1]

    # Calculate instantaneous drift by averaging over gyro-orbits
    window = 50
    from scipy.ndimage import uniform_filter1d
    smooth_y_e = uniform_filter1d(traj_e['y'], window)
    smooth_y_p = uniform_filter1d(traj_p['y'], window)
    v_drift_e = np.gradient(smooth_y_e, traj_e['t'])
    v_drift_p = np.gradient(smooth_y_p, traj_p['t'])

    ax4.plot(traj_e['t'] * 1e6, v_drift_e, 'b-', linewidth=2, alpha=0.7, label='Electron')
    ax4.plot(traj_p['t'] * 1e6, v_drift_p, 'r-', linewidth=2, alpha=0.7, label='Proton')
    ax4.axhline(-v_drift_e_theory, color='blue', linestyle='--',
                linewidth=2, label=f'Theory (e⁻): {-v_drift_e_theory:.2f} m/s')
    ax4.axhline(v_drift_p_theory, color='red', linestyle='--',
                linewidth=2, label=f'Theory (p⁺): {v_drift_p_theory:.2f} m/s')

    ax4.set_xlabel('Time [μs]', fontsize=12)
    ax4.set_ylabel('Drift Velocity v_y [m/s]', fontsize=12)
    ax4.set_title('Drift Velocity vs Theory', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('grad_b_drift.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_curvature_drift():
    """Plot curvature drift in dipole-like field."""

    # Parameters
    B0 = 1.0  # Tesla at equator
    R0 = 0.5  # Reference radius [m]

    # Particles
    q_e = -e
    q_p = e
    m_e_kg = m_e
    m_p_kg = m_p

    # Initial conditions (in x-z plane, with parallel velocity)
    v_para = 5e5  # m/s
    v_perp = 1e6  # m/s
    x0_e = np.array([R0, 0, 0])
    x0_p = np.array([R0, 0, 0])
    v0_e = np.array([0, v_perp, v_para])
    v0_p = np.array([0, v_perp, v_para])

    # Time parameters
    omega_ce = abs(q_e) * B0 / m_e_kg
    T_ce = 2 * np.pi / omega_ce
    dt = T_ce / 50
    n_steps = 1500

    # Simulate
    traj_e = simulate_dipole_drift(q_e, m_e_kg, B0, R0, v0_e, x0_e, dt, n_steps)
    traj_p = simulate_dipole_drift(q_p, m_p_kg, B0, R0, v0_p, x0_p, dt, n_steps)

    # Create figure
    fig = plt.figure(figsize=(14, 10))

    # 3D trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(traj_e['x'] * 1e2, traj_e['y'] * 1e2, traj_e['z'] * 1e2,
             'b-', linewidth=1.5, label='Electron')
    ax1.plot([traj_e['x'][0] * 1e2], [traj_e['y'][0] * 1e2], [traj_e['z'][0] * 1e2],
             'go', markersize=10, label='Start')

    ax1.set_xlabel('x [cm]', fontsize=11)
    ax1.set_ylabel('y [cm]', fontsize=11)
    ax1.set_zlabel('z [cm]', fontsize=11)
    ax1.set_title('3D Trajectory in Dipole Field', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)

    # x-y projection
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(traj_e['x'] * 1e2, traj_e['y'] * 1e2, 'b-',
             linewidth=1.5, alpha=0.7, label='Electron')
    ax2.plot(traj_p['x'] * 1e2, traj_p['y'] * 1e2, 'r-',
             linewidth=1.5, alpha=0.7, label='Proton')
    ax2.plot(0, 0, 'k*', markersize=15, label='Center')
    ax2.plot(traj_e['x'][0] * 1e2, traj_e['y'][0] * 1e2,
             'go', markersize=10, label='Start')

    ax2.set_xlabel('x [cm]', fontsize=12)
    ax2.set_ylabel('y [cm]', fontsize=12)
    ax2.set_title('x-y Projection (Curvature Drift)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # x-z projection
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(traj_e['x'] * 1e2, traj_e['z'] * 1e2, 'b-',
             linewidth=1.5, alpha=0.7, label='Electron')
    ax3.plot(traj_p['x'] * 1e2, traj_p['z'] * 1e2, 'r-',
             linewidth=1.5, alpha=0.7, label='Proton')
    ax3.plot(0, 0, 'k*', markersize=15, label='Center')

    ax3.set_xlabel('x [cm]', fontsize=12)
    ax3.set_ylabel('z [cm]', fontsize=12)
    ax3.set_title('x-z Projection', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # Radial distance vs time
    ax4 = fig.add_subplot(2, 2, 4)
    r_e = np.sqrt(traj_e['x']**2 + traj_e['y']**2 + traj_e['z']**2)
    r_p = np.sqrt(traj_p['x']**2 + traj_p['y']**2 + traj_p['z']**2)

    ax4.plot(traj_e['t'] * 1e6, r_e * 1e2, 'b-', linewidth=2, label='Electron')
    ax4.plot(traj_p['t'] * 1e6, r_p * 1e2, 'r-', linewidth=2, label='Proton')

    ax4.set_xlabel('Time [μs]', fontsize=12)
    ax4.set_ylabel('Radial Distance [cm]', fontsize=12)
    ax4.set_title('Distance from Center', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('curvature_drift.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("GRADIENT AND CURVATURE DRIFT SIMULATION")
    print("="*80)

    print("\nPart 1: Grad-B Drift")
    print("-" * 80)
    plot_grad_b_drift()

    print("\nPart 2: Curvature Drift in Dipole Field")
    print("-" * 80)
    plot_curvature_drift()

    print("\nKey observations:")
    print("  - Grad-B drift: Opposite directions for e⁻ and p⁺")
    print("  - E×B drift: Same direction for all species")
    print("  - Curvature drift: Related to parallel motion along curved field lines")

    print("\nDone! Generated 2 figures:")
    print("  - grad_b_drift.png")
    print("  - curvature_drift.png")
