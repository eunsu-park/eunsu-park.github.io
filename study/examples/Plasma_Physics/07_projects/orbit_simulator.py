#!/usr/bin/env python3
"""
Comprehensive Particle Orbit Simulator

This script simulates charged particle orbits in various electromagnetic field
configurations using the Boris algorithm.

Field configurations:
1. Uniform B: gyration
2. Uniform E + B: E×B drift
3. Gradient B: grad-B drift
4. Curved B: curvature drift
5. Mirror B: bounce motion
6. Tokamak-like: banana orbit

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
QE = 1.602176634e-19    # C
ME = 9.10938356e-31     # kg
MP = 1.672621898e-27    # kg

class ParticleOrbitSimulator:
    """Particle orbit simulator using Boris algorithm."""

    def __init__(self, q, m, dt=1e-10):
        """
        Initialize simulator.

        Parameters:
        -----------
        q : float
            Particle charge [C]
        m : float
            Particle mass [kg]
        dt : float
            Time step [s]
        """
        self.q = q
        self.m = m
        self.dt = dt

    def boris_push(self, r, v, E, B):
        """
        Boris algorithm for particle pushing.

        Parameters:
        -----------
        r : array (3,)
            Position [m]
        v : array (3,)
            Velocity [m/s]
        E : array (3,)
            Electric field [V/m]
        B : array (3,)
            Magnetic field [T]

        Returns:
        --------
        r_new, v_new : updated position and velocity
        """
        # Half acceleration from E field
        v_minus = v + 0.5 * (self.q / self.m) * E * self.dt

        # Rotation from B field
        t = 0.5 * (self.q / self.m) * B * self.dt
        s = 2 * t / (1 + np.dot(t, t))

        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)

        # Half acceleration from E field
        v_new = v_plus + 0.5 * (self.q / self.m) * E * self.dt

        # Update position
        r_new = r + v_new * self.dt

        return r_new, v_new

    def simulate(self, r0, v0, E_func, B_func, t_max, save_every=1):
        """
        Run simulation.

        Parameters:
        -----------
        r0 : array (3,)
            Initial position [m]
        v0 : array (3,)
            Initial velocity [m/s]
        E_func : callable
            E_func(r) returns E field at position r
        B_func : callable
            B_func(r) returns B field at position r
        t_max : float
            Maximum simulation time [s]
        save_every : int
            Save every N steps

        Returns:
        --------
        t, r, v : time and trajectory arrays
        """
        n_steps = int(t_max / self.dt)

        # Storage
        r_traj = [r0]
        v_traj = [v0]
        t_traj = [0]

        r = r0.copy()
        v = v0.copy()

        for step in range(1, n_steps + 1):
            E = E_func(r)
            B = B_func(r)

            r, v = self.boris_push(r, v, E, B)

            if step % save_every == 0:
                r_traj.append(r.copy())
                v_traj.append(v.copy())
                t_traj.append(step * self.dt)

        return np.array(t_traj), np.array(r_traj), np.array(v_traj)

# Field configuration functions

def uniform_B_field(B0):
    """Uniform magnetic field in z direction."""
    def B_func(r):
        return np.array([0, 0, B0])
    return B_func

def uniform_E_and_B(E0, B0):
    """Uniform E (x-dir) and B (z-dir) for E×B drift."""
    def E_func(r):
        return np.array([E0, 0, 0])
    def B_func(r):
        return np.array([0, 0, B0])
    return E_func, B_func

def gradient_B_field(B0, L):
    """Gradient B field: B = B0(1 + x/L) ẑ."""
    def B_func(r):
        return np.array([0, 0, B0 * (1 + r[0] / L)])
    return B_func

def curved_B_field(B0, R_c):
    """Curved B field (toroidal-like)."""
    def B_func(r):
        x, y, z = r
        R = np.sqrt(x**2 + y**2)
        if R < 1e-10:
            return np.array([0, 0, B0])
        # Toroidal field: B_phi = B0 * R0/R
        B_mag = B0 * R_c / (R + R_c)
        B_x = -B_mag * y / R
        B_y = B_mag * x / R
        return np.array([B_x, B_y, 0])
    return B_func

def mirror_B_field(B0, L_mirror):
    """Mirror field: B = B0(1 + (z/L)²) ẑ."""
    def B_func(r):
        return np.array([0, 0, B0 * (1 + (r[2] / L_mirror)**2)])
    return B_func

def tokamak_field(B_tor, B_pol, R0, r_minor):
    """
    Simplified tokamak field (toroidal + poloidal).

    Parameters:
    -----------
    B_tor : float
        Toroidal field at major radius [T]
    B_pol : float
        Poloidal field strength [T]
    R0 : float
        Major radius [m]
    r_minor : float
        Minor radius [m]
    """
    def B_func(r):
        x, y, z = r
        R = np.sqrt(x**2 + y**2)

        if R < 1e-10:
            return np.array([0, 0, B_tor])

        # Toroidal field (1/R dependence)
        B_t = B_tor * R0 / R
        B_phi_x = -B_t * y / R
        B_phi_y = B_t * x / R

        # Poloidal field (simplified: radial gradient)
        theta = np.arctan2(z, R - R0)
        B_p_r = B_pol * np.sin(theta)
        B_p_z = B_pol * np.cos(theta)

        B_x = B_phi_x + B_p_r * (R - R0) / R if R > 0 else 0
        B_y = B_phi_y
        B_z = B_p_z

        return np.array([B_x, B_y, B_z])

    return B_func

def plot_orbit_comparison():
    """
    Compare particle orbits in different field configurations.
    """
    print("=" * 70)
    print("Particle Orbit Simulator: Field Configuration Comparison")
    print("=" * 70)

    # Particle parameters
    q = QE  # Proton
    m = MP
    E_kin = 100  # eV
    v0_mag = np.sqrt(2 * E_kin * QE / m)

    # Field parameters
    B0 = 1.0  # T
    E0 = 1000  # V/m

    # Create simulator
    dt = 1e-9  # 1 ns
    sim = ParticleOrbitSimulator(q, m, dt)

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

    configurations = []

    # Configuration 1: Uniform B (gyration)
    print("\n1. Uniform B field (gyration)...")
    r0 = np.array([0, 0, 0])
    v0 = np.array([v0_mag, 0, 0])
    B_func = uniform_B_field(B0)
    E_func = lambda r: np.array([0, 0, 0])

    t, r, v = sim.simulate(r0, v0, E_func, B_func, t_max=1e-7, save_every=10)

    # Compute theoretical Larmor radius
    rho_L = m * v0_mag / (q * B0)
    omega_c = q * B0 / m

    configurations.append({
        'name': '1. Uniform B: Gyration',
        'r': r,
        't': t,
        'theory': f'ρL = {rho_L*1e3:.2f} mm, fc = {omega_c/(2*np.pi)/1e6:.1f} MHz',
        'drift': np.array([0, 0, 0])
    })

    # Configuration 2: E×B drift
    print("2. E×B drift...")
    r0 = np.array([0, 0, 0])
    v0 = np.array([v0_mag, 0, 0])
    E_func, B_func = uniform_E_and_B(E0, B0)

    t, r, v = sim.simulate(r0, v0, E_func, B_func, t_max=1e-7, save_every=10)

    v_ExB = E0 / B0

    configurations.append({
        'name': '2. E×B Drift',
        'r': r,
        't': t,
        'theory': f'vE×B = {v_ExB/1e3:.1f} km/s',
        'drift': np.array([0, v_ExB, 0])
    })

    # Configuration 3: Grad-B drift
    print("3. Grad-B drift...")
    L_grad = 1.0  # m
    r0 = np.array([0, 0, 0])
    v0 = np.array([v0_mag, 0, 0])
    B_func = gradient_B_field(B0, L_grad)
    E_func = lambda r: np.array([0, 0, 0])

    t, r, v = sim.simulate(r0, v0, E_func, B_func, t_max=5e-7, save_every=10)

    # Theoretical grad-B drift (perpendicular energy)
    v_perp = v0_mag
    v_gradB = -m * v_perp**2 / (2 * q * B0**2 * L_grad)

    configurations.append({
        'name': '3. Grad-B Drift',
        'r': r,
        't': t,
        'theory': f'v∇B ≈ {v_gradB:.1f} m/s',
        'drift': np.array([0, v_gradB, 0])
    })

    # Configuration 4: Curved B (curvature drift)
    print("4. Curvature drift...")
    R_c = 1.0  # m
    r0 = np.array([R_c, 0, 0])
    v0 = np.array([0, 0, v0_mag])  # Parallel velocity
    B_func = curved_B_field(B0, R_c)
    E_func = lambda r: np.array([0, 0, 0])

    t, r, v = sim.simulate(r0, v0, E_func, B_func, t_max=1e-6, save_every=10)

    # Theoretical curvature drift
    v_parallel = v0_mag
    v_curv = m * v_parallel**2 / (q * B0 * R_c**2)

    configurations.append({
        'name': '4. Curvature Drift',
        'r': r,
        't': t,
        'theory': f'vcurv ≈ {v_curv:.1f} m/s',
        'drift': None
    })

    # Configuration 5: Mirror field (bounce motion)
    print("5. Mirror field (bounce)...")
    L_mirror = 0.5  # m
    r0 = np.array([0.01, 0, 0])  # Small offset
    v_parallel = 0.5 * v0_mag
    v_perp = np.sqrt(v0_mag**2 - v_parallel**2)
    v0 = np.array([v_perp, 0, v_parallel])
    B_func = mirror_B_field(B0, L_mirror)
    E_func = lambda r: np.array([0, 0, 0])

    t, r, v = sim.simulate(r0, v0, E_func, B_func, t_max=5e-7, save_every=10)

    configurations.append({
        'name': '5. Mirror Field: Bounce',
        'r': r,
        't': t,
        'theory': f'Mirror ratio = 2.0',
        'drift': None
    })

    # Configuration 6: Tokamak (banana orbit)
    print("6. Tokamak field (banana orbit)...")
    R0 = 1.0  # m
    r_minor = 0.3  # m
    B_tor = 2.0  # T
    B_pol = 0.2  # T

    r0 = np.array([R0 + 0.1, 0, 0])
    v0 = np.array([v0_mag * 0.3, v0_mag * 0.3, v0_mag * 0.9])
    B_func = tokamak_field(B_tor, B_pol, R0, r_minor)
    E_func = lambda r: np.array([0, 0, 0])

    t, r, v = sim.simulate(r0, v0, E_func, B_func, t_max=2e-6, save_every=20)

    configurations.append({
        'name': '6. Tokamak: Banana Orbit',
        'r': r,
        't': t,
        'theory': f'R0={R0:.1f}m, a={r_minor:.1f}m',
        'drift': None
    })

    # Plot all configurations
    for idx, config in enumerate(configurations):
        row = idx // 3
        col = idx % 3

        if idx < 5:
            ax = fig.add_subplot(gs[row, col], projection='3d')
        else:
            ax = fig.add_subplot(gs[row, col])

        r = config['r']

        if idx < 5:
            # 3D plot
            ax.plot(r[:, 0] * 1e3, r[:, 1] * 1e3, r[:, 2] * 1e3,
                   'b-', linewidth=1, alpha=0.7)
            ax.scatter(r[0, 0] * 1e3, r[0, 1] * 1e3, r[0, 2] * 1e3,
                      c='green', s=50, marker='o', label='Start')
            ax.scatter(r[-1, 0] * 1e3, r[-1, 1] * 1e3, r[-1, 2] * 1e3,
                      c='red', s=50, marker='x', label='End')

            ax.set_xlabel('x (mm)', fontsize=9)
            ax.set_ylabel('y (mm)', fontsize=9)
            ax.set_zlabel('z (mm)', fontsize=9)
            ax.set_title(config['name'] + '\n' + config['theory'],
                        fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
        else:
            # 2D projection for tokamak
            R = np.sqrt(r[:, 0]**2 + r[:, 1]**2)
            Z = r[:, 2]

            ax.plot(R, Z, 'b-', linewidth=1, alpha=0.7)
            ax.scatter(R[0], Z[0], c='green', s=50, marker='o', label='Start')
            ax.scatter(R[-1], Z[-1], c='red', s=50, marker='x', label='End')

            # Draw tokamak boundary
            theta = np.linspace(0, 2 * np.pi, 100)
            R_boundary = R0 + r_minor * np.cos(theta)
            Z_boundary = r_minor * np.sin(theta)
            ax.plot(R_boundary, Z_boundary, 'k--', linewidth=1, label='Boundary')

            ax.set_xlabel('R (m)', fontsize=9)
            ax.set_ylabel('Z (m)', fontsize=9)
            ax.set_title(config['name'] + '\n' + config['theory'],
                        fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

    plt.suptitle('Particle Orbit Simulator: Drift and Bounce Motions',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('orbit_simulator.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'orbit_simulator.png'")
    print("=" * 70)

    plt.show()

if __name__ == "__main__":
    plot_orbit_comparison()
