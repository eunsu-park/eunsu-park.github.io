#!/usr/bin/env python3
"""
1D Vlasov-Poisson Solver

This script implements a 1D electrostatic Vlasov-Poisson solver using
operator splitting (Strang splitting) with cubic spline interpolation.
Demonstrates plasma oscillations.

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, epsilon_0, k
from scipy.interpolate import CubicSpline
from scipy.fft import fft, ifft, fftfreq


class VlasovPoisson1D:
    """
    1D-1V Vlasov-Poisson solver using operator splitting.

    ∂f/∂t + v∂f/∂x + (q/m)E∂f/∂v = 0
    ∂E/∂x = ρ/ε₀ = (q/ε₀)(∫f dv - n₀)
    """

    def __init__(self, Nx, Nv, Lx, v_max, n0, T, q=-e, m=m_e):
        """
        Initialize solver.

        Parameters:
        -----------
        Nx, Nv : int
            Grid points in x and v
        Lx : float
            System length [m]
        v_max : float
            Maximum velocity [m/s]
        n0 : float
            Background density [m^-3]
        T : float
            Temperature [K]
        q, m : float
            Charge and mass
        """
        self.Nx = Nx
        self.Nv = Nv
        self.Lx = Lx
        self.v_max = v_max
        self.n0 = n0
        self.T = T
        self.q = q
        self.m = m

        # Spatial grid
        self.x = np.linspace(0, Lx, Nx, endpoint=False)
        self.dx = Lx / Nx

        # Velocity grid
        self.v = np.linspace(-v_max, v_max, Nv)
        self.dv = 2 * v_max / Nv

        # Distribution function f(x, v)
        self.f = np.zeros((Nx, Nv))

        # Electric field
        self.E = np.zeros(Nx)

        # Wavenumbers for FFT
        self.kx = 2 * np.pi * fftfreq(Nx, Lx / Nx)

    def initialize_maxwellian(self, perturbation=0.0, k_pert=1):
        """
        Initialize with perturbed Maxwellian.

        f(x, v, t=0) = n₀(1 + α cos(kx)) * f_M(v)

        Parameters:
        -----------
        perturbation : float
            Amplitude of density perturbation
        k_pert : int
            Mode number of perturbation
        """
        v_th = np.sqrt(2 * k * self.T / self.m)

        # Maxwellian in velocity
        f_M = (self.n0 / (np.sqrt(2 * np.pi) * v_th) *
               np.exp(-self.v**2 / (2 * v_th**2)))

        # Spatial perturbation
        k = 2 * np.pi * k_pert / self.Lx
        n_pert = 1 + perturbation * np.cos(k * self.x)

        # f(x, v)
        for i in range(self.Nx):
            self.f[i, :] = n_pert[i] * f_M

    def compute_density(self):
        """Compute density n(x) = ∫ f(x,v) dv."""
        return np.trapz(self.f, self.v, axis=1)

    def compute_electric_field(self):
        """
        Solve Poisson equation for E field using FFT.

        ∂E/∂x = ρ/ε₀ = (q/ε₀)(n - n₀)
        """
        n = self.compute_density()
        rho = self.q * (n - self.n0)

        # FFT of charge density
        rho_k = fft(rho)

        # Solve in Fourier space: ik E_k = rho_k / ε₀
        # E_k = -i * rho_k / (k * ε₀)
        E_k = np.zeros_like(rho_k, dtype=complex)
        E_k[1:] = -1j * rho_k[1:] / (self.kx[1:] * epsilon_0)
        E_k[0] = 0  # Zero mode (charge neutrality)

        # Inverse FFT
        self.E = np.real(ifft(E_k))

    def advect_x(self, dt):
        """
        Advection in x: ∂f/∂t + v∂f/∂x = 0

        Use cubic spline interpolation for accuracy.
        """
        f_new = np.zeros_like(self.f)

        for j in range(self.Nv):
            # For each velocity, advect in x
            # x_new = x - v * dt (backwards)
            x_old = (self.x - self.v[j] * dt) % self.Lx  # Periodic BC

            # Interpolate
            cs = CubicSpline(self.x, self.f[:, j], bc_type='periodic')
            f_new[:, j] = cs(x_old)

        self.f = f_new

    def advect_v(self, dt):
        """
        Advection in v: ∂f/∂t + (q/m)E∂f/∂v = 0

        Use cubic spline interpolation.
        """
        f_new = np.zeros_like(self.f)
        a = self.q * self.E / self.m  # Acceleration

        for i in range(self.Nx):
            # For each position, advect in v
            # v_new = v - a * dt (backwards)
            v_old = self.v - a[i] * dt

            # Handle boundaries: extrapolate or zero
            mask = (v_old >= -self.v_max) & (v_old <= self.v_max)

            if np.any(mask):
                cs = CubicSpline(self.v, self.f[i, :], bc_type='natural')
                f_new[i, mask] = cs(v_old[mask])
                f_new[i, mask] = np.maximum(f_new[i, mask], 0)  # Non-negative

        self.f = f_new

    def step_strang(self, dt):
        """
        Strang splitting: A(dt/2) B(dt) A(dt/2)

        A = x-advection, B = v-advection
        """
        # Half step in x
        self.advect_x(dt / 2)

        # Full step in v (requires E field)
        self.compute_electric_field()
        self.advect_v(dt)

        # Half step in x
        self.advect_x(dt / 2)

    def run(self, t_end, dt):
        """
        Run simulation.

        Parameters:
        -----------
        t_end : float
            End time [s]
        dt : float
            Timestep [s]

        Returns:
        --------
        history : dict
            Time history of fields
        """
        n_steps = int(t_end / dt)

        # Storage
        t_history = np.zeros(n_steps + 1)
        E_history = np.zeros((n_steps + 1, self.Nx))
        n_history = np.zeros((n_steps + 1, self.Nx))
        f_history = []

        # Initial state
        self.compute_electric_field()
        t_history[0] = 0
        E_history[0, :] = self.E
        n_history[0, :] = self.compute_density()
        f_history.append(self.f.copy())

        # Time loop
        for n in range(n_steps):
            self.step_strang(dt)

            t_history[n + 1] = (n + 1) * dt
            E_history[n + 1, :] = self.E
            n_history[n + 1, :] = self.compute_density()

            # Store f occasionally
            if n % max(1, n_steps // 20) == 0:
                f_history.append(self.f.copy())

        return {
            't': t_history,
            'E': E_history,
            'n': n_history,
            'f_snapshots': f_history,
            'x': self.x,
            'v': self.v
        }


def simulate_plasma_oscillation():
    """Simulate plasma oscillation and verify frequency."""

    # Parameters
    n0 = 1e16  # m^-3
    T = 1e4    # K (~ 1 eV)
    v_th = np.sqrt(2 * k * T / m_e)

    # Plasma frequency
    omega_pe = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))
    f_pe = omega_pe / (2 * np.pi)
    T_pe = 1 / f_pe

    print(f"\nPlasma Parameters:")
    print(f"  Density: {n0:.2e} m^-3")
    print(f"  Temperature: {T:.2e} K ({T * k / e:.2f} eV)")
    print(f"  Thermal velocity: {v_th:.2e} m/s")
    print(f"  Plasma frequency: {f_pe:.2e} Hz")
    print(f"  Plasma period: {T_pe:.2e} s")

    # Debye length
    lambda_D = np.sqrt(epsilon_0 * k * T / (n0 * e**2))
    print(f"  Debye length: {lambda_D:.2e} m")

    # Grid
    Nx = 64
    Nv = 128
    Lx = 10 * lambda_D  # System size
    v_max = 6 * v_th

    # Initialize solver
    solver = VlasovPoisson1D(Nx, Nv, Lx, v_max, n0, T)
    solver.initialize_maxwellian(perturbation=0.01, k_pert=1)

    # Run simulation
    t_end = 10 * T_pe
    dt = T_pe / 50

    print(f"\nSimulation:")
    print(f"  Grid: {Nx} × {Nv}")
    print(f"  System size: {Lx:.2e} m ({Lx/lambda_D:.1f} λ_D)")
    print(f"  Time: {t_end:.2e} s ({t_end/T_pe:.1f} T_pe)")
    print(f"  Timestep: {dt:.2e} s ({dt/T_pe:.3f} T_pe)")

    history = solver.run(t_end, dt)

    return history, omega_pe, lambda_D


def plot_results(history, omega_pe, lambda_D):
    """Plot simulation results."""

    t = history['t']
    E = history['E']
    n = history['n']
    x = history['x']
    v = history['v']
    f_snapshots = history['f_snapshots']

    T_pe = 2 * np.pi / omega_pe

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Electric field vs time (at one location)
    ax1 = plt.subplot(3, 3, 1)
    i_mid = len(x) // 2
    ax1.plot(t / T_pe, E[:, i_mid], 'b-', linewidth=2)
    ax1.set_xlabel('Time [T_pe]', fontsize=11)
    ax1.set_ylabel('E(x=L/2) [V/m]', fontsize=11)
    ax1.set_title('Electric Field Oscillation', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: E field energy vs time
    ax2 = plt.subplot(3, 3, 2)
    E_energy = np.sum(E**2, axis=1) * (x[1] - x[0])
    ax2.semilogy(t / T_pe, E_energy, 'r-', linewidth=2)
    ax2.set_xlabel('Time [T_pe]', fontsize=11)
    ax2.set_ylabel('∫E² dx [a.u.]', fontsize=11)
    ax2.set_title('Electric Field Energy', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Density perturbation
    ax3 = plt.subplot(3, 3, 3)
    n0 = np.mean(n[0, :])
    dn = (n - n0) / n0
    cs = ax3.contourf(x / lambda_D, t / T_pe, dn, levels=20, cmap='RdBu_r')
    ax3.set_xlabel('x [λ_D]', fontsize=11)
    ax3.set_ylabel('Time [T_pe]', fontsize=11)
    ax3.set_title('Density Perturbation δn/n₀', fontsize=12, fontweight='bold')
    plt.colorbar(cs, ax=ax3)

    # Plot 4-9: Phase space snapshots
    snapshot_indices = [0, len(f_snapshots)//4, len(f_snapshots)//2,
                       3*len(f_snapshots)//4, len(f_snapshots)-1]

    for idx, snap_idx in enumerate(snapshot_indices[:6]):
        ax = plt.subplot(3, 3, 4 + idx)
        f = f_snapshots[snap_idx]

        # Time of snapshot
        t_snap = snap_idx * (t[-1] / (len(f_snapshots) - 1))

        # Plot phase space
        v_th = np.sqrt(2 * k * 1e4 / m_e)
        cs = ax.contourf(x / lambda_D, v / v_th, f.T, levels=20, cmap='hot')
        ax.set_xlabel('x [λ_D]', fontsize=10)
        ax.set_ylabel('v [v_th]', fontsize=10)
        ax.set_title(f't = {t_snap/T_pe:.2f} T_pe', fontsize=11, fontweight='bold')
        ax.set_ylim([-3, 3])

    plt.tight_layout()
    plt.savefig('vlasov_plasma_oscillation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # FFT analysis
    fig2, ax = plt.subplots(figsize=(10, 6))

    # FFT of E field at midpoint
    E_mid = E[:, i_mid]
    dt = t[1] - t[0]
    freqs = fftfreq(len(t), dt)
    E_fft = np.abs(fft(E_mid))

    # Only positive frequencies
    pos_freqs = freqs > 0
    ax.semilogy(freqs[pos_freqs] / (omega_pe / (2*np.pi)), E_fft[pos_freqs],
                'b-', linewidth=2)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2,
               label=f'ω_pe = {omega_pe/(2*np.pi):.2e} Hz')

    ax.set_xlabel('Frequency [f_pe]', fontsize=12)
    ax.set_ylabel('|FFT(E)| [a.u.]', fontsize=12)
    ax.set_title('Frequency Spectrum of E Field', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 3])

    plt.tight_layout()
    plt.savefig('vlasov_frequency_spectrum.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("1D VLASOV-POISSON SOLVER: PLASMA OSCILLATIONS")
    print("="*80)

    history, omega_pe, lambda_D = simulate_plasma_oscillation()

    print("\nGenerating plots...")
    plot_results(history, omega_pe, lambda_D)

    print("\nDone! Generated 2 figures:")
    print("  - vlasov_plasma_oscillation.png")
    print("  - vlasov_frequency_spectrum.png")
