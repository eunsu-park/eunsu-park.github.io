#!/usr/bin/env python3
"""
Landau Damping Simulation

This script demonstrates Landau damping using the 1D Vlasov-Poisson solver.
Shows exponential damping of electric field energy and phase space filamentation.
Also demonstrates inverse Landau damping (bump-on-tail instability).

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, epsilon_0, k
from scipy.interpolate import CubicSpline
from scipy.fft import fft, ifft, fftfreq
from scipy.special import erf


class VlasovPoisson1D:
    """1D-1V Vlasov-Poisson solver (same as vlasov_1d.py)."""

    def __init__(self, Nx, Nv, Lx, v_max, n0, T, q=-e, m=m_e):
        self.Nx = Nx
        self.Nv = Nv
        self.Lx = Lx
        self.v_max = v_max
        self.n0 = n0
        self.T = T
        self.q = q
        self.m = m

        self.x = np.linspace(0, Lx, Nx, endpoint=False)
        self.dx = Lx / Nx
        self.v = np.linspace(-v_max, v_max, Nv)
        self.dv = 2 * v_max / Nv

        self.f = np.zeros((Nx, Nv))
        self.E = np.zeros(Nx)
        self.kx = 2 * np.pi * fftfreq(Nx, Lx / Nx)

    def initialize_maxwellian(self, perturbation=0.0, k_pert=1):
        v_th = np.sqrt(2 * k * self.T / self.m)
        f_M = (self.n0 / (np.sqrt(2 * np.pi) * v_th) *
               np.exp(-self.v**2 / (2 * v_th**2)))
        k_wave = 2 * np.pi * k_pert / self.Lx
        n_pert = 1 + perturbation * np.cos(k_wave * self.x)
        for i in range(self.Nx):
            self.f[i, :] = n_pert[i] * f_M

    def initialize_bump_on_tail(self, n_beam_frac=0.1, v_beam_factor=3.0,
                                T_beam_factor=0.5, perturbation=0.01, k_pert=1):
        """Initialize with bump-on-tail distribution."""
        v_th = np.sqrt(2 * k * self.T / self.m)
        v_beam = v_beam_factor * v_th
        T_beam = T_beam_factor * self.T
        v_th_beam = np.sqrt(2 * k * T_beam / self.m)

        n_bg = self.n0 * (1 - n_beam_frac)
        n_beam = self.n0 * n_beam_frac

        # Background Maxwellian
        f_bg = (n_bg / (np.sqrt(2 * np.pi) * v_th) *
                np.exp(-self.v**2 / (2 * v_th**2)))

        # Beam
        f_beam = (n_beam / (np.sqrt(2 * np.pi) * v_th_beam) *
                  np.exp(-(self.v - v_beam)**2 / (2 * v_th_beam**2)))

        # Total
        f_total = f_bg + f_beam

        # Spatial perturbation
        k_wave = 2 * np.pi * k_pert / self.Lx
        n_pert = 1 + perturbation * np.cos(k_wave * self.x)

        for i in range(self.Nx):
            self.f[i, :] = n_pert[i] * f_total

    def compute_density(self):
        return np.trapz(self.f, self.v, axis=1)

    def compute_electric_field(self):
        n = self.compute_density()
        rho = self.q * (n - self.n0)
        rho_k = fft(rho)
        E_k = np.zeros_like(rho_k, dtype=complex)
        E_k[1:] = -1j * rho_k[1:] / (self.kx[1:] * epsilon_0)
        E_k[0] = 0
        self.E = np.real(ifft(E_k))

    def advect_x(self, dt):
        f_new = np.zeros_like(self.f)
        for j in range(self.Nv):
            x_old = (self.x - self.v[j] * dt) % self.Lx
            cs = CubicSpline(self.x, self.f[:, j], bc_type='periodic')
            f_new[:, j] = cs(x_old)
        self.f = f_new

    def advect_v(self, dt):
        f_new = np.zeros_like(self.f)
        a = self.q * self.E / self.m
        for i in range(self.Nx):
            v_old = self.v - a[i] * dt
            mask = (v_old >= -self.v_max) & (v_old <= self.v_max)
            if np.any(mask):
                cs = CubicSpline(self.v, self.f[i, :], bc_type='natural')
                f_new[i, mask] = cs(v_old[mask])
                f_new[i, mask] = np.maximum(f_new[i, mask], 0)
        self.f = f_new

    def step_strang(self, dt):
        self.advect_x(dt / 2)
        self.compute_electric_field()
        self.advect_v(dt)
        self.advect_x(dt / 2)

    def run(self, t_end, dt, save_interval=10):
        n_steps = int(t_end / dt)
        t_history = np.zeros(n_steps + 1)
        E_history = np.zeros((n_steps + 1, self.Nx))
        f_history = []

        self.compute_electric_field()
        t_history[0] = 0
        E_history[0, :] = self.E
        f_history.append(self.f.copy())

        for n in range(n_steps):
            self.step_strang(dt)
            t_history[n + 1] = (n + 1) * dt
            E_history[n + 1, :] = self.E

            if n % save_interval == 0:
                f_history.append(self.f.copy())

        return {
            't': t_history,
            'E': E_history,
            'f_snapshots': f_history,
            'x': self.x,
            'v': self.v
        }


def landau_damping_rate_theory(k, v_th):
    """
    Theoretical Landau damping rate (small k limit).

    γ_L ≈ -sqrt(π/8) * (ω_pe/k)³ * (1/v_th³) * exp(-1/(2k²λ_D²) - 3/2)

    For k*λ_D << 1 (long wavelength).

    Parameters:
    -----------
    k : float
        Wavenumber [m^-1]
    v_th : float
        Thermal velocity [m/s]

    Returns:
    --------
    gamma : float
        Damping rate [s^-1]
    """
    # Using simplified formula for k*lambda_D ~ 0.5
    # γ ≈ -sqrt(π/8) * ω_pe * exp(-1/(2k²λ_D²) - 3/2)

    # For the standard case, use empirical fit
    # This is approximate; exact value requires solving dispersion relation
    omega_pe = np.sqrt(1e16 * e**2 / (epsilon_0 * m_e))  # Nominal
    k_lambda_D = k * v_th / omega_pe  # Approximation

    gamma = -np.sqrt(np.pi / 8) * omega_pe * np.exp(-1 / (2 * k_lambda_D**2) - 1.5)

    return gamma


def simulate_landau_damping():
    """Simulate Landau damping."""

    # Parameters
    n0 = 1e16  # m^-3
    T = 1e4    # K
    v_th = np.sqrt(2 * k * T / m_e)

    omega_pe = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))
    T_pe = 2 * np.pi / omega_pe
    lambda_D = np.sqrt(epsilon_0 * k * T / (n0 * e**2))

    print(f"\nLandau Damping Simulation:")
    print(f"  Plasma frequency: {omega_pe/(2*np.pi):.2e} Hz")
    print(f"  Debye length: {lambda_D:.2e} m")
    print(f"  Thermal velocity: {v_th:.2e} m/s")

    # Grid
    Nx = 64
    Nv = 256  # Higher resolution for damping
    Lx = 10 * lambda_D
    v_max = 6 * v_th

    k_pert = 1
    k_wave = 2 * np.pi * k_pert / Lx

    # Theoretical damping rate
    gamma_theory = landau_damping_rate_theory(k_wave, v_th)
    print(f"  Wavenumber k: {k_wave:.2e} m^-1")
    print(f"  k*λ_D: {k_wave * lambda_D:.3f}")
    print(f"  Theoretical damping rate: {gamma_theory:.2e} s^-1")
    print(f"  Damping time: {-1/gamma_theory:.2e} s ({-1/(gamma_theory*T_pe):.2f} T_pe)")

    # Initialize solver
    solver = VlasovPoisson1D(Nx, Nv, Lx, v_max, n0, T)
    solver.initialize_maxwellian(perturbation=0.1, k_pert=k_pert)

    # Run
    t_end = 50 * T_pe
    dt = T_pe / 50

    print(f"  Running for {t_end/T_pe:.1f} T_pe...")

    history = solver.run(t_end, dt, save_interval=max(1, int(t_end/dt) // 50))

    return history, omega_pe, lambda_D, gamma_theory


def simulate_bump_on_tail():
    """Simulate bump-on-tail instability (inverse Landau damping)."""

    # Parameters
    n0 = 1e16
    T = 1e4
    v_th = np.sqrt(2 * k * T / m_e)

    omega_pe = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))
    T_pe = 2 * np.pi / omega_pe
    lambda_D = np.sqrt(epsilon_0 * k * T / (n0 * e**2))

    print(f"\nBump-on-Tail Simulation:")
    print(f"  Beam fraction: 10%")
    print(f"  Beam velocity: 3 v_th")

    # Grid
    Nx = 64
    Nv = 256
    Lx = 10 * lambda_D
    v_max = 8 * v_th  # Wider for beam

    # Initialize with bump-on-tail
    solver = VlasovPoisson1D(Nx, Nv, Lx, v_max, n0, T)
    solver.initialize_bump_on_tail(n_beam_frac=0.1, v_beam_factor=3.0,
                                   T_beam_factor=0.5, perturbation=0.01, k_pert=1)

    # Run
    t_end = 100 * T_pe
    dt = T_pe / 50

    print(f"  Running for {t_end/T_pe:.1f} T_pe...")

    history = solver.run(t_end, dt, save_interval=max(1, int(t_end/dt) // 50))

    return history, omega_pe, lambda_D


def plot_landau_damping(history, omega_pe, lambda_D, gamma_theory):
    """Plot Landau damping results."""

    t = history['t']
    E = history['E']
    x = history['x']
    v = history['v']
    f_snapshots = history['f_snapshots']

    T_pe = 2 * np.pi / omega_pe
    v_th = np.sqrt(2 * k * 1e4 / m_e)

    # Electric field energy
    E_energy = np.sum(E**2, axis=1)

    # Fit exponential decay
    log_E = np.log(E_energy + 1e-20)
    t_fit = t[t < 30 * T_pe]
    log_E_fit = log_E[t < 30 * T_pe]

    # Linear fit to log(E)
    p = np.polyfit(t_fit, log_E_fit, 1)
    gamma_measured = p[0]

    print(f"\nMeasured damping rate: {gamma_measured:.2e} s^-1")
    print(f"Theory: {gamma_theory:.2e} s^-1")
    print(f"Error: {abs(gamma_measured - gamma_theory) / abs(gamma_theory) * 100:.1f}%")

    # Create figure
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: E field energy (log scale)
    ax1 = plt.subplot(2, 3, 1)
    ax1.semilogy(t / T_pe, E_energy, 'b-', linewidth=2, label='Simulation')
    ax1.semilogy(t_fit / T_pe, np.exp(p[0] * t_fit + p[1]), 'r--', linewidth=2,
                 label=f'Fit: γ = {gamma_measured:.2e} s⁻¹')
    ax1.semilogy(t / T_pe, E_energy[0] * np.exp(2 * gamma_theory * t), 'g:',
                 linewidth=2, label=f'Theory: γ = {gamma_theory:.2e} s⁻¹')

    ax1.set_xlabel('Time [T_pe]', fontsize=11)
    ax1.set_ylabel('∫E² dx [a.u.]', fontsize=11)
    ax1.set_title('Electric Field Energy Decay', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2-7: Phase space evolution
    snapshot_times = [0, 10, 20, 30, 40, 50]

    for idx, t_snap_idx in enumerate(range(min(6, len(f_snapshots)))):
        if t_snap_idx >= len(f_snapshots):
            break

        ax = plt.subplot(2, 3, 2 + idx)
        f = f_snapshots[t_snap_idx]
        t_actual = t_snap_idx * (t[-1] / (len(f_snapshots) - 1))

        cs = ax.contourf(x / lambda_D, v / v_th, f.T, levels=30, cmap='hot')
        ax.set_xlabel('x [λ_D]', fontsize=10)
        ax.set_ylabel('v [v_th]', fontsize=10)
        ax.set_title(f't = {t_actual/T_pe:.1f} T_pe', fontsize=11, fontweight='bold')
        ax.set_ylim([-4, 4])

    plt.tight_layout()
    plt.savefig('landau_damping.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_bump_on_tail(history, omega_pe, lambda_D):
    """Plot bump-on-tail (growth) results."""

    t = history['t']
    E = history['E']
    x = history['x']
    v = history['v']
    f_snapshots = history['f_snapshots']

    T_pe = 2 * np.pi / omega_pe
    v_th = np.sqrt(2 * k * 1e4 / m_e)

    # Electric field energy
    E_energy = np.sum(E**2, axis=1)

    # Fit exponential growth (linear phase)
    t_linear = (t > 10 * T_pe) & (t < 40 * T_pe)
    log_E = np.log(E_energy[t_linear] + 1e-20)
    p = np.polyfit(t[t_linear], log_E, 1)
    gamma_growth = p[0]

    print(f"\nMeasured growth rate: {gamma_growth:.2e} s^-1")

    # Create figure
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: E field energy (log scale, showing growth)
    ax1 = plt.subplot(2, 3, 1)
    ax1.semilogy(t / T_pe, E_energy, 'b-', linewidth=2, label='Simulation')
    ax1.semilogy(t[t_linear] / T_pe, np.exp(p[0] * t[t_linear] + p[1]), 'r--',
                 linewidth=2, label=f'Growth: γ = {gamma_growth:.2e} s⁻¹')

    ax1.set_xlabel('Time [T_pe]', fontsize=11)
    ax1.set_ylabel('∫E² dx [a.u.]', fontsize=11)
    ax1.set_title('Electric Field Energy Growth (Instability)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2-7: Phase space showing plateau formation
    for idx in range(min(6, len(f_snapshots))):
        ax = plt.subplot(2, 3, 2 + idx)
        f = f_snapshots[idx]
        t_actual = idx * (t[-1] / (len(f_snapshots) - 1))

        cs = ax.contourf(x / lambda_D, v / v_th, f.T, levels=30, cmap='hot')
        ax.set_xlabel('x [λ_D]', fontsize=10)
        ax.set_ylabel('v [v_th]', fontsize=10)
        ax.set_title(f't = {t_actual/T_pe:.1f} T_pe', fontsize=11, fontweight='bold')
        ax.set_ylim([-5, 6])

    plt.tight_layout()
    plt.savefig('bump_on_tail_growth.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("LANDAU DAMPING AND BUMP-ON-TAIL INSTABILITY")
    print("="*80)

    print("\nPart 1: Landau Damping (Collisionless Damping)")
    print("-" * 80)
    history_ld, omega_pe, lambda_D, gamma_theory = simulate_landau_damping()
    plot_landau_damping(history_ld, omega_pe, lambda_D, gamma_theory)

    print("\nPart 2: Bump-on-Tail (Inverse Landau Damping)")
    print("-" * 80)
    history_bt, omega_pe, lambda_D = simulate_bump_on_tail()
    plot_bump_on_tail(history_bt, omega_pe, lambda_D)

    print("\nDone! Generated 2 figures:")
    print("  - landau_damping.png")
    print("  - bump_on_tail_growth.png")
