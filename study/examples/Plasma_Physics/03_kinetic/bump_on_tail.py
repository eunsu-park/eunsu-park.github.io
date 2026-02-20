#!/usr/bin/env python3
"""
Bump-on-Tail Instability - Detailed Analysis

This script provides a detailed analysis of bump-on-tail instability including:
- Linear growth phase
- Nonlinear saturation
- Plateau formation (quasilinear theory)
- Particle trapping in wave potential

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, epsilon_0, k
from scipy.interpolate import CubicSpline
from scipy.fft import fft, ifft, fftfreq


class VlasovPoisson1D:
    """1D-1V Vlasov-Poisson solver."""

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

    def initialize_bump_on_tail(self, n_beam_frac=0.1, v_beam_factor=3.0,
                                T_beam_factor=0.5, perturbation=0.01, k_pert=1):
        """Initialize with bump-on-tail distribution."""
        v_th = np.sqrt(2 * k * self.T / self.m)
        v_beam = v_beam_factor * v_th
        T_beam = T_beam_factor * self.T
        v_th_beam = np.sqrt(2 * k * T_beam / self.m)

        n_bg = self.n0 * (1 - n_beam_frac)
        n_beam = self.n0 * n_beam_frac

        # Background
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

        # Store initial distribution for comparison
        self.f_initial_v = f_total.copy()

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

    def compute_potential(self):
        """Compute electrostatic potential φ from E = -dφ/dx."""
        E_k = fft(self.E)
        phi_k = np.zeros_like(E_k, dtype=complex)
        phi_k[1:] = 1j * E_k[1:] / self.kx[1:]
        phi_k[0] = 0
        return np.real(ifft(phi_k))

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
        phi_history = np.zeros((n_steps + 1, self.Nx))
        f_history = []
        f_avg_history = []  # Velocity-averaged f(v)

        self.compute_electric_field()
        t_history[0] = 0
        E_history[0, :] = self.E
        phi_history[0, :] = self.compute_potential()
        f_history.append(self.f.copy())
        f_avg_history.append(np.mean(self.f, axis=0))

        for n in range(n_steps):
            self.step_strang(dt)
            t_history[n + 1] = (n + 1) * dt
            E_history[n + 1, :] = self.E
            phi_history[n + 1, :] = self.compute_potential()

            if n % save_interval == 0:
                f_history.append(self.f.copy())
                f_avg_history.append(np.mean(self.f, axis=0))

        return {
            't': t_history,
            'E': E_history,
            'phi': phi_history,
            'f_snapshots': f_history,
            'f_avg': np.array(f_avg_history),
            'x': self.x,
            'v': self.v
        }


def simulate_full_evolution():
    """Simulate full nonlinear evolution of bump-on-tail."""

    # Parameters
    n0 = 1e16
    T = 1e4
    v_th = np.sqrt(2 * k * T / m_e)

    omega_pe = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))
    T_pe = 2 * np.pi / omega_pe
    lambda_D = np.sqrt(epsilon_0 * k * T / (n0 * e**2))

    print(f"\nBump-on-Tail Full Evolution:")
    print(f"  Thermal velocity: {v_th:.2e} m/s")
    print(f"  Plasma period: {T_pe:.2e} s")

    # Grid
    Nx = 128
    Nv = 512  # High resolution for plateau formation
    Lx = 20 * lambda_D
    v_max = 8 * v_th

    # Initialize
    solver = VlasovPoisson1D(Nx, Nv, Lx, v_max, n0, T)
    solver.initialize_bump_on_tail(n_beam_frac=0.15, v_beam_factor=3.0,
                                   T_beam_factor=0.3, perturbation=0.02, k_pert=1)

    # Run long enough to see saturation
    t_end = 200 * T_pe
    dt = T_pe / 50

    print(f"  Running for {t_end/T_pe:.1f} T_pe...")

    history = solver.run(t_end, dt, save_interval=max(1, int(t_end/dt) // 100))

    return history, solver, omega_pe, lambda_D, v_th


def plot_full_evolution(history, solver, omega_pe, lambda_D, v_th):
    """Plot complete evolution: linear, nonlinear, saturation."""

    t = history['t']
    E = history['E']
    phi = history['phi']
    x = history['x']
    v = history['v']
    f_snapshots = history['f_snapshots']
    f_avg = history['f_avg']

    T_pe = 2 * np.pi / omega_pe

    # Electric field energy
    E_energy = np.sum(E**2, axis=1)

    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 14))

    # Plot 1: Energy evolution (3 phases)
    ax1 = plt.subplot(3, 4, 1)
    ax1.semilogy(t / T_pe, E_energy, 'b-', linewidth=2)

    # Mark phases
    t_linear_end = 50
    t_nonlinear_end = 120

    ax1.axvspan(0, t_linear_end, alpha=0.2, color='green', label='Linear growth')
    ax1.axvspan(t_linear_end, t_nonlinear_end, alpha=0.2, color='orange',
                label='Nonlinear')
    ax1.axvspan(t_nonlinear_end, t[-1]/T_pe, alpha=0.2, color='red',
                label='Saturation')

    ax1.set_xlabel('Time [T_pe]', fontsize=11)
    ax1.set_ylabel('∫E² dx [a.u.]', fontsize=11)
    ax1.set_title('Energy Evolution', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distribution evolution in velocity
    ax2 = plt.subplot(3, 4, 2)

    # Plot initial, intermediate, and final distributions
    time_indices = [0, len(f_avg)//3, 2*len(f_avg)//3, -1]
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['Initial', 'Linear', 'Nonlinear', 'Saturated']

    for idx, color, label in zip(time_indices, colors, labels):
        ax2.plot(v / v_th, f_avg[idx] * v_th, color=color, linewidth=2,
                label=label, alpha=0.8)

    ax2.set_xlabel('v [v_th]', fontsize=11)
    ax2.set_ylabel('f(v) × v_th', fontsize=11)
    ax2.set_title('Distribution Function Evolution', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-2, 6])

    # Plot 3: df/dv showing slope flattening
    ax3 = plt.subplot(3, 4, 3)

    for idx, color, label in zip(time_indices, colors, labels):
        dfdv = np.gradient(f_avg[idx], v)
        ax3.plot(v / v_th, dfdv * v_th**2, color=color, linewidth=2,
                label=label, alpha=0.8)

    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('v [v_th]', fontsize=11)
    ax3.set_ylabel('df/dv × v_th²', fontsize=11)
    ax3.set_title('Slope (Plateau Formation)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([1, 5])

    # Plot 4: Growth rate measurement
    ax4 = plt.subplot(3, 4, 4)

    # Calculate instantaneous growth rate
    log_E = np.log(E_energy + 1e-30)
    gamma_inst = np.gradient(log_E, t)

    ax4.plot(t / T_pe, gamma_inst * T_pe, 'b-', linewidth=2)
    ax4.axhline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Time [T_pe]', fontsize=11)
    ax4.set_ylabel('Growth Rate γ × T_pe', fontsize=11)
    ax4.set_title('Instantaneous Growth Rate', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([-0.1, 0.3])

    # Plots 5-12: Phase space snapshots at key times
    snapshot_times = [0, 30, 60, 90, 120, 150, 180, -1]

    for plot_idx, snap_idx in enumerate(snapshot_times):
        ax = plt.subplot(3, 4, 5 + plot_idx)

        if snap_idx == -1:
            snap_idx = len(f_snapshots) - 1

        if snap_idx < len(f_snapshots):
            f = f_snapshots[snap_idx]
            t_actual = snap_idx * (t[-1] / (len(f_snapshots) - 1))

            # Determine phase
            if t_actual < t_linear_end * T_pe:
                phase = 'Linear'
            elif t_actual < t_nonlinear_end * T_pe:
                phase = 'Nonlinear'
            else:
                phase = 'Saturated'

            cs = ax.contourf(x / lambda_D, v / v_th, f.T, levels=30, cmap='hot')
            ax.set_xlabel('x [λ_D]', fontsize=9)
            ax.set_ylabel('v [v_th]', fontsize=9)
            ax.set_title(f'{phase}: t={t_actual/T_pe:.1f} T_pe',
                        fontsize=10, fontweight='bold')
            ax.set_ylim([-3, 6])

    plt.tight_layout()
    plt.savefig('bump_on_tail_full_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_trapping_analysis(history, omega_pe, lambda_D, v_th):
    """Analyze particle trapping in wave potential."""

    t = history['t']
    E = history['E']
    phi = history['phi']
    x = history['x']
    v = history['v']
    f_snapshots = history['f_snapshots']

    T_pe = 2 * np.pi / omega_pe

    # Select time in nonlinear phase (when trapping is strong)
    t_trap_idx = len(f_snapshots) // 2

    f_trap = f_snapshots[t_trap_idx]
    phi_trap = phi[t_trap_idx * (len(t) // len(f_snapshots))]
    E_trap = E[t_trap_idx * (len(t) // len(f_snapshots))]

    # Wave phase velocity (estimate from k and omega)
    k_wave = 2 * np.pi / (x[-1] - x[0])
    # Approximate omega from df/dv=0 location (resonant velocity)
    v_phase = 3 * v_th  # Approximate

    # Trapping width
    phi_amplitude = (np.max(phi_trap) - np.min(phi_trap)) / 2
    omega_bounce = np.sqrt(abs(e * k_wave * phi_amplitude / m_e))
    v_trap = omega_bounce / k_wave

    print(f"\nTrapping Analysis:")
    print(f"  Wave phase velocity: {v_phase/v_th:.2f} v_th")
    print(f"  Potential amplitude: {phi_amplitude:.2e} V")
    print(f"  Bounce frequency: {omega_bounce:.2e} rad/s")
    print(f"  Trapping width: {v_trap/v_th:.2f} v_th")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Phase space with separatrix
    ax = axes[0, 0]
    cs = ax.contourf(x / lambda_D, v / v_th, f_trap.T, levels=30, cmap='hot')
    ax.set_xlabel('x [λ_D]', fontsize=12)
    ax.set_ylabel('v [v_th]', fontsize=12)
    ax.set_title('Phase Space (Nonlinear Phase)', fontsize=13, fontweight='bold')
    ax.set_ylim([1, 5])
    plt.colorbar(cs, ax=ax)

    # Plot 2: Wave potential
    ax = axes[0, 1]
    ax.plot(x / lambda_D, phi_trap, 'b-', linewidth=2)
    ax.set_xlabel('x [λ_D]', fontsize=12)
    ax.set_ylabel('φ [V]', fontsize=12)
    ax.set_title('Wave Potential', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: Trapped particle orbits (schematic in v-x)
    ax = axes[1, 0]
    # Draw approximate separatrix
    x_sep = np.linspace(0, x[-1], 100)
    v_sep_upper = v_phase + v_trap * np.cos(k_wave * x_sep)
    v_sep_lower = v_phase - v_trap * np.cos(k_wave * x_sep)

    ax.plot(x_sep / lambda_D, v_sep_upper / v_th, 'g--', linewidth=2,
            label='Separatrix (approx)')
    ax.plot(x_sep / lambda_D, v_sep_lower / v_th, 'g--', linewidth=2)
    ax.axhline(v_phase / v_th, color='red', linestyle=':', linewidth=2,
               label=f'v_phase = {v_phase/v_th:.1f} v_th')

    # Overlay phase space contours
    cs = ax.contour(x / lambda_D, v / v_th, f_trap.T, levels=10,
                    colors='white', linewidths=0.5, alpha=0.5)

    ax.set_xlabel('x [λ_D]', fontsize=12)
    ax.set_ylabel('v [v_th]', fontsize=12)
    ax.set_title('Trapping Region (Separatrix)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim([2, 4])

    # Plot 4: Cut through phase space at x=0
    ax = axes[1, 1]
    i_x0 = 0
    ax.plot(v / v_th, f_trap[i_x0, :] * v_th, 'b-', linewidth=2)
    ax.axvline(v_phase / v_th, color='red', linestyle='--', linewidth=2,
               label='Phase velocity')
    ax.axvspan((v_phase - v_trap) / v_th, (v_phase + v_trap) / v_th,
               alpha=0.2, color='green', label='Trapping region')

    ax.set_xlabel('v [v_th]', fontsize=12)
    ax.set_ylabel('f(v) × v_th', fontsize=12)
    ax.set_title('Velocity Distribution at x=0', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 5])

    plt.tight_layout()
    plt.savefig('bump_on_tail_trapping.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("BUMP-ON-TAIL INSTABILITY: DETAILED ANALYSIS")
    print("="*80)

    # Run simulation
    history, solver, omega_pe, lambda_D, v_th = simulate_full_evolution()

    print("\nGenerating analysis plots...")
    print("  1. Full evolution (linear → nonlinear → saturation)...")
    plot_full_evolution(history, solver, omega_pe, lambda_D, v_th)

    print("  2. Particle trapping analysis...")
    plot_trapping_analysis(history, omega_pe, lambda_D, v_th)

    print("\nKey Physics:")
    print("  - Linear phase: Exponential growth due to positive df/dv")
    print("  - Nonlinear phase: Wave-particle interaction, trapping begins")
    print("  - Saturation: Plateau formation (df/dv → 0), trapping vortices")

    print("\nDone! Generated 2 figures:")
    print("  - bump_on_tail_full_evolution.png")
    print("  - bump_on_tail_trapping.png")
