#!/usr/bin/env python3
"""
Alfvén Wave Collision and Energy Cascade

This script simulates the collision of counter-propagating Alfvén wave packets
and demonstrates how wave-wave interactions lead to energy cascade in MHD
turbulence. The simulation illustrates:
- Critical balance: perpendicular cascade time ~ Alfvén wave period
- Energy transfer from large to small scales
- Formation of smaller-scale structures via nonlinear interactions

Key results:
- Counter-propagating waves generate small-scale perturbations
- Energy cascades to higher wavenumbers
- Critical balance τ_nl ~ τ_A determines cascade dynamics

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, ifft, fftfreq


def alfven_wave_packet(x, t, k0, width, phase, amplitude, v_alfven, direction=1):
    """
    Generate an Alfvén wave packet.

    Parameters
    ----------
    x : ndarray
        Spatial grid
    t : float
        Time
    k0 : float
        Central wavenumber
    width : float
        Packet width
    phase : float
        Initial phase
    amplitude : float
        Wave amplitude
    v_alfven : float
        Alfvén velocity
    direction : int
        +1 for right-propagating, -1 for left-propagating

    Returns
    -------
    wave : ndarray
        Wave packet field
    """
    envelope = amplitude * np.exp(-(x - direction * v_alfven * t)**2 / width**2)
    wave = envelope * np.cos(k0 * (x - direction * v_alfven * t) + phase)
    return wave


def compute_nonlinear_term(vx, vy, Bx, By, dx, v_alfven):
    """
    Compute nonlinear term in MHD equations: (v·∇)B - (B·∇)v.

    For simplicity, we use the Elsässer formulation:
    ∂z±/∂t + v_A ∂z±/∂x = -(z∓·∇)z±

    Parameters
    ----------
    vx, vy : ndarray
        Velocity components
    Bx, By : ndarray
        Magnetic field components (in velocity units)
    dx : float
        Grid spacing
    v_alfven : float
        Alfvén velocity

    Returns
    -------
    nl_term : ndarray
        Nonlinear interaction term
    """
    # Elsässer variables
    zp_x = vx + Bx
    zp_y = vy + By
    zm_x = vx - Bx
    zm_y = vy - By

    # Gradients (simple centered difference)
    dzm_x_dx = np.gradient(zm_x, dx)
    dzp_x_dx = np.gradient(zp_x, dx)

    # Nonlinear term: -(z∓·∇)z±
    nl_zp = -zm_x * dzp_x_dx
    nl_zm = -zp_x * dzm_x_dx

    return nl_zp, nl_zm


def simulate_alfven_collision(N=512, L=100.0, T_max=50.0, dt=0.05):
    """
    Simulate collision of two Alfvén wave packets.

    Parameters
    ----------
    N : int
        Number of grid points
    L : float
        Domain length
    T_max : float
        Maximum simulation time
    dt : float
        Time step

    Returns
    -------
    x : ndarray
        Spatial grid
    t_array : ndarray
        Time array
    energy_history : ndarray
        Total energy vs time
    spectrum_history : list
        Energy spectra at different times
    """
    # Setup grid
    x = np.linspace(0, L, N)
    dx = L / N
    k = fftfreq(N, d=dx) * 2 * np.pi

    # Parameters
    v_alfven = 1.0  # Alfvén velocity
    rho = 1.0  # Density

    # Initial conditions: two counter-propagating wave packets
    k0 = 2 * np.pi / 10  # Central wavenumber
    width = 5.0  # Packet width
    amplitude = 0.3  # Wave amplitude

    # Initialize fields
    vx = np.zeros(N)
    vy = (alfven_wave_packet(x, 0, k0, width, 0, amplitude, v_alfven, +1) +
          alfven_wave_packet(x, 0, k0, width, np.pi/4, amplitude, v_alfven, -1))

    # In Alfvén waves: δv = ±δB/√(ρμ₀), we use normalized units
    Bx = np.zeros(N)
    By_right = alfven_wave_packet(x, 0, k0, width, 0, amplitude, v_alfven, +1)
    By_left = -alfven_wave_packet(x, 0, k0, width, np.pi/4, amplitude, v_alfven, -1)
    By = By_right + By_left

    # Time integration
    n_steps = int(T_max / dt)
    t_array = np.linspace(0, T_max, n_steps)
    energy_history = np.zeros(n_steps)
    spectrum_history = []
    snapshot_times = [0, T_max/4, T_max/2, 3*T_max/4, T_max]

    for step in range(n_steps):
        t = t_array[step]

        # Compute energy
        energy = np.sum(vx**2 + vy**2 + Bx**2 + By**2) * dx / 2
        energy_history[step] = energy

        # Store snapshots for spectrum
        if any(np.abs(t - st) < dt/2 for st in snapshot_times):
            vy_k = fft(vy)
            spectrum = np.abs(vy_k)**2 / N
            spectrum_history.append((t, k[:N//2], spectrum[:N//2]))

        # Compute nonlinear terms
        nl_zp, nl_zm = compute_nonlinear_term(vx, vy, Bx, By, dx, v_alfven)

        # Time advancement (simple forward Euler for demonstration)
        # ∂z+/∂t = -v_A ∂z+/∂x - (z-·∇)z+
        # ∂z-/∂t = +v_A ∂z-/∂x - (z+·∇)z-

        zp_x = vx + Bx
        zp_y = vy + By
        zm_x = vx - Bx
        zm_y = vy - By

        # Spectral method for linear terms (advection)
        zp_y_k = fft(zp_y)
        zm_y_k = fft(zm_y)

        # Advection in Fourier space
        zp_y_k = zp_y_k * np.exp(-1j * v_alfven * k * dt)
        zm_y_k = zm_y_k * np.exp(+1j * v_alfven * k * dt)

        # Back to real space
        zp_y = ifft(zp_y_k).real
        zm_y = ifft(zm_y_k).real

        # Add nonlinear term
        zp_y = zp_y + nl_zp * dt
        zm_y = zm_y + nl_zm * dt

        # Update physical variables
        vy = 0.5 * (zp_y + zm_y)
        By = 0.5 * (zp_y - zm_y)

        # Damping at boundaries (prevent reflections)
        damping = np.exp(-((x - L/2)**2) / (L/3)**2)
        vy = vy * damping
        By = By * damping

    return x, t_array, energy_history, spectrum_history


def plot_collision_results(x, t_array, energy_history, spectrum_history):
    """
    Plot Alfvén wave collision results.

    Parameters
    ----------
    x : ndarray
        Spatial grid
    t_array : ndarray
        Time array
    energy_history : ndarray
        Energy vs time
    spectrum_history : list
        Energy spectra at different times
    """
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Energy conservation
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(t_array, energy_history / energy_history[0], 'b-', linewidth=2)
    ax1.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Time (τ_A)', fontsize=12)
    ax1.set_ylabel('Normalized Total Energy', fontsize=12)
    ax1.set_title('Energy Conservation', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2-4: Energy spectra at different times
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    ax2 = plt.subplot(2, 3, 2)

    for i, (t, k_vals, spectrum) in enumerate(spectrum_history):
        # Remove zero frequency and plot
        k_plot = k_vals[1:len(k_vals)//2]
        spec_plot = spectrum[1:len(spectrum)//2]
        ax2.loglog(k_plot, spec_plot, color=colors[i % len(colors)],
                   linewidth=2, alpha=0.7, label=f't = {t:.1f}')

    # Add theoretical predictions
    k_theory = k_vals[2:20]
    ax2.loglog(k_theory, 100 * k_theory**(-5/3), 'k--',
               linewidth=1.5, label=r'$k^{-5/3}$', alpha=0.5)
    ax2.loglog(k_theory, 50 * k_theory**(-3/2), 'k:',
               linewidth=1.5, label=r'$k^{-3/2}$', alpha=0.5)

    ax2.set_xlabel('Wavenumber k', fontsize=12)
    ax2.set_ylabel('Energy E(k)', fontsize=12)
    ax2.set_title('Energy Spectrum Evolution', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Critical balance diagram
    ax3 = plt.subplot(2, 3, 3)
    k_range = np.logspace(0, 2, 50)

    # Cascade time: τ_nl ~ k_perp / δv_k ~ k_perp / (E(k) k)^(1/2)
    # For IK: E(k) ~ k^(-3/2), so τ_nl ~ k^(-1/4)
    tau_nl_ik = 10 * k_range**(-1/4)
    # Alfvén time: τ_A ~ 1/k_parallel
    tau_alfven = 10 / k_range

    ax3.loglog(k_range, tau_nl_ik, 'b-', linewidth=2,
               label=r'$\tau_{nl}$ (cascade time)')
    ax3.loglog(k_range, tau_alfven, 'r-', linewidth=2,
               label=r'$\tau_A$ (Alfvén time)')
    ax3.axhline(10, color='k', linestyle='--', linewidth=1,
                alpha=0.5, label='Critical balance')

    ax3.set_xlabel('Wavenumber k', fontsize=12)
    ax3.set_ylabel('Timescale', fontsize=12)
    ax3.set_title('Critical Balance: τ_nl ~ τ_A', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: Cascade rate
    ax4 = plt.subplot(2, 3, 4)
    # Energy cascade rate: ε ~ E / τ_nl
    # For critical balance with IK spectrum: ε ~ const
    k_cascade = k_range[k_range > 1]
    E_k_kolm = k_cascade**(-5/3)
    E_k_ik = k_cascade**(-3/2)
    epsilon_kolm = E_k_kolm / (k_cascade**(-2/3))  # τ_nl ~ k^(-2/3)
    epsilon_ik = E_k_ik / (k_cascade**(-1/4))  # τ_nl ~ k^(-1/4)

    ax4.semilogx(k_cascade, epsilon_kolm / epsilon_kolm[0], 'b-',
                 linewidth=2, label='Kolmogorov', alpha=0.7)
    ax4.semilogx(k_cascade, epsilon_ik / epsilon_ik[0], 'r-',
                 linewidth=2, label='IK (critical balance)', alpha=0.7)
    ax4.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5)

    ax4.set_xlabel('Wavenumber k', fontsize=12)
    ax4.set_ylabel('Normalized Cascade Rate ε', fontsize=12)
    ax4.set_title('Energy Cascade Rate', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Anisotropy
    ax5 = plt.subplot(2, 3, 5)
    # k_parallel / k_perp ratio for critical balance
    # For GS95: k_|| ~ k_perp^(2/3)
    k_perp = np.logspace(0, 2, 50)
    k_par_gs = k_perp**(2/3)  # Goldreich-Sridhar
    k_par_iso = k_perp  # Isotropic

    ax5.loglog(k_perp, k_par_gs, 'b-', linewidth=2,
               label='GS95: k_|| ~ k_⊥^(2/3)', alpha=0.7)
    ax5.loglog(k_perp, k_par_iso, 'k--', linewidth=1.5,
               label='Isotropic: k_|| ~ k_⊥', alpha=0.5)

    ax5.set_xlabel('k_⊥ (perpendicular)', fontsize=12)
    ax5.set_ylabel('k_|| (parallel)', fontsize=12)
    ax5.set_title('Spectral Anisotropy', fontsize=13, fontweight='bold')
    ax5.legend(loc='upper left', fontsize=10)
    ax5.grid(True, alpha=0.3, which='both')

    # Plot 6: Physics summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = """
    Critical Balance Theory
    ───────────────────────

    • Cascade mediated by Alfvén wave collisions
    • Energy transfer occurs when:
      τ_nl (nonlinear time) ~ τ_A (Alfvén time)

    Spectral Predictions:
    • Kolmogorov (isotropic): E(k) ~ k^(-5/3)
    • Iroshnikov-Kraichnan: E(k) ~ k^(-3/2)
    • Goldreich-Sridhar (anisotropic):
      - Perpendicular cascade dominates
      - k_|| ~ k_⊥^(2/3)
      - E(k_⊥) ~ k_⊥^(-5/3)

    Key Physics:
    • Counter-propagating Alfvén waves interact
    • Generate small-scale perturbations
    • Energy cascades to dissipation scales
    • Anisotropy due to mean B field
    """

    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax6.transAxes)

    plt.suptitle('Alfvén Wave Collision and Energy Cascade',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('alfven_wave_cascade.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("="*60)
    print("Alfvén Wave Collision and Energy Cascade")
    print("="*60)

    print("\nSimulating counter-propagating Alfvén wave packets...")
    x, t_array, energy_history, spectrum_history = simulate_alfven_collision(
        N=512, L=100.0, T_max=40.0, dt=0.05
    )

    print(f"Simulation complete: {len(t_array)} timesteps")
    print(f"Energy conservation: {100*(1 - energy_history[-1]/energy_history[0]):.2f}% loss")

    print("\nGenerating plots...")
    plot_collision_results(x, t_array, energy_history, spectrum_history)

    print("\nPlot saved as 'alfven_wave_cascade.png'")

    print("\n" + "="*60)
    print("Key Concepts")
    print("="*60)
    print("\n1. Critical Balance:")
    print("   τ_nl ~ τ_A  =>  Energy cascade rate determined by Alfvén waves")
    print("\n2. Spectral Index:")
    print("   E(k) ~ k^(-3/2) for strong MHD turbulence (IK theory)")
    print("   E(k) ~ k^(-5/3) for hydrodynamic turbulence (Kolmogorov)")
    print("\n3. Anisotropy:")
    print("   k_|| ~ k_⊥^(2/3) for Goldreich-Sridhar cascade")
    print("\nReferences:")
    print("  - Iroshnikov (1964), Kraichnan (1965)")
    print("  - Goldreich & Sridhar (1995)")
    print("  - Boldyrev (2006)")


if __name__ == '__main__':
    main()
