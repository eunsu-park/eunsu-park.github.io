#!/usr/bin/env python3
"""
Electromagnetic Wave Propagation in Unmagnetized Plasma

This script demonstrates EM wave propagation in a plasma, including:
- Dispersion relation: ω² = ωpe² + k²c²
- Cutoff at ω = ωpe (evanescent below)
- Refractive index: n = sqrt(1 - ωpe²/ω²)
- Wave field visualization entering a plasma density gradient

Key Physics:
- EM waves cannot propagate below the plasma frequency (cutoff)
- Evanescent decay with skin depth δ ~ c/ωpe
- Group velocity vg = c²k/ω < c

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

# Physical constants
EPS0 = 8.854187817e-12  # F/m
QE = 1.602176634e-19    # C
ME = 9.10938356e-31     # kg
C = 2.99792458e8        # m/s

def compute_plasma_frequency(ne):
    """
    Compute electron plasma frequency.

    Parameters:
    -----------
    ne : float or array
        Electron density [m^-3]

    Returns:
    --------
    omega_pe : float or array
        Plasma frequency [rad/s]
    """
    omega_pe = np.sqrt(ne * QE**2 / (ME * EPS0))
    return omega_pe

def em_wave_dispersion(k, omega_pe):
    """
    Compute EM wave dispersion in plasma: ω² = ωpe² + k²c².

    Parameters:
    -----------
    k : array
        Wavenumber [rad/m]
    omega_pe : float
        Plasma frequency [rad/s]

    Returns:
    --------
    omega : array
        Angular frequency [rad/s]
    """
    omega_sq = omega_pe**2 + (k * C)**2
    return np.sqrt(omega_sq)

def refractive_index(omega, omega_pe):
    """
    Compute refractive index: n = sqrt(1 - ωpe²/ω²).

    Parameters:
    -----------
    omega : array
        Angular frequency [rad/s]
    omega_pe : float
        Plasma frequency [rad/s]

    Returns:
    --------
    n : array
        Refractive index (complex if ω < ωpe)
    """
    ratio = omega_pe**2 / omega**2

    # Real refractive index when ω > ωpe
    # Imaginary when ω < ωpe (evanescent)
    n = np.sqrt(1 - ratio + 0j)  # Allow complex values

    return n

def skin_depth(omega, omega_pe):
    """
    Compute skin depth for evanescent waves (ω < ωpe).

    δ = c / sqrt(ωpe² - ω²)

    Parameters:
    -----------
    omega : float
        Angular frequency [rad/s]
    omega_pe : float
        Plasma frequency [rad/s]

    Returns:
    --------
    delta : float
        Skin depth [m]
    """
    if omega >= omega_pe:
        return np.inf
    else:
        delta = C / np.sqrt(omega_pe**2 - omega**2)
        return delta

def simulate_wave_entering_plasma(ne_max, freq, num_cycles=3):
    """
    Simulate EM wave entering a plasma density gradient.

    Parameters:
    -----------
    ne_max : float
        Maximum plasma density [m^-3]
    freq : float
        Wave frequency [Hz]
    num_cycles : int
        Number of wave cycles to simulate

    Returns:
    --------
    x, Ex_total : arrays for visualization
    """
    omega = 2 * np.pi * freq
    omega_pe_max = compute_plasma_frequency(ne_max)

    # Spatial grid
    x = np.linspace(-0.5, 2.0, 2000)  # meters

    # Density profile: linear ramp from 0 to ne_max
    ne = np.zeros_like(x)
    ramp_start = 0.0
    ramp_end = 1.0
    mask = (x >= ramp_start) & (x <= ramp_end)
    ne[mask] = ne_max * (x[mask] - ramp_start) / (ramp_end - ramp_start)
    ne[x > ramp_end] = ne_max

    # Plasma frequency profile
    omega_pe = compute_plasma_frequency(ne)

    # Refractive index profile
    n = refractive_index(omega, omega_pe)

    # Find cutoff position (where ω = ωpe)
    cutoff_idx = np.argmin(np.abs(omega - omega_pe[x >= 0]))
    x_cutoff = x[x >= 0][cutoff_idx] if omega < omega_pe_max else np.inf

    return x, ne, omega_pe, n, x_cutoff

def plot_em_wave_propagation():
    """
    Create comprehensive visualization of EM wave propagation in plasma.
    """
    # Plasma parameters
    ne = 1e17  # m^-3 (lower density for radio waves)

    omega_pe = compute_plasma_frequency(ne)
    f_pe = omega_pe / (2 * np.pi)

    print("=" * 70)
    print("EM Wave Propagation in Plasma")
    print("=" * 70)
    print(f"Electron density: {ne:.2e} m^-3")
    print(f"Plasma frequency: {f_pe/1e9:.3f} GHz")
    print(f"Cutoff wavelength: {C/f_pe:.3f} m")
    print("=" * 70)

    # Create figure
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Plot 1: Dispersion relation
    ax1 = fig.add_subplot(gs[0, :])

    k = np.linspace(0, 3 * omega_pe / C, 1000)
    omega = em_wave_dispersion(k, omega_pe)

    # Light line in vacuum
    omega_vacuum = k * C

    ax1.plot(k * C / omega_pe, omega / omega_pe,
             'b-', linewidth=2, label='Plasma')
    ax1.plot(k * C / omega_pe, omega_vacuum / omega_pe,
             'r--', linewidth=2, label='Vacuum (light line)')
    ax1.axhline(y=1.0, color='green', linestyle=':', linewidth=2,
                label=r'Cutoff ($\omega = \omega_{pe}$)')

    # Shade forbidden region
    ax1.fill_between([0, 3], [0, 0], [1, 1], alpha=0.2, color='red',
                     label='Forbidden (evanescent)')

    ax1.set_xlabel(r'$k c / \omega_{pe}$', fontsize=12)
    ax1.set_ylabel(r'$\omega / \omega_{pe}$', fontsize=12)
    ax1.set_title('EM Wave Dispersion Relation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.set_xlim([0, 3])
    ax1.set_ylim([0, 3])

    # Plot 2: Refractive index vs frequency
    ax2 = fig.add_subplot(gs[1, 0])

    omega_array = np.linspace(0.5 * omega_pe, 3 * omega_pe, 1000)
    n = refractive_index(omega_array, omega_pe)

    ax2.plot(omega_array / omega_pe, np.real(n),
             'b-', linewidth=2, label='Real part')
    ax2.plot(omega_array / omega_pe, np.abs(np.imag(n)),
             'r-', linewidth=2, label='Imaginary part')
    ax2.axvline(x=1.0, color='green', linestyle=':', linewidth=2,
                label='Cutoff')

    ax2.set_xlabel(r'$\omega / \omega_{pe}$', fontsize=12)
    ax2.set_ylabel('Refractive Index n', fontsize=12)
    ax2.set_title('Refractive Index', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim([0.5, 3])

    # Plot 3: Group velocity
    ax3 = fig.add_subplot(gs[1, 1])

    # For propagating waves (ω > ωpe)
    omega_prop = omega_array[omega_array > omega_pe]
    vg = C**2 * np.sqrt(omega_prop**2 - omega_pe**2) / omega_prop

    ax3.plot(omega_prop / omega_pe, vg / C,
             'b-', linewidth=2)
    ax3.axvline(x=1.0, color='green', linestyle=':', linewidth=2,
                label='Cutoff')
    ax3.axhline(y=1.0, color='r', linestyle='--', linewidth=1,
                label='Vacuum speed')

    ax3.set_xlabel(r'$\omega / \omega_{pe}$', fontsize=12)
    ax3.set_ylabel(r'$v_g / c$', fontsize=12)
    ax3.set_title('Group Velocity', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xlim([1, 3])
    ax3.set_ylim([0, 1.1])

    # Plot 4: Skin depth vs frequency
    ax4 = fig.add_subplot(gs[2, 0])

    omega_evan = np.linspace(0.1 * omega_pe, 0.99 * omega_pe, 1000)
    delta = np.array([skin_depth(w, omega_pe) for w in omega_evan])

    ax4.plot(omega_evan / omega_pe, delta * omega_pe / C,
             'r-', linewidth=2)
    ax4.axvline(x=1.0, color='green', linestyle=':', linewidth=2,
                label='Cutoff')

    ax4.set_xlabel(r'$\omega / \omega_{pe}$', fontsize=12)
    ax4.set_ylabel(r'Skin Depth ($\omega_{pe} / c$)', fontsize=12)
    ax4.set_title('Evanescent Wave Skin Depth', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_xlim([0.1, 1])
    ax4.set_yscale('log')

    # Plot 5 & 6: Wave entering plasma gradient
    # Case 1: Propagating (ω > ωpe)
    ax5 = fig.add_subplot(gs[2, 1])

    ne_max = 0.5e17  # Lower than ne, so wave can propagate
    freq_high = 1.5 * f_pe  # Above cutoff

    x, ne_profile, omega_pe_profile, n_profile, x_cutoff = \
        simulate_wave_entering_plasma(ne_max, freq_high)

    ax5_twin = ax5.twinx()
    ax5.plot(x, ne_profile / 1e17, 'g-', linewidth=2, label='Density')
    ax5.set_xlabel('Position (m)', fontsize=10)
    ax5.set_ylabel(r'$n_e$ ($10^{17}$ m$^{-3}$)', fontsize=10, color='g')
    ax5.tick_params(axis='y', labelcolor='g')

    ax5_twin.plot(x, np.real(n_profile), 'b-', linewidth=2, label='Re(n)')
    ax5_twin.set_ylabel('Refractive Index', fontsize=10, color='b')
    ax5_twin.tick_params(axis='y', labelcolor='b')
    ax5_twin.axhline(y=0, color='r', linestyle='--', linewidth=1)

    ax5.set_title(f'Propagating Wave (f = {freq_high/1e9:.2f} GHz > fpe)',
                  fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([-0.5, 2])

    # Case 2: Evanescent (ω < ωpe)
    ax6 = fig.add_subplot(gs[3, :])

    ne_max = 2e17  # Higher than ne, wave will be cutoff
    freq_low = 0.7 * f_pe  # Below cutoff

    x, ne_profile, omega_pe_profile, n_profile, x_cutoff = \
        simulate_wave_entering_plasma(ne_max, freq_low)

    ax6_twin = ax6.twinx()
    ax6.plot(x, ne_profile / 1e17, 'g-', linewidth=2, label='Density')
    ax6.axvline(x=x_cutoff, color='purple', linestyle=':', linewidth=2,
                label=f'Cutoff at x = {x_cutoff:.3f} m')
    ax6.set_xlabel('Position (m)', fontsize=10)
    ax6.set_ylabel(r'$n_e$ ($10^{17}$ m$^{-3}$)', fontsize=10, color='g')
    ax6.tick_params(axis='y', labelcolor='g')

    ax6_twin.plot(x, np.real(n_profile), 'b-', linewidth=2, label='Re(n)')
    ax6_twin.plot(x, np.abs(np.imag(n_profile)), 'r--', linewidth=2, label='|Im(n)|')
    ax6_twin.set_ylabel('Refractive Index', fontsize=10, color='b')
    ax6_twin.tick_params(axis='y', labelcolor='b')
    ax6_twin.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    ax6.set_title(f'Evanescent Wave (f = {freq_low/1e9:.2f} GHz < fpe)',
                  fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper left', fontsize=9)
    ax6_twin.legend(loc='upper right', fontsize=9)
    ax6.set_xlim([-0.5, 2])

    plt.suptitle('Electromagnetic Wave Propagation in Unmagnetized Plasma',
                 fontsize=16, fontweight='bold', y=0.997)

    plt.savefig('em_wave_plasma.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'em_wave_plasma.png'")

    plt.show()

if __name__ == "__main__":
    plot_em_wave_propagation()
