#!/usr/bin/env python3
"""
Whistler Wave Dispersion and Spectrogram

This script demonstrates whistler wave physics including:
- R-mode dispersion in magnetized plasma
- Whistler regime: ωci << ω << ωce
- Frequency-dependent group velocity
- Simulated whistler spectrogram showing falling tone

Key Physics:
- Dispersion: n² ≈ ωpe²/(ω·ωce) in whistler regime
- Group velocity: vg ∝ sqrt(ω) → higher frequencies travel faster
- Observed as falling tones in magnetospheric radio emissions

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Physical constants
EPS0 = 8.854187817e-12  # F/m
QE = 1.602176634e-19    # C
ME = 9.10938356e-31     # kg
MP = 1.672621898e-27    # kg
C = 2.99792458e8        # m/s

def compute_characteristic_frequencies(ne, B0, mi=MP):
    """
    Compute plasma and cyclotron frequencies.

    Parameters:
    -----------
    ne : float
        Electron density [m^-3]
    B0 : float
        Magnetic field strength [T]
    mi : float
        Ion mass [kg]

    Returns:
    --------
    dict : Characteristic frequencies
    """
    omega_pe = np.sqrt(ne * QE**2 / (ME * EPS0))
    omega_ce = QE * B0 / ME
    omega_ci = QE * B0 / mi

    f_pe = omega_pe / (2 * np.pi)
    f_ce = omega_ce / (2 * np.pi)
    f_ci = omega_ci / (2 * np.pi)

    return {
        'omega_pe': omega_pe,
        'omega_ce': omega_ce,
        'omega_ci': omega_ci,
        'f_pe': f_pe,
        'f_ce': f_ce,
        'f_ci': f_ci
    }

def whistler_dispersion_full(omega, ne, B0):
    """
    Full R-mode dispersion relation: n² = 1 - ωpe²/(ω(ω - ωce)).

    Parameters:
    -----------
    omega : array
        Angular frequency [rad/s]
    ne : float
        Electron density [m^-3]
    B0 : float
        Magnetic field [T]

    Returns:
    --------
    k : array
        Wavenumber [rad/m]
    """
    params = compute_characteristic_frequencies(ne, B0)
    omega_pe = params['omega_pe']
    omega_ce = params['omega_ce']

    # R-mode dispersion
    n_sq = 1 - omega_pe**2 / (omega * (omega - omega_ce))

    # Only valid for n² > 0
    n_sq = np.maximum(n_sq, 0)

    n = np.sqrt(n_sq)
    k = omega * n / C

    return k

def whistler_dispersion_approx(omega, ne, B0):
    """
    Approximate whistler dispersion: n² ≈ ωpe²/(ω·ωce).

    Valid in range: ωci << ω << ωce

    Parameters:
    -----------
    omega : array
        Angular frequency [rad/s]
    ne : float
        Electron density [m^-3]
    B0 : float
        Magnetic field [T]

    Returns:
    --------
    k : array
        Wavenumber [rad/m]
    """
    params = compute_characteristic_frequencies(ne, B0)
    omega_pe = params['omega_pe']
    omega_ce = params['omega_ce']

    # Whistler approximation
    n_sq = omega_pe**2 / (omega * omega_ce)

    n = np.sqrt(n_sq)
    k = omega * n / C

    return k

def group_velocity_whistler(omega, ne, B0):
    """
    Compute whistler group velocity: vg = dω/dk.

    In whistler regime: vg ∝ sqrt(ω)

    Parameters:
    -----------
    omega : array
        Angular frequency [rad/s]
    ne : float
        Electron density [m^-3]
    B0 : float
        Magnetic field [T]

    Returns:
    --------
    vg : array
        Group velocity [m/s]
    """
    # Use numerical derivative
    k = whistler_dispersion_full(omega, ne, B0)

    # Compute dω/dk
    vg = np.gradient(omega, k)

    return vg

def simulate_whistler_spectrogram(ne, B0, distance, f_start, f_end, num_freqs=100):
    """
    Simulate whistler spectrogram showing dispersion.

    Parameters:
    -----------
    ne : float
        Electron density [m^-3]
    B0 : float
        Magnetic field [T]
    distance : float
        Propagation distance [m]
    f_start : float
        Starting frequency [Hz]
    f_end : float
        Ending frequency [Hz]
    num_freqs : int
        Number of frequency components

    Returns:
    --------
    t, f, spectrogram : arrays for plotting
    """
    # Frequency array
    freqs = np.linspace(f_start, f_end, num_freqs)
    omega = 2 * np.pi * freqs

    # Compute group velocities
    vg = group_velocity_whistler(omega, ne, B0)

    # Arrival time at distance
    arrival_times = distance / vg

    # Normalize time to start at 0
    arrival_times = arrival_times - arrival_times.min()

    # Create time grid for spectrogram
    t_max = arrival_times.max() * 1.2
    t_grid = np.linspace(0, t_max, 500)

    # Create spectrogram (Gaussian packets)
    spectrogram = np.zeros((num_freqs, len(t_grid)))

    for i, (f, t_arrival) in enumerate(zip(freqs, arrival_times)):
        # Gaussian pulse centered at arrival time
        sigma = t_max / 50  # Pulse width
        pulse = np.exp(-(t_grid - t_arrival)**2 / (2 * sigma**2))
        spectrogram[i, :] = pulse

    return t_grid, freqs, spectrogram, arrival_times

def plot_whistler_waves():
    """
    Create comprehensive visualization of whistler wave properties.
    """
    # Magnetospheric parameters (Earth's plasmasphere)
    ne = 1e7  # m^-3 (10 cm^-3)
    B0 = 1e-6  # Tesla (10 nT)
    distance = 1e7  # 10,000 km propagation

    params = compute_characteristic_frequencies(ne, B0)

    print("=" * 70)
    print("Whistler Wave Parameters (Magnetospheric)")
    print("=" * 70)
    print(f"Electron density: {ne:.2e} m^-3")
    print(f"Magnetic field: {B0*1e9:.2f} nT")
    print(f"Electron plasma frequency: {params['f_pe']/1e3:.2f} kHz")
    print(f"Electron cyclotron frequency: {params['f_ce']/1e3:.2f} kHz")
    print(f"Ion cyclotron frequency: {params['f_ci']:.2f} Hz")
    print(f"Whistler regime: {params['f_ci']:.1f} Hz << f << {params['f_ce']/1e3:.1f} kHz")
    print("=" * 70)

    # Create figure
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Frequency range for whistler regime
    f_min = 100  # Hz
    f_max = 0.5 * params['f_ce'] / (2 * np.pi)  # Half electron cyclotron
    omega = 2 * np.pi * np.linspace(f_min, f_max, 1000)
    freqs = omega / (2 * np.pi)

    # Compute dispersion relations
    k_full = whistler_dispersion_full(omega, ne, B0)
    k_approx = whistler_dispersion_approx(omega, ne, B0)

    # Plot 1: Dispersion relation ω(k)
    ax1 = fig.add_subplot(gs[0, :])

    ax1.plot(k_full / 1e3, freqs / 1e3,
             'b-', linewidth=2, label='Full R-mode')
    ax1.plot(k_approx / 1e3, freqs / 1e3,
             'r--', linewidth=2, label='Whistler approximation')

    # Mark characteristic frequencies
    ax1.axhline(y=params['f_ce'] / (1e3 * 2 * np.pi), color='green',
                linestyle=':', linewidth=2, label=r'$f_{ce}$')
    ax1.axhline(y=params['f_ci'] / 1e3, color='purple',
                linestyle=':', linewidth=2, label=r'$f_{ci}$')

    ax1.set_xlabel('Wavenumber k (rad/km)', fontsize=12)
    ax1.set_ylabel('Frequency (kHz)', fontsize=12)
    ax1.set_title('Whistler Wave Dispersion Relation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, np.max(k_full) / 1e3])

    # Plot 2: Refractive index
    ax2 = fig.add_subplot(gs[1, 0])

    n_full = k_full * C / omega
    n_approx = k_approx * C / omega

    ax2.plot(freqs / 1e3, n_full,
             'b-', linewidth=2, label='Full')
    ax2.plot(freqs / 1e3, n_approx,
             'r--', linewidth=2, label='Approximation')

    ax2.set_xlabel('Frequency (kHz)', fontsize=12)
    ax2.set_ylabel('Refractive Index n', fontsize=12)
    ax2.set_title('Refractive Index', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_yscale('log')

    # Plot 3: Group velocity
    ax3 = fig.add_subplot(gs[1, 1])

    vg = group_velocity_whistler(omega, ne, B0)

    ax3.plot(freqs / 1e3, vg / 1e6,
             'b-', linewidth=2)

    # Theoretical vg ∝ sqrt(ω) in whistler regime
    vg_theory = 2 * C * omega / (params['omega_pe'] / np.sqrt(params['omega_ce'] / omega))
    ax3.plot(freqs / 1e3, vg_theory / 1e6,
             'r--', linewidth=2, label=r'$v_g \propto \sqrt{\omega}$')

    ax3.set_xlabel('Frequency (kHz)', fontsize=12)
    ax3.set_ylabel(r'Group Velocity (10$^6$ m/s)', fontsize=12)
    ax3.set_title('Group Velocity (Higher f travels faster)',
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # Plot 4: Phase velocity
    ax4 = fig.add_subplot(gs[2, 0])

    vph = omega / k_full

    ax4.plot(freqs / 1e3, vph / 1e6,
             'b-', linewidth=2)

    ax4.set_xlabel('Frequency (kHz)', fontsize=12)
    ax4.set_ylabel(r'Phase Velocity (10$^6$ m/s)', fontsize=12)
    ax4.set_title('Phase Velocity', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Whistler spectrogram
    ax5 = fig.add_subplot(gs[2, 1])

    f_start = 5e3  # 5 kHz
    f_end = 15e3   # 15 kHz

    t_grid, freqs_spec, spectrogram, arrival_times = \
        simulate_whistler_spectrogram(ne, B0, distance, f_start, f_end, num_freqs=100)

    # Plot spectrogram
    im = ax5.pcolormesh(t_grid * 1e3, freqs_spec / 1e3, spectrogram,
                        shading='auto', cmap='hot')

    # Overlay arrival time curve
    ax5.plot(arrival_times * 1e3, freqs_spec / 1e3,
             'c-', linewidth=2, label='Arrival time')

    ax5.set_xlabel('Time (ms)', fontsize=12)
    ax5.set_ylabel('Frequency (kHz)', fontsize=12)
    ax5.set_title(f'Whistler Spectrogram (Distance = {distance/1e6:.0f} km)',
                  fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9, loc='upper right')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('Intensity', fontsize=10)

    plt.suptitle('Whistler Waves in Magnetized Plasma',
                 fontsize=16, fontweight='bold', y=0.997)

    plt.savefig('whistler_wave.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'whistler_wave.png'")
    print(f"\nWhistler shows 'falling tone': high frequencies arrive first")
    print(f"Frequency drop from {f_end/1e3:.1f} to {f_start/1e3:.1f} kHz")
    print(f"over time span of {(arrival_times.max() - arrival_times.min())*1e3:.2f} ms")

    plt.show()

if __name__ == "__main__":
    plot_whistler_waves()
