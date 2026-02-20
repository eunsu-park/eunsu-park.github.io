#!/usr/bin/env python3
"""
Microwave Interferometry for Plasma Density Measurement

This script demonstrates microwave interferometry diagnostics including:
- Phase shift from line-integrated density
- Abel inversion for radial profile reconstruction
- Multi-chord interferometer simulation

Key Physics:
- Phase shift: Δφ = (e²/2mcε₀ω) ∫ ne dl
- Abel inversion: reconstruct n(r) from line integrals

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import simpson
from scipy.interpolate import interp1d

# Physical constants
QE = 1.602176634e-19    # C
ME = 9.10938356e-31     # kg
C = 2.99792458e8        # m/s
EPS0 = 8.854187817e-12  # F/m

def phase_shift_coefficient(freq):
    """
    Compute phase shift coefficient K.

    Δφ = K * ∫ ne dl

    where K = e²/(2·me·c·ε₀·ω) = e²·λ/(4π·me·c²·ε₀)

    Parameters:
    -----------
    freq : float
        Microwave frequency [Hz]

    Returns:
    --------
    K : float
        Phase shift coefficient [rad·m²]
    """
    omega = 2 * np.pi * freq
    K = QE**2 / (2 * ME * C * EPS0 * omega)
    return K

def parabolic_density_profile(r, ne0, a):
    """
    Parabolic density profile (typical for tokamak).

    ne(r) = ne0 * (1 - (r/a)²)^α

    Parameters:
    -----------
    r : array
        Radial position [m]
    ne0 : float
        Central density [m^-3]
    a : float
        Minor radius [m]

    Returns:
    --------
    ne : array
        Electron density [m^-3]
    """
    alpha = 2.0  # Profile shape parameter
    ne = ne0 * np.maximum(0, 1 - (r / a)**2)**alpha
    return ne

def line_integrated_density(impact_param, ne0, a, num_points=500):
    """
    Compute line-integrated density for a given impact parameter.

    ∫ ne(r) dl along chord at distance y from axis

    Parameters:
    -----------
    impact_param : float
        Impact parameter (distance from axis) [m]
    ne0 : float
        Central density [m^-3]
    a : float
        Minor radius [m]
    num_points : int
        Number of integration points

    Returns:
    --------
    ne_line : float
        Line-integrated density [m^-2]
    """
    if abs(impact_param) >= a:
        return 0.0

    # Integration along chord
    # For impact parameter y, chord goes from -sqrt(a²-y²) to +sqrt(a²-y²)
    x_max = np.sqrt(a**2 - impact_param**2)
    x = np.linspace(-x_max, x_max, num_points)

    # Radial distance at each point along chord
    r = np.sqrt(x**2 + impact_param**2)

    # Density at each point
    ne = parabolic_density_profile(r, ne0, a)

    # Integrate
    ne_line = simpson(ne, x=x)

    return ne_line

def compute_phase_shifts(impact_params, ne0, a, freq):
    """
    Compute phase shifts for multiple chords.

    Parameters:
    -----------
    impact_params : array
        Impact parameters for each chord [m]
    ne0 : float
        Central density [m^-3]
    a : float
        Minor radius [m]
    freq : float
        Microwave frequency [Hz]

    Returns:
    --------
    phase_shifts : array
        Phase shift for each chord [rad]
    ne_lines : array
        Line-integrated densities [m^-2]
    """
    K = phase_shift_coefficient(freq)

    ne_lines = np.array([line_integrated_density(y, ne0, a)
                        for y in impact_params])

    phase_shifts = K * ne_lines

    return phase_shifts, ne_lines

def abel_inversion_matrix(impact_params, r_grid):
    """
    Construct Abel inversion matrix using matrix method.

    For discrete measurements, we solve:
    ne_line = A · ne_radial

    where A is the Abel transform matrix.

    Parameters:
    -----------
    impact_params : array
        Impact parameters (sorted) [m]
    r_grid : array
        Radial grid points [m]

    Returns:
    --------
    A_matrix : 2D array
        Abel transform matrix
    """
    n_chords = len(impact_params)
    n_radial = len(r_grid)

    A = np.zeros((n_chords, n_radial))

    # For each chord i and radial shell j
    for i, y in enumerate(impact_params):
        for j in range(n_radial):
            if j == 0:
                r_inner = 0
                r_outer = (r_grid[0] + r_grid[1]) / 2 if n_radial > 1 else r_grid[0]
            elif j == n_radial - 1:
                r_inner = (r_grid[j-1] + r_grid[j]) / 2
                r_outer = r_grid[j]
            else:
                r_inner = (r_grid[j-1] + r_grid[j]) / 2
                r_outer = (r_grid[j] + r_grid[j+1]) / 2

            # Chord length through shell j
            if y < r_inner:
                # Chord passes through entire shell
                dl = 2 * (np.sqrt(r_outer**2 - y**2) - np.sqrt(r_inner**2 - y**2))
            elif y < r_outer:
                # Chord passes through outer part of shell
                dl = 2 * np.sqrt(r_outer**2 - y**2)
            else:
                # Chord misses shell
                dl = 0

            A[i, j] = dl

    return A

def reconstruct_density_profile(phase_shifts, impact_params, a, freq, n_radial=20):
    """
    Reconstruct radial density profile from phase shift measurements.

    Parameters:
    -----------
    phase_shifts : array
        Measured phase shifts [rad]
    impact_params : array
        Impact parameters [m]
    a : float
        Minor radius [m]
    freq : float
        Microwave frequency [Hz]
    n_radial : int
        Number of radial grid points

    Returns:
    --------
    r_grid : array
        Radial positions [m]
    ne_reconstructed : array
        Reconstructed density [m^-3]
    """
    K = phase_shift_coefficient(freq)

    # Convert phase shifts to line-integrated densities
    ne_lines = phase_shifts / K

    # Radial grid
    r_grid = np.linspace(0, a, n_radial)

    # Construct Abel matrix
    A = abel_inversion_matrix(impact_params, r_grid)

    # Solve via least squares (pseudo-inverse)
    ne_reconstructed = np.linalg.lstsq(A, ne_lines, rcond=None)[0]

    # Ensure non-negative
    ne_reconstructed = np.maximum(ne_reconstructed, 0)

    return r_grid, ne_reconstructed

def plot_interferometry():
    """
    Create comprehensive visualization of interferometry diagnostic.
    """
    # Plasma parameters (tokamak-like)
    ne0 = 5e19  # m^-3 (central density)
    a = 0.5     # m (minor radius)
    freq = 140e9  # Hz (140 GHz, typical for tokamak)

    # Wavelength
    wavelength = C / freq

    print("=" * 70)
    print("Microwave Interferometry Diagnostic")
    print("=" * 70)
    print(f"Central density: {ne0:.2e} m^-3")
    print(f"Minor radius: {a:.2f} m")
    print(f"Microwave frequency: {freq/1e9:.0f} GHz")
    print(f"Wavelength: {wavelength*1e3:.2f} mm")
    print(f"Phase shift coefficient: {phase_shift_coefficient(freq):.2e} rad·m²")
    print("=" * 70)

    # Create figure
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

    # True density profile
    r_fine = np.linspace(0, a, 500)
    ne_true = parabolic_density_profile(r_fine, ne0, a)

    # Plot 1: True density profile
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(r_fine * 100, ne_true / 1e19, 'b-', linewidth=2.5)
    ax1.set_xlabel('Radius (cm)', fontsize=11)
    ax1.set_ylabel(r'Density ($10^{19}$ m$^{-3}$)', fontsize=11)
    ax1.set_title('True Radial Density Profile', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

    # Simulate multi-chord interferometer
    n_chords = 8
    impact_params = np.linspace(-a * 0.9, a * 0.9, n_chords)

    # Compute phase shifts
    phase_shifts, ne_lines = compute_phase_shifts(impact_params, ne0, a, freq)

    # Add some noise
    np.random.seed(42)
    noise_level = 0.02
    phase_shifts_noisy = phase_shifts + np.random.normal(0, noise_level * phase_shifts.max(),
                                                          size=phase_shifts.shape)

    # Plot 2: Chord geometry
    ax2 = fig.add_subplot(gs[0, 1])

    # Draw plasma boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    ax2.plot(a * 100 * np.cos(theta), a * 100 * np.sin(theta),
            'k-', linewidth=2, label='Plasma boundary')

    # Draw chords
    colors = plt.cm.viridis(np.linspace(0, 1, n_chords))

    for i, (y, color) in enumerate(zip(impact_params, colors)):
        if abs(y) < a:
            x_max = np.sqrt(a**2 - y**2)
            ax2.plot([-x_max * 100, x_max * 100], [y * 100, y * 100],
                    color=color, linewidth=1.5, alpha=0.7, label=f'Chord {i+1}')

    ax2.set_xlabel('x (cm)', fontsize=11)
    ax2.set_ylabel('y (cm)', fontsize=11)
    ax2.set_title('Multi-Chord Interferometer Geometry',
                  fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, ncol=2, loc='upper right')

    # Plot 3: Phase shifts vs impact parameter
    ax3 = fig.add_subplot(gs[1, 0])

    ax3.plot(impact_params * 100, phase_shifts, 'b-', linewidth=2,
            label='True')
    ax3.plot(impact_params * 100, phase_shifts_noisy, 'ro', markersize=8,
            label='Measured (with noise)')

    ax3.set_xlabel('Impact Parameter (cm)', fontsize=11)
    ax3.set_ylabel('Phase Shift (rad)', fontsize=11)
    ax3.set_title('Measured Phase Shifts', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # Plot 4: Line-integrated density
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.plot(impact_params * 100, ne_lines / 1e19, 'b-', linewidth=2,
            label='True')
    ax4.plot(impact_params * 100, phase_shifts_noisy / phase_shift_coefficient(freq) / 1e19,
            'ro', markersize=8, label='From measured Δφ')

    ax4.set_xlabel('Impact Parameter (cm)', fontsize=11)
    ax4.set_ylabel(r'$\int n_e \, dl$ ($10^{19}$ m$^{-2}$)', fontsize=11)
    ax4.set_title('Line-Integrated Density', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    # Plot 5: Abel inversion reconstruction
    ax5 = fig.add_subplot(gs[2, :])

    # Reconstruct density profile
    r_recon, ne_recon = reconstruct_density_profile(phase_shifts_noisy,
                                                     np.abs(impact_params),
                                                     a, freq, n_radial=15)

    # Plot true vs reconstructed
    ax5.plot(r_fine * 100, ne_true / 1e19, 'b-', linewidth=2.5,
            label='True profile')
    ax5.plot(r_recon * 100, ne_recon / 1e19, 'ro-', linewidth=2,
            markersize=8, label=f'Reconstructed ({n_chords} chords)')

    # Try with fewer chords
    n_chords_few = 4
    impact_params_few = np.linspace(0, a * 0.9, n_chords_few)
    phase_shifts_few, _ = compute_phase_shifts(impact_params_few, ne0, a, freq)
    phase_shifts_few_noisy = phase_shifts_few + np.random.normal(
        0, noise_level * phase_shifts_few.max(), size=phase_shifts_few.shape)

    r_recon_few, ne_recon_few = reconstruct_density_profile(
        phase_shifts_few_noisy, impact_params_few, a, freq, n_radial=10)

    ax5.plot(r_recon_few * 100, ne_recon_few / 1e19, 'gs-', linewidth=2,
            markersize=8, alpha=0.7, label=f'Reconstructed ({n_chords_few} chords)')

    ax5.set_xlabel('Radius (cm)', fontsize=11)
    ax5.set_ylabel(r'Density ($10^{19}$ m$^{-3}$)', fontsize=11)
    ax5.set_title('Abel Inversion: Reconstructed Density Profile',
                  fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10, loc='upper right')
    ax5.set_xlim([0, a * 100])

    # Add text box with reconstruction quality
    rms_error_8 = np.sqrt(np.mean((np.interp(r_recon, r_fine, ne_true) - ne_recon)**2))
    rms_error_4 = np.sqrt(np.mean((np.interp(r_recon_few, r_fine, ne_true) - ne_recon_few)**2))

    textstr = '\n'.join([
        'Reconstruction Quality:',
        f'8 chords: RMS error = {rms_error_8/1e19:.2f}×10¹⁹ m⁻³',
        f'4 chords: RMS error = {rms_error_4/1e19:.2f}×10¹⁹ m⁻³',
        '',
        'More chords → better reconstruction'
    ])

    ax5.text(0.98, 0.97, textstr, transform=ax5.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Microwave Interferometry: Density Measurement & Abel Inversion',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('interferometry.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'interferometry.png'")

    print(f"\nReconstruction results:")
    print(f"  8 chords: RMS error = {rms_error_8/ne0*100:.1f}% of peak density")
    print(f"  4 chords: RMS error = {rms_error_4/ne0*100:.1f}% of peak density")
    print(f"\nCentral density (reconstructed, 8 chords): {ne_recon[0]:.2e} m^-3")
    print(f"Central density (true): {ne0:.2e} m^-3")
    print(f"Error: {abs(ne_recon[0] - ne0)/ne0*100:.1f}%")

    print("=" * 70)

    plt.show()

if __name__ == "__main__":
    plot_interferometry()
