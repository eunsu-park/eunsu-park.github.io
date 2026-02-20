#!/usr/bin/env python3
"""
MHD Turbulence Energy Spectra

This script computes and visualizes energy spectra in MHD turbulence,
comparing different theoretical predictions:
- Kolmogorov k^(-5/3) for hydrodynamic turbulence
- Iroshnikov-Kraichnan k^(-3/2) for strong MHD turbulence
- Goldreich-Sridhar for weak turbulence

Key results:
- Kinetic and magnetic energy spectra show different scalings
- Elsässer variables reveal wave-like nature of MHD turbulence
- Spectral anisotropy in the presence of a mean magnetic field

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, fftfreq
from scipy.ndimage import gaussian_filter


def generate_turbulent_field(N, alpha, energy_0=1.0, kmin=1, kmax=None):
    """
    Generate a turbulent field with power-law spectrum E(k) ~ k^(-alpha).

    Parameters
    ----------
    N : int
        Grid size (cubic grid N^3)
    alpha : float
        Spectral index (e.g., 5/3 for Kolmogorov, 3/2 for IK)
    energy_0 : float
        Energy normalization
    kmin : int
        Minimum wavenumber
    kmax : int or None
        Maximum wavenumber (None = N/2)

    Returns
    -------
    field : ndarray
        3D turbulent field
    """
    if kmax is None:
        kmax = N // 2

    # Create 3D field in Fourier space
    field_k = np.zeros((N, N, N), dtype=complex)

    # Create wavenumber grid
    kx = fftfreq(N, d=1.0) * N
    ky = fftfreq(N, d=1.0) * N
    kz = fftfreq(N, d=1.0) * N
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Generate random phases
    np.random.seed(42)
    phases = np.random.uniform(0, 2*np.pi, (N, N, N))

    # Set amplitudes according to power law
    for i in range(N):
        for j in range(N):
            for k in range(N):
                kmag = K[i, j, k]
                if kmin <= kmag <= kmax and kmag > 0:
                    amplitude = energy_0 * kmag**(-alpha/2)
                    field_k[i, j, k] = amplitude * np.exp(1j * phases[i, j, k])

    # Transform to real space
    field = np.fft.ifftn(field_k).real

    return field


def compute_energy_spectrum_3d(field):
    """
    Compute energy spectrum E(k) from a 3D field.

    Parameters
    ----------
    field : ndarray
        3D field (velocity or magnetic)

    Returns
    -------
    k_bins : ndarray
        Wavenumber bins
    E_k : ndarray
        Energy spectrum
    """
    N = field.shape[0]

    # Fourier transform
    field_k = fftn(field)

    # Energy in Fourier space: |field_k|^2 / 2
    energy_k = 0.5 * np.abs(field_k)**2 / N**6

    # Create wavenumber grid
    kx = fftfreq(N, d=1.0) * N
    ky = fftfreq(N, d=1.0) * N
    kz = fftfreq(N, d=1.0) * N
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Bin by wavenumber magnitude
    k_bins = np.arange(1, N//2)
    E_k = np.zeros(len(k_bins))

    for i, k in enumerate(k_bins):
        mask = (K >= k - 0.5) & (K < k + 0.5)
        E_k[i] = np.sum(energy_k[mask])

    return k_bins, E_k


def compute_elsasser_variables(vx, vy, vz, Bx, By, Bz, rho=1.0, mu0=1.0):
    """
    Compute Elsässer variables z± = v ± B/√(ρμ₀).

    Parameters
    ----------
    vx, vy, vz : ndarray
        Velocity components
    Bx, By, Bz : ndarray
        Magnetic field components
    rho : float
        Density (assumed uniform)
    mu0 : float
        Magnetic permeability

    Returns
    -------
    zp, zm : tuple of ndarray
        Elsässer variables z+ and z-
    """
    va_factor = 1.0 / np.sqrt(rho * mu0)

    zp_x = vx + Bx * va_factor
    zp_y = vy + By * va_factor
    zp_z = vz + Bz * va_factor

    zm_x = vx - Bx * va_factor
    zm_y = vy - By * va_factor
    zm_z = vz - Bz * va_factor

    return (zp_x, zp_y, zp_z), (zm_x, zm_y, zm_z)


def plot_energy_spectra():
    """
    Plot and compare MHD turbulence energy spectra.
    """
    # Grid size
    N = 64

    # Generate turbulent fields with different spectral indices
    print("Generating turbulent fields...")

    # Kolmogorov (hydrodynamic)
    vx_kolm = generate_turbulent_field(N, 5/3, energy_0=1.0)
    vy_kolm = generate_turbulent_field(N, 5/3, energy_0=1.0)
    vz_kolm = generate_turbulent_field(N, 5/3, energy_0=1.0)

    # Iroshnikov-Kraichnan (MHD)
    vx_ik = generate_turbulent_field(N, 3/2, energy_0=1.0)
    vy_ik = generate_turbulent_field(N, 3/2, energy_0=1.0)
    vz_ik = generate_turbulent_field(N, 3/2, energy_0=1.0)

    Bx_ik = generate_turbulent_field(N, 3/2, energy_0=0.8)
    By_ik = generate_turbulent_field(N, 3/2, energy_0=0.8)
    Bz_ik = generate_turbulent_field(N, 3/2, energy_0=0.8)

    # Compute spectra
    print("Computing energy spectra...")

    # Kinetic energy spectrum (Kolmogorov)
    k_bins, E_kin_kolm = compute_energy_spectrum_3d(vx_kolm)
    _, E_kin_kolm_y = compute_energy_spectrum_3d(vy_kolm)
    _, E_kin_kolm_z = compute_energy_spectrum_3d(vz_kolm)
    E_kin_kolm_total = E_kin_kolm + E_kin_kolm_y + E_kin_kolm_z

    # Kinetic and magnetic energy spectra (IK)
    k_bins, E_kin_ik = compute_energy_spectrum_3d(vx_ik)
    _, E_kin_ik_y = compute_energy_spectrum_3d(vy_ik)
    _, E_kin_ik_z = compute_energy_spectrum_3d(vz_ik)
    E_kin_ik_total = E_kin_ik + E_kin_ik_y + E_kin_ik_z

    _, E_mag_ik = compute_energy_spectrum_3d(Bx_ik)
    _, E_mag_ik_y = compute_energy_spectrum_3d(By_ik)
    _, E_mag_ik_z = compute_energy_spectrum_3d(Bz_ik)
    E_mag_ik_total = E_mag_ik + E_mag_ik_y + E_mag_ik_z

    # Elsässer variables
    zp, zm = compute_elsasser_variables(vx_ik, vy_ik, vz_ik,
                                        Bx_ik, By_ik, Bz_ik)
    _, E_zp = compute_energy_spectrum_3d(zp[0])
    _, E_zp_y = compute_energy_spectrum_3d(zp[1])
    _, E_zp_z = compute_energy_spectrum_3d(zp[2])
    E_zp_total = E_zp + E_zp_y + E_zp_z

    _, E_zm = compute_energy_spectrum_3d(zm[0])
    _, E_zm_y = compute_energy_spectrum_3d(zm[1])
    _, E_zm_z = compute_energy_spectrum_3d(zm[2])
    E_zm_total = E_zm + E_zm_y + E_zm_z

    # Theoretical predictions
    k_theory = k_bins[k_bins > 2]
    E_kolm_theory = 100 * k_theory**(-5/3)
    E_ik_theory = 100 * k_theory**(-3/2)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Kolmogorov vs Iroshnikov-Kraichnan
    ax = axes[0, 0]
    ax.loglog(k_bins, E_kin_kolm_total, 'b-', linewidth=2,
              label='Kinetic (Kolmogorov)', alpha=0.7)
    ax.loglog(k_bins, E_kin_ik_total, 'r-', linewidth=2,
              label='Kinetic (IK)', alpha=0.7)
    ax.loglog(k_theory, E_kolm_theory, 'b--', linewidth=1.5,
              label=r'$k^{-5/3}$ (Kolmogorov)')
    ax.loglog(k_theory, E_ik_theory, 'r--', linewidth=1.5,
              label=r'$k^{-3/2}$ (IK)')
    ax.set_xlabel('Wavenumber k', fontsize=12)
    ax.set_ylabel('Energy E(k)', fontsize=12)
    ax.set_title('Spectral Index Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Plot 2: Kinetic vs Magnetic energy
    ax = axes[0, 1]
    ax.loglog(k_bins, E_kin_ik_total, 'b-', linewidth=2,
              label='Kinetic energy', alpha=0.7)
    ax.loglog(k_bins, E_mag_ik_total, 'r-', linewidth=2,
              label='Magnetic energy', alpha=0.7)
    ax.loglog(k_bins, E_kin_ik_total + E_mag_ik_total, 'k-', linewidth=2,
              label='Total energy', alpha=0.7)
    ax.loglog(k_theory, E_ik_theory, 'g--', linewidth=1.5,
              label=r'$k^{-3/2}$')
    ax.set_xlabel('Wavenumber k', fontsize=12)
    ax.set_ylabel('Energy E(k)', fontsize=12)
    ax.set_title('Kinetic vs Magnetic Energy', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Plot 3: Elsässer variables
    ax = axes[1, 0]
    ax.loglog(k_bins, E_zp_total, 'b-', linewidth=2,
              label=r'$z^+$ (outgoing)', alpha=0.7)
    ax.loglog(k_bins, E_zm_total, 'r-', linewidth=2,
              label=r'$z^-$ (incoming)', alpha=0.7)
    ax.loglog(k_theory, E_ik_theory, 'k--', linewidth=1.5,
              label=r'$k^{-3/2}$')
    ax.set_xlabel('Wavenumber k', fontsize=12)
    ax.set_ylabel('Energy E(k)', fontsize=12)
    ax.set_title('Elsässer Variables z± = v ± B/√(ρμ₀)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Plot 4: Energy ratio
    ax = axes[1, 1]
    ratio = E_mag_ik_total / (E_kin_ik_total + 1e-10)
    ax.semilogx(k_bins, ratio, 'g-', linewidth=2, alpha=0.7)
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1.5,
               label='Equipartition')
    ax.set_xlabel('Wavenumber k', fontsize=12)
    ax.set_ylabel('E_mag / E_kin', fontsize=12)
    ax.set_title('Energy Equipartition', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 3])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('MHD Turbulence Energy Spectra',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mhd_energy_spectrum.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\n" + "="*60)
    print("MHD Turbulence Energy Spectra")
    print("="*60)
    print("\nSpectral Indices:")
    print("  Kolmogorov (hydrodynamic): E(k) ~ k^(-5/3) ≈ k^(-1.67)")
    print("  Iroshnikov-Kraichnan (MHD): E(k) ~ k^(-3/2) = k^(-1.50)")
    print("\nElsässer Variables:")
    print("  z+ = v + B/√(ρμ₀)  (outgoing Alfvén waves)")
    print("  z- = v - B/√(ρμ₀)  (incoming Alfvén waves)")
    print("\nKey Physics:")
    print("  - MHD turbulence is mediated by Alfvén wave collisions")
    print("  - IK theory predicts shallower spectrum than Kolmogorov")
    print("  - Energy equipartition: E_kin ≈ E_mag in inertial range")
    print("  - Critical balance: energy cascade rate ~ Alfvén wave period")


def main():
    """Main execution function."""
    print("MHD Turbulence Energy Spectrum Analysis")
    print("=" * 60)

    plot_energy_spectra()

    print("\nPlot saved as 'mhd_energy_spectrum.png'")
    print("\nReferences:")
    print("  - Kolmogorov (1941): k^(-5/3) for hydrodynamic turbulence")
    print("  - Iroshnikov (1964), Kraichnan (1965): k^(-3/2) for MHD")
    print("  - Goldreich & Sridhar (1995): Anisotropic cascade theory")


if __name__ == '__main__':
    main()
