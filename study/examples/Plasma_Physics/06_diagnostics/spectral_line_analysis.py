#!/usr/bin/env python3
"""
Spectral Line Analysis for Plasma Temperature and Density

This script demonstrates spectral line diagnostics including:
- Doppler broadening (temperature measurement)
- Stark broadening (density measurement)
- Voigt profile fitting

Key Physics:
- Doppler width: ΔλD/λ = sqrt(2kTi/mic²)
- Stark width: proportional to electron density
- Line shape: convolution of Gaussian (Doppler) + Lorentzian (Stark)

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import wofz
from scipy.optimize import curve_fit

# Physical constants
QE = 1.602176634e-19    # C
ME = 9.10938356e-31     # kg
MP = 1.672621898e-27    # kg
C = 2.99792458e8        # m/s
KB = 1.380649e-23       # J/K

def gaussian_profile(wavelength, lambda0, Ti, mi):
    """
    Gaussian (Doppler) line profile.

    I(λ) ∝ exp(-(λ-λ0)²/(2σD²))

    where σD = (λ0/c) * sqrt(2kTi/mi)

    Parameters:
    -----------
    wavelength : array
        Wavelength [m]
    lambda0 : float
        Central wavelength [m]
    Ti : float
        Ion temperature [eV]
    mi : float
        Ion mass [kg]

    Returns:
    --------
    profile : array
        Normalized intensity
    """
    Ti_joule = Ti * QE

    # Doppler width
    sigma_D = (lambda0 / C) * np.sqrt(2 * Ti_joule / mi)

    # Gaussian profile
    profile = np.exp(-(wavelength - lambda0)**2 / (2 * sigma_D**2))
    profile /= profile.max()  # Normalize

    return profile

def lorentzian_profile(wavelength, lambda0, ne):
    """
    Lorentzian (Stark) line profile.

    I(λ) ∝ γ² / ((λ-λ0)² + γ²)

    where γ is the Stark width, proportional to ne.

    Parameters:
    -----------
    wavelength : array
        Wavelength [m]
    lambda0 : float
        Central wavelength [m]
    ne : float
        Electron density [m^-3]

    Returns:
    --------
    profile : array
        Normalized intensity
    """
    # Stark width (empirical for hydrogen Balmer-α at 656.3 nm)
    # γ ≈ 1.0e-26 * ne^(2/3) [m]
    gamma = 1.0e-26 * ne**(2/3)

    # Lorentzian profile
    profile = gamma**2 / ((wavelength - lambda0)**2 + gamma**2)
    profile /= profile.max()  # Normalize

    return profile

def voigt_profile(wavelength, lambda0, Ti, mi, ne):
    """
    Voigt profile: convolution of Gaussian and Lorentzian.

    Uses the Faddeeva function for efficient computation.

    Parameters:
    -----------
    wavelength : array
        Wavelength [m]
    lambda0 : float
        Central wavelength [m]
    Ti : float
        Ion temperature [eV]
    mi : float
        Ion mass [kg]
    ne : float
        Electron density [m^-3]

    Returns:
    --------
    profile : array
        Normalized intensity
    """
    Ti_joule = Ti * QE

    # Doppler width
    sigma_D = (lambda0 / C) * np.sqrt(2 * Ti_joule / mi)

    # Stark width
    gamma = 1.0e-26 * ne**(2/3)

    # Voigt profile using Faddeeva function
    z = ((wavelength - lambda0) + 1j * gamma) / (sigma_D * np.sqrt(2))
    profile = np.real(wofz(z)) / (sigma_D * np.sqrt(2 * np.pi))

    profile /= profile.max()  # Normalize

    return profile

def instrument_broadening(wavelength, profile, sigma_inst):
    """
    Apply instrumental broadening (Gaussian convolution).

    Parameters:
    -----------
    wavelength : array
        Wavelength [m]
    profile : array
        Original profile
    sigma_inst : float
        Instrumental width [m]

    Returns:
    --------
    broadened : array
        Broadened profile
    """
    # Gaussian kernel
    kernel = np.exp(-wavelength**2 / (2 * sigma_inst**2))
    kernel /= kernel.sum()

    # Convolve
    broadened = np.convolve(profile, kernel, mode='same')
    broadened /= broadened.max()

    return broadened

def fit_voigt_profile(wavelength, intensity, lambda0_guess, Ti_guess, mi, ne_guess):
    """
    Fit Voigt profile to extract Ti and ne.

    Parameters:
    -----------
    wavelength : array
        Wavelength [m]
    intensity : array
        Measured intensity
    lambda0_guess : float
        Initial guess for central wavelength [m]
    Ti_guess : float
        Initial guess for temperature [eV]
    mi : float
        Ion mass [kg]
    ne_guess : float
        Initial guess for density [m^-3]

    Returns:
    --------
    Ti_fit, ne_fit, lambda0_fit : fitted parameters
    """
    # Define fitting function
    def voigt_fit_func(lam, lambda0, Ti, ne):
        return voigt_profile(lam, lambda0, Ti, mi, ne)

    # Initial guess
    p0 = [lambda0_guess, Ti_guess, ne_guess]

    # Bounds
    bounds = ([lambda0_guess - 1e-10, 0.1, 1e16],
              [lambda0_guess + 1e-10, 1000, 1e21])

    # Fit
    try:
        popt, _ = curve_fit(voigt_fit_func, wavelength, intensity,
                           p0=p0, bounds=bounds, maxfev=10000)
        lambda0_fit, Ti_fit, ne_fit = popt
        return Ti_fit, ne_fit, lambda0_fit
    except:
        return None, None, None

def plot_spectral_line_analysis():
    """
    Create comprehensive visualization of spectral line analysis.
    """
    # Plasma parameters (true values)
    Ti_true = 10.0      # eV
    ne_true = 5e17      # m^-3
    mi = MP             # Hydrogen
    lambda0 = 656.3e-9  # m (H-alpha line)

    # Instrumental width
    sigma_inst = 0.05e-9  # m (50 pm resolution)

    print("=" * 70)
    print("Spectral Line Analysis: Doppler and Stark Broadening")
    print("=" * 70)
    print(f"True parameters:")
    print(f"  Ion temperature: {Ti_true:.1f} eV")
    print(f"  Electron density: {ne_true:.2e} m^-3")
    print(f"  Line: H-alpha at {lambda0*1e9:.1f} nm")
    print(f"  Instrumental resolution: {sigma_inst*1e12:.0f} pm")
    print("=" * 70)

    # Create wavelength array
    dlambda = 2e-9  # ±2 nm around line center
    wavelength = np.linspace(lambda0 - dlambda, lambda0 + dlambda, 2000)

    # Create figure
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Plot 1: Individual broadening mechanisms
    ax1 = fig.add_subplot(gs[0, :])

    # Gaussian (Doppler)
    profile_gaussian = gaussian_profile(wavelength, lambda0, Ti_true, mi)

    # Lorentzian (Stark)
    profile_lorentzian = lorentzian_profile(wavelength, lambda0, ne_true)

    # Voigt (combined)
    profile_voigt = voigt_profile(wavelength, lambda0, Ti_true, mi, ne_true)

    ax1.plot((wavelength - lambda0) * 1e12, profile_gaussian, 'b-', linewidth=2,
            label='Doppler (Gaussian)')
    ax1.plot((wavelength - lambda0) * 1e12, profile_lorentzian, 'r-', linewidth=2,
            label='Stark (Lorentzian)')
    ax1.plot((wavelength - lambda0) * 1e12, profile_voigt, 'g-', linewidth=2.5,
            label='Voigt (convolution)')

    ax1.set_xlabel('Wavelength - λ₀ (pm)', fontsize=12)
    ax1.set_ylabel('Normalized Intensity', fontsize=12)
    ax1.set_title('Line Broadening Mechanisms', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xlim([-2000, 2000])

    # Plot 2: Effect of temperature on Doppler width
    ax2 = fig.add_subplot(gs[1, 0])

    Ti_values = [1, 5, 10, 50]  # eV
    colors_T = plt.cm.Reds(np.linspace(0.3, 1, len(Ti_values)))

    for Ti, color in zip(Ti_values, colors_T):
        profile = gaussian_profile(wavelength, lambda0, Ti, mi)
        ax2.plot((wavelength - lambda0) * 1e12, profile, color=color,
                linewidth=2, label=f'Ti = {Ti} eV')

    ax2.set_xlabel('Wavelength - λ₀ (pm)', fontsize=11)
    ax2.set_ylabel('Normalized Intensity', fontsize=11)
    ax2.set_title('Doppler Broadening vs Temperature',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim([-1000, 1000])

    # Plot 3: Effect of density on Stark width
    ax3 = fig.add_subplot(gs[1, 1])

    ne_values = [1e17, 5e17, 1e18, 5e18]  # m^-3
    colors_n = plt.cm.Blues(np.linspace(0.3, 1, len(ne_values)))

    for ne, color in zip(ne_values, colors_n):
        profile = lorentzian_profile(wavelength, lambda0, ne)
        ax3.plot((wavelength - lambda0) * 1e12, profile, color=color,
                linewidth=2, label=f'ne = {ne:.1e} m⁻³')

    ax3.set_xlabel('Wavelength - λ₀ (pm)', fontsize=11)
    ax3.set_ylabel('Normalized Intensity', fontsize=11)
    ax3.set_title('Stark Broadening vs Density', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xlim([-500, 500])

    # Plot 4: Measured spectrum with noise and fitting
    ax4 = fig.add_subplot(gs[2, 0])

    # Generate synthetic measured spectrum
    profile_measured = voigt_profile(wavelength, lambda0, Ti_true, mi, ne_true)

    # Apply instrumental broadening
    profile_measured = instrument_broadening(wavelength, profile_measured, sigma_inst)

    # Add noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, size=profile_measured.shape)
    profile_measured_noisy = profile_measured + noise

    # Plot measured data
    ax4.plot((wavelength - lambda0) * 1e12, profile_measured_noisy, 'k.',
            markersize=2, alpha=0.5, label='Measured (with noise)')

    # Fit Voigt profile
    Ti_fit, ne_fit, lambda0_fit = fit_voigt_profile(
        wavelength, profile_measured_noisy, lambda0, Ti_true, mi, ne_true)

    if Ti_fit is not None:
        profile_fit = voigt_profile(wavelength, lambda0_fit, Ti_fit, mi, ne_fit)

        ax4.plot((wavelength - lambda0) * 1e12, profile_fit, 'r-', linewidth=2,
                label=f'Fit: Ti={Ti_fit:.1f} eV, ne={ne_fit:.2e}')

        print(f"\nFitting results:")
        print(f"  Ti (fitted): {Ti_fit:.2f} eV (true: {Ti_true:.1f} eV)")
        print(f"  ne (fitted): {ne_fit:.2e} m^-3 (true: {ne_true:.2e} m^-3)")
        print(f"  λ0 (fitted): {lambda0_fit*1e9:.4f} nm (true: {lambda0*1e9:.4f} nm)")

        Ti_error = abs(Ti_fit - Ti_true) / Ti_true * 100
        ne_error = abs(ne_fit - ne_true) / ne_true * 100

        print(f"\nFitting errors:")
        print(f"  Ti error: {Ti_error:.1f}%")
        print(f"  ne error: {ne_error:.1f}%")

    ax4.set_xlabel('Wavelength - λ₀ (pm)', fontsize=11)
    ax4.set_ylabel('Normalized Intensity', fontsize=11)
    ax4.set_title('Voigt Profile Fitting', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_xlim([-1000, 1000])

    # Plot 5: Width extraction method
    ax5 = fig.add_subplot(gs[2, 1])

    # Plot FWHM measurement
    profile_clean = voigt_profile(wavelength, lambda0, Ti_true, mi, ne_true)

    # Find FWHM
    half_max = profile_clean.max() / 2
    idx_half = np.where(profile_clean >= half_max)[0]
    lambda_left = wavelength[idx_half[0]]
    lambda_right = wavelength[idx_half[-1]]
    fwhm = lambda_right - lambda_left

    ax5.plot((wavelength - lambda0) * 1e12, profile_clean, 'b-', linewidth=2)
    ax5.axhline(y=half_max, color='r', linestyle='--', linewidth=1,
                label=f'FWHM = {fwhm*1e12:.1f} pm')
    ax5.axvline(x=(lambda_left - lambda0) * 1e12, color='r', linestyle=':', linewidth=1)
    ax5.axvline(x=(lambda_right - lambda0) * 1e12, color='r', linestyle=':', linewidth=1)

    # Mark FWHM points
    ax5.plot([(lambda_left - lambda0) * 1e12, (lambda_right - lambda0) * 1e12],
            [half_max, half_max], 'ro', markersize=8)

    # Theoretical Doppler FWHM
    Ti_joule = Ti_true * QE
    sigma_D = (lambda0 / C) * np.sqrt(2 * Ti_joule / mi)
    fwhm_doppler = 2 * np.sqrt(2 * np.log(2)) * sigma_D

    ax5.text(0.05, 0.95, f'FWHM (measured): {fwhm*1e12:.1f} pm\n'
                        f'FWHM (Doppler only): {fwhm_doppler*1e12:.1f} pm\n'
                        f'Broadening ratio: {fwhm/fwhm_doppler:.2f}',
            transform=ax5.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

    ax5.set_xlabel('Wavelength - λ₀ (pm)', fontsize=11)
    ax5.set_ylabel('Normalized Intensity', fontsize=11)
    ax5.set_title('FWHM Measurement', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10)
    ax5.set_xlim([-800, 800])

    plt.suptitle('Spectral Line Analysis: Temperature and Density Diagnostics',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('spectral_line_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'spectral_line_analysis.png'")
    print("=" * 70)

    plt.show()

if __name__ == "__main__":
    plot_spectral_line_analysis()
