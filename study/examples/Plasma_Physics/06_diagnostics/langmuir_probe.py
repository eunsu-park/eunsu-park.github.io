#!/usr/bin/env python3
"""
Langmuir Probe I-V Characteristic Analysis

This script generates and analyzes synthetic Langmuir probe I-V curves
to extract plasma parameters (Te, ne, Vp).

Key Physics:
- Ion saturation: I = Isat (V << Vp)
- Electron retardation: I ∝ exp(eV/kTe) (V < Vp)
- Electron saturation: I → Iesat (V > Vp)

Probe configurations:
- Single probe
- Double probe
- Triple probe

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

# Physical constants
QE = 1.602176634e-19    # C
ME = 9.10938356e-31     # kg
MP = 1.672621898e-27    # kg
KB = 1.380649e-23       # J/K

def ion_saturation_current(ne, Te, A_probe, mi=MP):
    """
    Compute ion saturation current.

    Isat ≈ 0.61 * ne * e * A * sqrt(kTe/mi)

    Parameters:
    -----------
    ne : float
        Electron density [m^-3]
    Te : float
        Electron temperature [eV]
    A_probe : float
        Probe area [m^2]
    mi : float
        Ion mass [kg]

    Returns:
    --------
    Isat : float
        Ion saturation current [A]
    """
    Te_joule = Te * QE
    cs = np.sqrt(Te_joule / mi)  # Bohm velocity
    Isat = 0.61 * ne * QE * A_probe * cs
    return Isat

def electron_saturation_current(ne, Te, A_probe):
    """
    Compute electron saturation current.

    Iesat = ne * e * A * sqrt(kTe/(2πme))

    Parameters:
    -----------
    ne : float
        Electron density [m^-3]
    Te : float
        Electron temperature [eV]
    A_probe : float
        Probe area [m^2]

    Returns:
    --------
    Iesat : float
        Electron saturation current [A]
    """
    Te_joule = Te * QE
    vth_e = np.sqrt(8 * Te_joule / (np.pi * ME))
    Iesat = ne * QE * A_probe * vth_e / 4
    return Iesat

def langmuir_probe_iv(V, ne, Te, Vp, A_probe, mi=MP):
    """
    Generate idealized Langmuir probe I-V characteristic.

    Parameters:
    -----------
    V : array
        Probe voltage [V]
    ne : float
        Electron density [m^-3]
    Te : float
        Electron temperature [eV]
    Vp : float
        Plasma potential [V]
    A_probe : float
        Probe area [m^2]
    mi : float
        Ion mass [kg]

    Returns:
    --------
    I : array
        Probe current [A]
    """
    Isat = ion_saturation_current(ne, Te, A_probe, mi)
    Iesat = electron_saturation_current(ne, Te, A_probe)

    I = np.zeros_like(V)

    for i, v in enumerate(V):
        if v < Vp - 5 * Te:  # Deep in ion saturation
            I[i] = -Isat
        elif v < Vp:  # Electron retardation region
            I[i] = -Isat + Iesat * np.exp(QE * (v - Vp) / (Te * QE))
        else:  # Electron saturation
            I[i] = -Isat + Iesat * (1 + 0.1 * (v - Vp) / Te)

    return I

def add_noise(I, noise_level=0.02):
    """Add Gaussian noise to current."""
    noise = np.random.normal(0, noise_level * np.abs(I).max(), size=I.shape)
    return I + noise

def fit_electron_temperature(V, I, Vp_guess):
    """
    Fit electron temperature from electron retardation region.

    ln(Ie) = ln(I0) + e·V/(kTe)

    Parameters:
    -----------
    V : array
        Voltage [V]
    I : array
        Current [A]
    Vp_guess : float
        Initial guess for plasma potential [V]

    Returns:
    --------
    Te_fit : float
        Fitted temperature [eV]
    Vp_fit : float
        Fitted plasma potential [V]
    """
    # Select retardation region (where current is positive and increasing)
    mask = (V < Vp_guess) & (V > Vp_guess - 20) & (I > 0)

    if mask.sum() < 5:
        return None, None

    V_fit = V[mask]
    I_fit = I[mask]

    # Linear fit to ln(I) vs V
    ln_I = np.log(I_fit)

    # Remove any infinities or NaNs
    valid = np.isfinite(ln_I)
    V_fit = V_fit[valid]
    ln_I = ln_I[valid]

    if len(V_fit) < 3:
        return None, None

    # Fit: ln(I) = a + b*V, where b = e/(kTe)
    coeffs = np.polyfit(V_fit, ln_I, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    # Extract Te (in eV)
    Te_fit = 1.0 / slope  # Already in eV since slope = 1/Te

    # Extract Vp from where the fitted line crosses zero current
    # Actually, find floating potential (where I=0)
    Vf = -intercept / slope

    # Plasma potential is a few Te above floating potential
    Vp_fit = Vf + 3 * Te_fit

    return Te_fit, Vp_fit

def plot_langmuir_probe():
    """
    Create comprehensive visualization of Langmuir probe analysis.
    """
    # True plasma parameters
    ne_true = 1e17  # m^-3
    Te_true = 3.0   # eV
    Ti_true = 0.3   # eV
    Vp_true = 10.0  # V
    A_probe = 1e-4  # m^2 (1 cm^2)

    print("=" * 70)
    print("Langmuir Probe I-V Characteristic Analysis")
    print("=" * 70)
    print("True plasma parameters:")
    print(f"  Electron density: {ne_true:.2e} m^-3")
    print(f"  Electron temperature: {Te_true:.2f} eV")
    print(f"  Plasma potential: {Vp_true:.2f} V")
    print(f"  Probe area: {A_probe*1e4:.2f} cm^2")
    print("=" * 70)

    # Generate voltage sweep
    V = np.linspace(-30, 30, 500)

    # Generate ideal I-V curve
    I_ideal = langmuir_probe_iv(V, ne_true, Te_true, Vp_true, A_probe)

    # Add noise
    np.random.seed(42)
    I_noisy = add_noise(I_ideal, noise_level=0.05)

    # Create figure
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Plot 1: I-V characteristic
    ax1 = fig.add_subplot(gs[0, :])

    ax1.plot(V, I_ideal * 1e3, 'b-', linewidth=2, label='Ideal')
    ax1.plot(V, I_noisy * 1e3, 'r.', markersize=3, alpha=0.5, label='With noise')

    # Mark important points
    ax1.axvline(x=Vp_true, color='g', linestyle='--', linewidth=2,
                label=f'Plasma potential Vp = {Vp_true:.1f} V')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Mark floating potential (where I=0)
    Vf_idx = np.argmin(np.abs(I_ideal))
    Vf = V[Vf_idx]
    ax1.plot(Vf, 0, 'mo', markersize=10, label=f'Floating potential Vf = {Vf:.1f} V')

    ax1.set_xlabel('Probe Voltage (V)', fontsize=12)
    ax1.set_ylabel('Probe Current (mA)', fontsize=12)
    ax1.set_title('Langmuir Probe I-V Characteristic',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='upper left')

    # Annotate regions
    ax1.text(-20, -5, 'Ion\nSaturation', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax1.text(5, 10, 'Electron\nRetardation', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.5))
    ax1.text(25, 30, 'Electron\nSaturation', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))

    # Plot 2: Semi-log plot for temperature fitting
    ax2 = fig.add_subplot(gs[1, 0])

    # Plot ln(I) vs V in retardation region
    mask_retard = (V > Vf) & (V < Vp_true) & (I_noisy > 0)
    V_retard = V[mask_retard]
    I_retard = I_noisy[mask_retard]

    ax2.semilogy(V, np.abs(I_noisy) * 1e3, 'r.', markersize=3, alpha=0.5,
                label='Data')

    # Fit temperature
    Te_fit, Vp_fit = fit_electron_temperature(V, I_noisy, Vp_true + 5)

    if Te_fit is not None:
        # Plot fitted line
        V_fit_range = np.linspace(Vf, Vp_true, 100)
        I_fit_line = np.exp((V_fit_range - (Vp_fit - 3*Te_fit)) / Te_fit) * 1e-3

        ax2.semilogy(V_fit_range, I_fit_line * 1e3, 'b-', linewidth=2,
                    label=f'Fit: Te = {Te_fit:.2f} eV')

        print(f"\nFitted parameters:")
        print(f"  Te (fitted): {Te_fit:.2f} eV (true: {Te_true:.2f} eV)")
        print(f"  Vp (fitted): {Vp_fit:.2f} V (true: {Vp_true:.2f} V)")

    ax2.set_xlabel('Probe Voltage (V)', fontsize=12)
    ax2.set_ylabel('|Current| (mA)', fontsize=12)
    ax2.set_title('Semi-log Plot for Te Extraction',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)
    ax2.set_xlim([Vf - 5, Vp_true + 5])

    # Plot 3: First derivative (dI/dV)
    ax3 = fig.add_subplot(gs[1, 1])

    dI_dV = np.gradient(I_noisy, V)

    ax3.plot(V, dI_dV * 1e3, 'b-', linewidth=1.5)
    ax3.axvline(x=Vp_true, color='g', linestyle='--', linewidth=2,
                label='True Vp')

    # Find maximum of derivative (plasma potential estimate)
    Vp_derivative = V[np.argmax(dI_dV)]
    ax3.axvline(x=Vp_derivative, color='r', linestyle='--', linewidth=2,
                label=f'Vp from d²I/dV² = {Vp_derivative:.1f} V')

    ax3.set_xlabel('Probe Voltage (V)', fontsize=12)
    ax3.set_ylabel('dI/dV (mA/V)', fontsize=12)
    ax3.set_title('First Derivative (Plasma Potential from Peak)',
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # Plot 4: Double probe configuration
    ax4 = fig.add_subplot(gs[2, 0])

    # Double probe: symmetric, no electron saturation
    # Current limited by smaller of two ion saturation currents
    I_double = np.zeros_like(V)
    Isat = ion_saturation_current(ne_true, Te_true, A_probe)

    for i, v in enumerate(V):
        if v < -10:
            I_double[i] = -Isat
        elif v > 10:
            I_double[i] = Isat
        else:
            # Transition region
            I_double[i] = Isat * np.tanh(v / (2 * Te_true))

    I_double_noisy = add_noise(I_double, noise_level=0.05)

    ax4.plot(V, I_double_noisy * 1e3, 'b-', linewidth=1.5)
    ax4.axhline(y=Isat * 1e3, color='r', linestyle='--', linewidth=2,
                label=f'Ion saturation ±{Isat*1e3:.2f} mA')
    ax4.axhline(y=-Isat * 1e3, color='r', linestyle='--', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    ax4.set_xlabel('Voltage Difference V1-V2 (V)', fontsize=12)
    ax4.set_ylabel('Current (mA)', fontsize=12)
    ax4.set_title('Double Probe Configuration',
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    # Plot 5: Triple probe schematic and analysis
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # Triple probe explanation
    triple_text = """
    Triple Probe Configuration

    Three electrodes:
    • Probe 1: Floating (I₁ = 0)
    • Probe 2: Floating (I₂ = 0)
    • Probe 3: Draws current (I₃ = -I₁ - I₂)

    Advantages:
    ✓ No time-varying bias needed
    ✓ Fast measurement (~μs)
    ✓ Good for fluctuations

    Formulas:
    Te = (V₁ - V₂) / ln[(I₃+ - I₃⁺)/(I₃⁺ - I₃⁻)]
    ne ∝ I_sat / sqrt(Te)

    Limitations:
    ✗ Assumes Maxwellian distribution
    ✗ Requires calibration
    ✗ More complex geometry
    """

    ax5.text(0.1, 0.95, triple_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Add simple schematic
    from matplotlib.patches import Circle, FancyArrowPatch

    # Draw three probes
    probe_y = [0.4, 0.35, 0.3]
    probe_x = [0.7, 0.75, 0.8]

    for i, (x, y) in enumerate(zip(probe_x, probe_y)):
        circle = Circle((x, y), 0.02, transform=ax5.transAxes,
                       facecolor='red', edgecolor='black', linewidth=2)
        ax5.add_patch(circle)
        ax5.text(x, y - 0.05, f'P{i+1}', transform=ax5.transAxes,
                ha='center', fontsize=9, fontweight='bold')

    plt.suptitle('Langmuir Probe Diagnostics: I-V Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('langmuir_probe.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'langmuir_probe.png'")

    # Error analysis
    if Te_fit is not None:
        Te_error = abs(Te_fit - Te_true) / Te_true * 100
        Vp_error = abs(Vp_fit - Vp_true) / Vp_true * 100

        print(f"\nFitting errors:")
        print(f"  Te error: {Te_error:.1f}%")
        print(f"  Vp error: {Vp_error:.1f}%")

    print("=" * 70)

    plt.show()

if __name__ == "__main__":
    plot_langmuir_probe()
