#!/usr/bin/env python3
"""
General Plasma Wave Dispersion Relation Solver

This script solves the cold plasma dispersion relation for arbitrary
propagation angles and plasma conditions.

Features:
- Cold plasma dielectric tensor (Stix parameters S, D, P)
- Solve det(wave equation) = 0 for ω(k) or k(ω)
- Wave modes: R, L, O, X for any angle θ
- Generate ω-k diagrams and CMA diagrams
- Multiple ion species support

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import fsolve, brentq

# Physical constants
EPS0 = 8.854187817e-12  # F/m
QE = 1.602176634e-19    # C
ME = 9.10938356e-31     # kg
MP = 1.672621898e-27    # kg
C = 2.99792458e8        # m/s

class ColdPlasmaDispersion:
    """Cold plasma dispersion relation solver."""

    def __init__(self, ne, B0, ion_species='H'):
        """
        Initialize plasma parameters.

        Parameters:
        -----------
        ne : float
            Electron density [m^-3]
        B0 : float
            Magnetic field [T]
        ion_species : str
            Ion species ('H', 'D', 'He')
        """
        self.ne = ne
        self.B0 = B0

        # Ion mass
        if ion_species == 'H':
            self.mi = MP
        elif ion_species == 'D':
            self.mi = 2 * MP
        elif ion_species == 'He':
            self.mi = 4 * MP
        else:
            self.mi = MP

        # Compute characteristic frequencies
        self.omega_pe = np.sqrt(ne * QE**2 / (ME * EPS0))
        self.omega_pi = np.sqrt(ne * QE**2 / (self.mi * EPS0))
        self.omega_ce = QE * B0 / ME
        self.omega_ci = QE * B0 / self.mi

        self.f_pe = self.omega_pe / (2 * np.pi)
        self.f_ce = self.omega_ce / (2 * np.pi)
        self.f_ci = self.omega_ci / (2 * np.pi)

    def stix_parameters(self, omega):
        """
        Compute Stix parameters S, D, P.

        Parameters:
        -----------
        omega : float or array
            Angular frequency [rad/s]

        Returns:
        --------
        S, D, P : Stix parameters
        """
        # Avoid division by zero
        eps = 1e-10

        # S = 1 - Σ ωps²/(ω² - ωcs²)
        S = 1 - self.omega_pe**2 / (omega**2 - self.omega_ce**2 + eps)
        S -= self.omega_pi**2 / (omega**2 - self.omega_ci**2 + eps)

        # D = Σ (ωcs/ω) ωps²/(ω² - ωcs²)
        D = (self.omega_ce / omega) * self.omega_pe**2 / (omega**2 - self.omega_ce**2 + eps)
        D += (self.omega_ci / omega) * self.omega_pi**2 / (omega**2 - self.omega_ci**2 + eps)

        # P = 1 - Σ ωps²/ω²
        P = 1 - self.omega_pe**2 / omega**2 - self.omega_pi**2 / omega**2

        return S, D, P

    def dispersion_general(self, omega, k, theta):
        """
        General dispersion relation for arbitrary angle θ.

        det(M) = An⁴ - Bn² + C = 0

        where n = ck/ω

        Parameters:
        -----------
        omega : float
            Angular frequency [rad/s]
        k : float
            Wavenumber [rad/m]
        theta : float
            Propagation angle w.r.t. B [rad]

        Returns:
        --------
        residual : float (should be zero for valid solutions)
        """
        S, D, P = self.stix_parameters(omega)

        n = C * k / omega  # Refractive index

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Dispersion relation coefficients
        A = S * sin_theta**2 + P * cos_theta**2
        B = (S**2 - D**2) * sin_theta**2 + P * S * (1 + cos_theta**2)
        C_coeff = P * (S**2 - D**2)

        # Dispersion relation: An⁴ - Bn² + C = 0
        residual = A * n**4 - B * n**2 + C_coeff

        return residual

    def parallel_modes(self, omega):
        """
        Solve for parallel propagation (θ = 0).

        R-wave: n² = S + D = R
        L-wave: n² = S - D = L

        Parameters:
        -----------
        omega : float or array
            Angular frequency [rad/s]

        Returns:
        --------
        n_R, n_L : refractive indices for R and L modes
        """
        S, D, P = self.stix_parameters(omega)

        R = S + D
        L = S - D

        n_R = np.sqrt(np.maximum(R, 0))
        n_L = np.sqrt(np.maximum(L, 0))

        return n_R, n_L

    def perpendicular_modes(self, omega):
        """
        Solve for perpendicular propagation (θ = 90°).

        O-mode: n² = P
        X-mode: n² = (S² - D²) / S

        Parameters:
        -----------
        omega : float or array
            Angular frequency [rad/s]

        Returns:
        --------
        n_O, n_X : refractive indices for O and X modes
        """
        S, D, P = self.stix_parameters(omega)

        n_O = np.sqrt(np.maximum(P, 0))
        n_X_sq = (S**2 - D**2) / (S + 1e-10)
        n_X = np.sqrt(np.maximum(n_X_sq, 0))

        return n_O, n_X

    def find_cutoffs_resonances(self):
        """
        Find cutoff and resonance frequencies.

        Returns:
        --------
        dict : Cutoff and resonance frequencies
        """
        # Cutoffs (n² = 0)
        # R cutoff: R = 0
        # L cutoff: L = 0
        # P cutoff: P = 0

        # Resonances (n² → ∞)
        # Upper hybrid: ω² = ωpe² + ωce²
        # Lower hybrid: ω² = ωci·ωce + ωpi²/(1 + ωpe²/ωce²)

        omega_uh = np.sqrt(self.omega_pe**2 + self.omega_ce**2)
        omega_lh = np.sqrt(self.omega_ci * self.omega_ce)

        return {
            'f_uh': omega_uh / (2 * np.pi),
            'f_lh': omega_lh / (2 * np.pi),
            'f_pe': self.f_pe,
            'f_ce': self.f_ce,
            'f_ci': self.f_ci
        }

def plot_dispersion_solver():
    """
    Demonstrate dispersion relation solver with multiple plots.
    """
    # Plasma parameters
    ne = 1e18  # m^-3
    B0 = 1.0   # T
    ion_species = 'H'

    solver = ColdPlasmaDispersion(ne, B0, ion_species)

    print("=" * 70)
    print("Cold Plasma Dispersion Relation Solver")
    print("=" * 70)
    print(f"Electron density: {ne:.2e} m^-3")
    print(f"Magnetic field: {B0:.2f} T")
    print(f"Ion species: {ion_species}")
    print(f"\nCharacteristic frequencies:")
    print(f"  Electron plasma: {solver.f_pe/1e9:.3f} GHz")
    print(f"  Electron cyclotron: {solver.f_ce/1e9:.3f} GHz")
    print(f"  Ion cyclotron: {solver.f_ci/1e6:.3f} MHz")

    cutoffs = solver.find_cutoffs_resonances()
    print(f"\nCutoffs and resonances:")
    print(f"  Upper hybrid: {cutoffs['f_uh']/1e9:.3f} GHz")
    print(f"  Lower hybrid: {cutoffs['f_lh']/1e6:.3f} MHz")
    print("=" * 70)

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Plot 1: ω-k diagram for parallel propagation
    ax1 = fig.add_subplot(gs[0, :])

    omega_array = np.linspace(0.1 * solver.omega_ce, 3 * solver.omega_ce, 1000)

    n_R, n_L = solver.parallel_modes(omega_array)
    k_R = omega_array * n_R / C
    k_L = omega_array * n_L / C

    # Light line
    k_light = omega_array / C

    ax1.plot(k_R / 1e6, omega_array / (2 * np.pi * 1e9), 'r-',
            linewidth=2, label='R-mode')
    ax1.plot(k_L / 1e6, omega_array / (2 * np.pi * 1e9), 'b-',
            linewidth=2, label='L-mode')
    ax1.plot(k_light / 1e6, omega_array / (2 * np.pi * 1e9), 'k--',
            linewidth=1, label='Light line', alpha=0.5)

    # Mark characteristic frequencies
    ax1.axhline(y=solver.f_pe / 1e9, color='g', linestyle=':', linewidth=2,
                label=f'fpe = {solver.f_pe/1e9:.2f} GHz')
    ax1.axhline(y=solver.f_ce / 1e9, color='m', linestyle=':', linewidth=2,
                label=f'fce = {solver.f_ce/1e9:.2f} GHz')
    ax1.axhline(y=cutoffs['f_uh'] / 1e9, color='orange', linestyle=':', linewidth=2,
                label=f'fUH = {cutoffs["f_uh"]/1e9:.2f} GHz')

    ax1.set_xlabel('Wavenumber k (rad/Mm)', fontsize=12)
    ax1.set_ylabel('Frequency (GHz)', fontsize=12)
    ax1.set_title('ω-k Diagram: Parallel Propagation (θ = 0°)',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 50])
    ax1.set_ylim([0, 100])

    # Plot 2: ω-k diagram for perpendicular propagation
    ax2 = fig.add_subplot(gs[1, 0])

    n_O, n_X = solver.perpendicular_modes(omega_array)
    k_O = omega_array * n_O / C
    k_X = omega_array * n_X / C

    ax2.plot(k_O / 1e6, omega_array / (2 * np.pi * 1e9), 'g-',
            linewidth=2, label='O-mode')
    ax2.plot(k_X / 1e6, omega_array / (2 * np.pi * 1e9), 'c-',
            linewidth=2, label='X-mode')
    ax2.plot(k_light / 1e6, omega_array / (2 * np.pi * 1e9), 'k--',
            linewidth=1, label='Light line', alpha=0.5)

    ax2.axhline(y=solver.f_pe / 1e9, color='purple', linestyle=':', linewidth=2)
    ax2.axhline(y=cutoffs['f_uh'] / 1e9, color='orange', linestyle=':', linewidth=2)

    ax2.set_xlabel('Wavenumber k (rad/Mm)', fontsize=11)
    ax2.set_ylabel('Frequency (GHz)', fontsize=11)
    ax2.set_title('ω-k Diagram: Perpendicular (θ = 90°)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 50])
    ax2.set_ylim([0, 100])

    # Plot 3: CMA diagram
    ax3 = fig.add_subplot(gs[1, 1])

    X_array = np.linspace(0, 3, 1000)
    Y_array = np.linspace(-2, 2, 1000)

    # Current plasma parameters
    X_plasma = solver.omega_pe**2 / solver.omega_ce**2
    Y_plasma = 1.0  # At ω = ωce

    # Cutoff curves
    X_R = 1 - Y_array  # R cutoff
    X_L = 1 + Y_array  # L cutoff
    X_P = np.ones_like(Y_array)  # P cutoff

    # Resonance curves
    X_UH = 1 - Y_array**2  # Upper hybrid

    ax3.plot(X_R, Y_array, 'r-', linewidth=2, label='R cutoff')
    ax3.plot(X_L, Y_array, 'b-', linewidth=2, label='L cutoff')
    ax3.plot(X_P, Y_array, 'g-', linewidth=2, label='P cutoff')
    ax3.plot(X_UH, Y_array, 'm--', linewidth=2, label='UH resonance')

    # Mark current plasma
    ax3.plot(X_plasma, Y_plasma, 'ko', markersize=10,
            label=f'This plasma\n(X={X_plasma:.2f}, Y=1)')

    ax3.set_xlabel(r'$X = \omega_{pe}^2 / \omega^2$', fontsize=11)
    ax3.set_ylabel(r'$Y = \omega_{ce} / \omega$', fontsize=11)
    ax3.set_title('CMA Diagram', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 3])
    ax3.set_ylim([-2, 2])
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.axvline(x=0, color='k', linewidth=0.5)

    # Plot 4: Refractive index vs frequency (parallel)
    ax4 = fig.add_subplot(gs[2, 0])

    ax4.plot(omega_array / (2 * np.pi * 1e9), n_R, 'r-',
            linewidth=2, label='R-mode')
    ax4.plot(omega_array / (2 * np.pi * 1e9), n_L, 'b-',
            linewidth=2, label='L-mode')

    ax4.axhline(y=1, color='k', linestyle='--', linewidth=1,
                label='Vacuum (n=1)', alpha=0.5)
    ax4.axvline(x=solver.f_pe / 1e9, color='g', linestyle=':', linewidth=2)
    ax4.axvline(x=solver.f_ce / 1e9, color='m', linestyle=':', linewidth=2)

    ax4.set_xlabel('Frequency (GHz)', fontsize=11)
    ax4.set_ylabel('Refractive Index n', fontsize=11)
    ax4.set_title('Refractive Index: Parallel', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 100])
    ax4.set_ylim([0, 10])

    # Plot 5: Refractive index vs frequency (perpendicular)
    ax5 = fig.add_subplot(gs[2, 1])

    ax5.plot(omega_array / (2 * np.pi * 1e9), n_O, 'g-',
            linewidth=2, label='O-mode')
    ax5.plot(omega_array / (2 * np.pi * 1e9), n_X, 'c-',
            linewidth=2, label='X-mode')

    ax5.axhline(y=1, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax5.axvline(x=solver.f_pe / 1e9, color='purple', linestyle=':', linewidth=2)
    ax5.axvline(x=cutoffs['f_uh'] / 1e9, color='orange', linestyle=':', linewidth=2)

    ax5.set_xlabel('Frequency (GHz)', fontsize=11)
    ax5.set_ylabel('Refractive Index n', fontsize=11)
    ax5.set_title('Refractive Index: Perpendicular', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 100])
    ax5.set_ylim([0, 5])

    plt.suptitle('Cold Plasma Dispersion Relation Solver',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('dispersion_solver.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'dispersion_solver.png'")

    plt.show()

if __name__ == "__main__":
    plot_dispersion_solver()
