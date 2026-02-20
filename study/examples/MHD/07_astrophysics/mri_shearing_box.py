#!/usr/bin/env python3
"""
Magnetorotational Instability (MRI) - Shearing Box Analysis

This module analyzes the magnetorotational instability (MRI), which is crucial
for understanding angular momentum transport in accretion disks around compact
objects (black holes, neutron stars, young stars).

Key physics:
- MRI destabilizes otherwise stable Keplerian disks when threaded by
  weak magnetic fields
- Growth rate: γ_max ≈ (3/4)Ω for ideal MHD
- Requires weak vertical field: B_z > B_crit
- Resistivity provides a cutoff at small scales

The analysis uses local shearing box approximation with dispersion relation.

Author: MHD Course Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


class MRIShearingBox:
    """
    MRI dispersion relation solver in shearing box approximation.

    Attributes:
        Omega (float): Angular velocity at fiducial radius
        q (float): Shear parameter (q=3/2 for Keplerian)
        B_z (float): Vertical magnetic field
        rho (float): Density
        eta (float): Magnetic diffusivity
        nu (float): Kinematic viscosity
    """

    def __init__(self, Omega=1.0, q=1.5, B_z=0.1, rho=1.0,
                 eta=0.0, nu=0.0):
        """
        Initialize MRI shearing box model.

        Parameters:
            Omega (float): Angular velocity
            q (float): Shear parameter (q=3/2 for Keplerian)
            B_z (float): Vertical field strength
            rho (float): Density
            eta (float): Magnetic diffusivity
            nu (float): Kinematic viscosity
        """
        self.Omega = Omega
        self.q = q
        self.B_z = B_z
        self.rho = rho
        self.eta = eta
        self.nu = nu

        # Derived quantities
        self.kappa2 = 2 * Omega**2 * (2 - q)  # Epicyclic frequency squared
        self.v_A = B_z / np.sqrt(rho)  # Alfvén velocity

        # Magnetic Prandtl number
        self.Pm = nu / eta if eta > 0 else np.inf

    def dispersion_relation_ideal(self, gamma, k_z):
        """
        Ideal MHD dispersion relation for MRI.

        (γ² + ν²_A k_z²)(γ² + κ²) - 4Ω²ν²_A k_z² = 0

        Parameters:
            gamma (complex): Growth rate
            k_z (float): Vertical wavenumber

        Returns:
            complex: Residual
        """
        kappa2 = self.kappa2
        Omega = self.Omega
        v_A = self.v_A

        term1 = gamma**2 + v_A**2 * k_z**2
        term2 = gamma**2 + kappa2
        coupling = 4 * Omega**2 * v_A**2 * k_z**2

        return term1 * term2 - coupling

    def dispersion_relation_resistive(self, gamma, k_z):
        """
        Resistive MHD dispersion relation.

        Includes magnetic diffusion: adds -iηk² terms

        Parameters:
            gamma (complex): Growth rate
            k_z (float): Vertical wavenumber

        Returns:
            complex: Residual
        """
        if self.eta == 0:
            return self.dispersion_relation_ideal(gamma, k_z)

        kappa2 = self.kappa2
        Omega = self.Omega
        v_A = self.v_A
        eta = self.eta

        # Modified dispersion relation with resistivity
        gamma_eff = gamma + 1j * eta * k_z**2

        term1 = gamma_eff**2 + v_A**2 * k_z**2
        term2 = gamma_eff**2 + kappa2
        coupling = 4 * Omega**2 * v_A**2 * k_z**2

        return term1 * term2 - coupling

    def find_growth_rate_ideal(self, k_z):
        """
        Find growth rate for given k_z (ideal MHD).

        Parameters:
            k_z (float): Vertical wavenumber

        Returns:
            float: Maximum growth rate (real part)
        """
        # Analytical solution exists for ideal MRI
        kappa2 = self.kappa2
        Omega = self.Omega
        v_A = self.v_A

        # MRI growth rate formula
        omega_A2 = v_A**2 * k_z**2

        discriminant = omega_A2 + kappa2
        gamma2 = 0.5 * (omega_A2 + kappa2 -
                       np.sqrt((omega_A2 + kappa2)**2 - 16 * Omega**2 * omega_A2))

        if gamma2 > 0:
            return np.sqrt(gamma2)
        else:
            return 0.0

    def find_growth_rate_resistive(self, k_z):
        """
        Find growth rate for given k_z (resistive MHD).

        Uses numerical root finding.

        Parameters:
            k_z (float): Vertical wavenumber

        Returns:
            float: Maximum growth rate
        """
        # Initial guess from ideal solution
        gamma_ideal = self.find_growth_rate_ideal(k_z)

        # Solve dispersion relation numerically
        def residual(gamma):
            return abs(self.dispersion_relation_resistive(gamma, k_z))

        # Search for maximum growth rate
        gamma_range = np.linspace(0, gamma_ideal * 1.5, 100)
        residuals = [residual(g) for g in gamma_range]
        idx_min = np.argmin(residuals)

        # Refine
        try:
            gamma_opt = fsolve(residual, gamma_range[idx_min])[0]
            return max(gamma_opt, 0.0)
        except:
            return 0.0

    def compute_growth_rate_spectrum(self, k_z_array):
        """
        Compute growth rate as function of k_z.

        Parameters:
            k_z_array (ndarray): Array of wavenumbers

        Returns:
            ndarray: Growth rates
        """
        gamma_array = np.zeros_like(k_z_array)

        for i, k_z in enumerate(k_z_array):
            if self.eta == 0:
                gamma_array[i] = self.find_growth_rate_ideal(k_z)
            else:
                gamma_array[i] = self.find_growth_rate_resistive(k_z)

        return gamma_array

    def critical_wavenumber(self):
        """
        Compute critical wavenumber below which MRI is stable.

        Returns:
            float: k_crit
        """
        # For MRI, need B_z > 0 and k_z > 0
        # Critical condition from dispersion relation
        v_A = self.v_A
        Omega = self.Omega

        if v_A == 0:
            return np.inf

        # Rough estimate
        k_crit = Omega / v_A

        return k_crit

    def plot_growth_rate_vs_k(self):
        """
        Plot growth rate spectrum.
        """
        k_z_array = np.logspace(-2, 2, 200) * self.Omega / self.v_A
        gamma_array = self.compute_growth_rate_spectrum(k_z_array)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(k_z_array * self.v_A / self.Omega, gamma_array / self.Omega,
               'b-', linewidth=2, label='MRI growth rate')

        # Maximum growth rate
        gamma_max = np.max(gamma_array)
        k_max = k_z_array[np.argmax(gamma_array)]

        ax.axhline(y=gamma_max / self.Omega, color='r', linestyle='--',
                  alpha=0.5, label=f'γ_max/Ω = {gamma_max/self.Omega:.3f}')
        ax.axvline(x=k_max * self.v_A / self.Omega, color='g', linestyle='--',
                  alpha=0.5, label=f'k_max v_A/Ω = {k_max*self.v_A/self.Omega:.2f}')

        ax.set_xlabel(r'$k_z v_A / \Omega$', fontsize=12)
        ax.set_ylabel(r'$\gamma / \Omega$', fontsize=12)
        ax.set_title(f'MRI Growth Rate Spectrum (q={self.q}, Pm={self.Pm:.1e})',
                    fontsize=14)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        # Add theoretical max for ideal MRI
        if self.eta == 0:
            ax.axhline(y=0.75, color='orange', linestyle=':', linewidth=2,
                      alpha=0.6, label='Ideal limit: 3Ω/4')
            ax.legend(fontsize=11)

        plt.tight_layout()
        return fig

    def plot_stability_boundary(self):
        """
        Plot stability boundary in (k, Pm) space.
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        # Array of Pm values
        Pm_array = np.logspace(-2, 2, 50)
        k_array = np.logspace(-2, 2, 100) * self.Omega / self.v_A

        # Compute growth rate for each (k, Pm)
        gamma_grid = np.zeros((len(Pm_array), len(k_array)))

        for i, Pm in enumerate(Pm_array):
            # Set resistivity
            eta_temp = self.nu / Pm if Pm > 0 else 0
            eta_original = self.eta
            self.eta = eta_temp

            for j, k_z in enumerate(k_array):
                gamma_grid[i, j] = self.find_growth_rate_resistive(k_z)

            # Restore
            self.eta = eta_original

        # Normalize
        gamma_grid /= self.Omega

        # Contour plot
        K, P = np.meshgrid(k_array * self.v_A / self.Omega, Pm_array)
        levels = np.linspace(0, 0.75, 16)
        contour = ax.contourf(K, P, gamma_grid, levels=levels, cmap='viridis')
        plt.colorbar(contour, ax=ax, label=r'$\gamma / \Omega$')

        # Stability boundary (γ = 0)
        ax.contour(K, P, gamma_grid, levels=[0], colors='red',
                  linewidths=2, linestyles='--')

        ax.set_xlabel(r'$k_z v_A / \Omega$', fontsize=12)
        ax.set_ylabel(r'Magnetic Prandtl Number $Pm$', fontsize=12)
        ax.set_title('MRI Stability Boundary', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def main():
    """
    Main function demonstrating MRI analysis.
    """
    print("=" * 60)
    print("Magnetorotational Instability (MRI) Analysis")
    print("=" * 60)

    # Create MRI model - ideal case
    print("\n1. Ideal MHD (no resistivity):")
    mri_ideal = MRIShearingBox(
        Omega=1.0,
        q=1.5,      # Keplerian
        B_z=0.1,
        rho=1.0,
        eta=0.0,
        nu=0.0
    )

    print(f"   Alfvén velocity: {mri_ideal.v_A:.3f}")
    print(f"   Epicyclic frequency: κ = {np.sqrt(mri_ideal.kappa2):.3f}")

    # Resistive case
    print("\n2. Resistive MHD:")
    mri_resistive = MRIShearingBox(
        Omega=1.0,
        q=1.5,
        B_z=0.1,
        rho=1.0,
        eta=0.01,
        nu=0.01
    )
    print(f"   Magnetic Prandtl number: Pm = {mri_resistive.Pm:.2f}")

    # Plot growth rate spectrum
    fig1 = mri_ideal.plot_growth_rate_vs_k()

    # Plot stability boundary
    fig2 = mri_ideal.plot_stability_boundary()

    plt.savefig('/tmp/mri_shearing_box.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to /tmp/mri_shearing_box.png")

    plt.show()


if __name__ == "__main__":
    main()
