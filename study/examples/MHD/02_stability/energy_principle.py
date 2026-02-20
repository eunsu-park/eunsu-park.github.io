#!/usr/bin/env python3
"""
MHD Energy Principle

Evaluates the potential energy δW for a given plasma perturbation ξ.

The energy principle states that an equilibrium is stable if δW > 0
for all perturbations ξ.

δW = δW_field + δW_compression + δW_pressure

where:
- δW_field: field line bending energy
- δW_compression: magnetic compression energy
- δW_pressure: pressure-driven destabilization

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson


class EnergyPrinciple:
    """
    Class for evaluating the MHD energy principle.

    Attributes
    ----------
    r : ndarray
        Radial grid
    """

    def __init__(self, r_max=0.01, n_points=200):
        """
        Initialize with radial grid.

        Parameters
        ----------
        r_max : float
            Maximum radius (m)
        n_points : int
            Number of radial points
        """
        self.r = np.linspace(0, r_max, n_points)
        self.r_max = r_max
        self.n_points = n_points

    def equilibrium_fields(self, B_z, I_total):
        """
        Compute equilibrium magnetic fields for a Z-pinch.

        Parameters
        ----------
        B_z : float
            Axial magnetic field (T)
        I_total : float
            Total current (A)

        Returns
        -------
        B_theta : ndarray
            Azimuthal field
        p : ndarray
            Pressure profile
        """
        mu_0 = 4 * np.pi * 1e-7

        # Current density (parabolic profile)
        J_z = 2 * I_total / (np.pi * self.r_max**2) * (1 - (self.r/self.r_max)**2)
        J_z = np.where(self.r <= self.r_max, J_z, 0.0)

        # Azimuthal field from Ampere's law
        B_theta = np.zeros_like(self.r)
        for i, ri in enumerate(self.r):
            if ri > 1e-10:
                I_enclosed = np.pi * ri**2 * 2 * I_total / (np.pi * self.r_max**2) * (1 - 0.5*(ri/self.r_max)**2)
                B_theta[i] = mu_0 * I_enclosed / (2 * np.pi * ri)

        # Pressure from force balance
        p = np.zeros_like(self.r)
        p_edge = 100.0  # Pa
        p[-1] = p_edge

        for i in range(len(self.r)-2, -1, -1):
            dr = self.r[i+1] - self.r[i]
            p[i] = p[i+1] - J_z[i] * B_theta[i] * dr

        return B_theta, p, J_z

    def perturbation_radial(self, m, amplitude=1e-4):
        """
        Generate radial displacement perturbation.

        ξ_r(r) = A * r^m * (1 - r/r_max)

        Parameters
        ----------
        m : int
            Mode number
        amplitude : float
            Perturbation amplitude (m)

        Returns
        -------
        xi_r : ndarray
            Radial displacement
        """
        xi_r = amplitude * (self.r/self.r_max)**m * (1 - self.r/self.r_max)
        return xi_r

    def field_bending_energy(self, xi_r, B_z, B_theta, m):
        """
        Compute field line bending energy.

        δW_bend = (1/2μ₀) ∫ |δB_⊥|² dV

        Parameters
        ----------
        xi_r : ndarray
            Radial displacement
        B_z : float
            Axial field
        B_theta : ndarray
            Azimuthal field
        m : int
            Mode number

        Returns
        -------
        W_bend : float
            Field bending energy (J/m)
        """
        mu_0 = 4 * np.pi * 1e-7

        # Perturbed field components (simplified)
        # δB ~ B·∇ξ
        dxi_dr = np.gradient(xi_r, self.r)

        # Perpendicular field perturbation
        delta_B_perp_sq = B_z**2 * (m * xi_r / (self.r + 1e-10))**2 + B_theta**2 * dxi_dr**2

        # Integrate over volume (per unit length in z)
        integrand = delta_B_perp_sq * self.r
        W_bend = (np.pi / mu_0) * simpson(integrand, x=self.r)

        return W_bend

    def compression_energy(self, xi_r, B_z, B_theta):
        """
        Compute magnetic compression energy.

        δW_comp = (1/2μ₀) ∫ |B|² |∇·ξ|² dV

        Parameters
        ----------
        xi_r : ndarray
            Radial displacement
        B_z : float
            Axial field
        B_theta : ndarray
            Azimuthal field

        Returns
        -------
        W_comp : float
            Compression energy (J/m)
        """
        mu_0 = 4 * np.pi * 1e-7

        # Divergence of displacement (cylindrical): ∇·ξ = (1/r) d(r*ξ_r)/dr
        div_xi = np.gradient(self.r * xi_r, self.r) / (self.r + 1e-10)

        # Total field squared
        B_squared = B_z**2 + B_theta**2

        # Integrate
        integrand = B_squared * div_xi**2 * self.r
        W_comp = (np.pi / mu_0) * simpson(integrand, x=self.r)

        return W_comp

    def pressure_energy(self, xi_r, p):
        """
        Compute pressure-driven energy (destabilizing).

        δW_p = -∫ ξ·∇p (∇·ξ) dV

        Parameters
        ----------
        xi_r : ndarray
            Radial displacement
        p : ndarray
            Pressure profile

        Returns
        -------
        W_p : float
            Pressure energy (J/m)
        """
        # Pressure gradient
        dp_dr = np.gradient(p, self.r)

        # Divergence
        div_xi = np.gradient(self.r * xi_r, self.r) / (self.r + 1e-10)

        # This term is typically destabilizing (negative)
        integrand = -xi_r * dp_dr * div_xi * self.r
        W_p = 2 * np.pi * simpson(integrand, x=self.r)

        return W_p

    def total_energy(self, xi_r, B_z, B_theta, p, m):
        """
        Compute total potential energy δW.

        Parameters
        ----------
        xi_r : ndarray
            Radial displacement
        B_z : float
            Axial field
        B_theta : ndarray
            Azimuthal field
        p : ndarray
            Pressure
        m : int
            Mode number

        Returns
        -------
        dict
            Dictionary with energy components
        """
        W_bend = self.field_bending_energy(xi_r, B_z, B_theta, m)
        W_comp = self.compression_energy(xi_r, B_z, B_theta)
        W_p = self.pressure_energy(xi_r, p)

        W_total = W_bend + W_comp + W_p

        return {
            'W_bend': W_bend,
            'W_comp': W_comp,
            'W_p': W_p,
            'W_total': W_total,
            'stable': W_total > 0
        }


def scan_mode_numbers(ep, B_z, B_theta, p, m_max=5, n_max=5):
    """
    Scan over mode numbers (m, n) and compute δW.

    Parameters
    ----------
    ep : EnergyPrinciple
        Energy principle instance
    B_z : float
        Axial field
    B_theta : ndarray
        Azimuthal field
    p : ndarray
        Pressure
    m_max, n_max : int
        Maximum mode numbers

    Returns
    -------
    W_map : ndarray
        2D array of δW values
    """
    m_values = np.arange(0, m_max + 1)
    n_values = np.arange(1, n_max + 1)

    W_map = np.zeros((len(m_values), len(n_values)))

    for i, m in enumerate(m_values):
        for j, n in enumerate(n_values):
            if m == 0:
                m_eff = 1  # Avoid m=0 singularity
            else:
                m_eff = m

            xi_r = ep.perturbation_radial(m_eff, amplitude=1e-4)
            result = ep.total_energy(xi_r, B_z, B_theta, p, m_eff)
            W_map[i, j] = result['W_total']

    return W_map, m_values, n_values


def plot_energy_decomposition(ep, B_z, B_theta, p, m=2):
    """
    Plot energy decomposition for a given mode.

    Parameters
    ----------
    ep : EnergyPrinciple
        Energy principle instance
    B_z : float
        Axial field
    B_theta : ndarray
        Azimuthal field
    p : ndarray
        Pressure
    m : int
        Mode number
    """
    xi_r = ep.perturbation_radial(m, amplitude=1e-4)
    result = ep.total_energy(xi_r, B_z, B_theta, p, m)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot perturbation
    axes[0].plot(ep.r * 100, xi_r * 1e6, 'b-', linewidth=2)
    axes[0].set_xlabel('Radius (cm)', fontsize=12)
    axes[0].set_ylabel('ξ_r (μm)', fontsize=12)
    axes[0].set_title(f'Radial Displacement (m={m})', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot energy components
    components = ['W_bend', 'W_comp', 'W_p', 'W_total']
    values = [result[key] for key in components]
    colors = ['blue', 'green', 'red', 'purple']
    labels = ['Field Bending', 'Compression', 'Pressure', 'Total']

    bars = axes[1].bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_ylabel('Energy δW (J/m)', fontsize=12)
    axes[1].set_title('Energy Decomposition', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2e}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9)

    # Stability text
    stability_text = "STABLE" if result['stable'] else "UNSTABLE"
    stability_color = "green" if result['stable'] else "red"
    axes[1].text(0.5, 0.95, stability_text,
                transform=axes[1].transAxes,
                fontsize=14, fontweight='bold',
                color=stability_color,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(f'MHD Energy Principle Analysis (m={m})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('energy_principle.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_stability_map(W_map, m_values, n_values):
    """
    Plot 2D stability map.

    Parameters
    ----------
    W_map : ndarray
        Energy values
    m_values, n_values : ndarray
        Mode numbers
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(W_map, cmap='RdBu_r', aspect='auto',
                   extent=[n_values[0]-0.5, n_values[-1]+0.5,
                          m_values[-1]+0.5, m_values[0]-0.5],
                   vmin=-np.max(np.abs(W_map)), vmax=np.max(np.abs(W_map)))

    # Contour at δW = 0
    ax.contour(n_values, m_values, W_map, levels=[0],
               colors='black', linewidths=3)

    # Labels
    ax.set_xlabel('Toroidal mode number n', fontsize=13)
    ax.set_ylabel('Poloidal mode number m', fontsize=13)
    ax.set_title('MHD Stability Map (δW > 0: stable, δW < 0: unstable)',
                 fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('δW (J/m)', fontsize=12)

    # Add text annotations for stable/unstable regions
    for i, m in enumerate(m_values):
        for j, n in enumerate(n_values):
            if W_map[i, j] > 0:
                marker = '✓'
                color = 'green'
            else:
                marker = '✗'
                color = 'red'
            ax.text(n, m, marker, ha='center', va='center',
                   color=color, fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('stability_map.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    # Initialize
    r_max = 0.01  # 1 cm
    ep = EnergyPrinciple(r_max=r_max, n_points=200)

    # Equilibrium parameters
    B_z = 0.5  # T
    I_total = 50e3  # 50 kA

    print("MHD Energy Principle Analysis")
    print("=" * 60)
    print(f"Configuration: Z-pinch")
    print(f"  Radius: {r_max*100:.1f} cm")
    print(f"  Axial field: {B_z:.2f} T")
    print(f"  Current: {I_total/1e3:.1f} kA")
    print()

    # Compute equilibrium
    B_theta, p, J_z = ep.equilibrium_fields(B_z, I_total)

    print("Equilibrium computed:")
    print(f"  Peak B_θ: {np.max(B_theta)*1e3:.2f} mT")
    print(f"  Peak pressure: {np.max(p)/1e3:.2f} kPa")
    print()

    # Analyze single mode
    m = 2
    print(f"Analyzing m={m} mode...")
    xi_r = ep.perturbation_radial(m, amplitude=1e-4)
    result = ep.total_energy(xi_r, B_z, B_theta, p, m)

    print(f"  Field bending energy: {result['W_bend']:.3e} J/m")
    print(f"  Compression energy: {result['W_comp']:.3e} J/m")
    print(f"  Pressure energy: {result['W_p']:.3e} J/m")
    print(f"  Total δW: {result['W_total']:.3e} J/m")
    print(f"  Stability: {'STABLE' if result['stable'] else 'UNSTABLE'}")
    print()

    # Plot energy decomposition
    plot_energy_decomposition(ep, B_z, B_theta, p, m=2)
    print("Energy decomposition plot saved as 'energy_principle.png'")

    # Scan mode numbers
    print("Scanning mode numbers (m, n)...")
    W_map, m_values, n_values = scan_mode_numbers(ep, B_z, B_theta, p,
                                                   m_max=5, n_max=5)

    # Plot stability map
    plot_stability_map(W_map, m_values, n_values)
    print("Stability map saved as 'stability_map.png'")


if __name__ == '__main__':
    main()
