#!/usr/bin/env python3
"""
Tokamak Profile Analysis

Generates and analyzes standard tokamak radial profiles including:
- Pressure p(ψ)
- Safety factor q(ψ)
- Current density J(ψ)
- Density n(ψ)
- Temperature T(ψ)

Compares L-mode and H-mode (with pedestal) profiles.
Computes key dimensionless parameters: βN, li, βp.

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.constants import mu_0, e, m_p

# Physical constants
MU_0 = mu_0
E_CHARGE = e
M_PROTON = m_p


class TokamakProfiles:
    """
    Class to generate and analyze tokamak equilibrium profiles.

    Attributes
    ----------
    psi_norm : ndarray
        Normalized flux coordinate (0 at axis, 1 at edge)
    """

    def __init__(self, n_points=200):
        """
        Initialize profile generator.

        Parameters
        ----------
        n_points : int
            Number of radial points
        """
        self.psi_norm = np.linspace(0, 1, n_points)
        self.n_points = n_points

    def temperature_profile(self, T0, T_edge, mode='L', alpha_T=2.0):
        """
        Generate temperature profile.

        Parameters
        ----------
        T0 : float
            Central temperature (keV)
        T_edge : float
            Edge temperature (keV)
        mode : str
            'L' for L-mode, 'H' for H-mode with pedestal
        alpha_T : float
            Profile shape parameter

        Returns
        -------
        T : ndarray
            Temperature profile (keV)
        """
        psi = self.psi_norm

        if mode == 'L':
            # L-mode: smooth parabolic profile
            T = T_edge + (T0 - T_edge) * (1 - psi**alpha_T)

        elif mode == 'H':
            # H-mode: core + pedestal
            T_ped = 0.2 * T0  # Pedestal temperature
            ped_width = 0.05
            ped_position = 0.9

            # Core profile
            T_core = T_ped + (T0 - T_ped) * (1 - (psi/ped_position)**alpha_T)
            T_core = np.where(psi < ped_position, T_core, T_ped)

            # Pedestal drop
            pedestal_mask = (psi >= ped_position) & (psi <= ped_position + ped_width)
            T_drop = T_ped + (T_edge - T_ped) * ((psi - ped_position) / ped_width)
            T = np.where(pedestal_mask, T_drop, T_core)
            T = np.where(psi > ped_position + ped_width, T_edge, T)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return T

    def density_profile(self, n0, n_edge, mode='L', alpha_n=0.5):
        """
        Generate density profile.

        Parameters
        ----------
        n0 : float
            Central density (10^19 m^-3)
        n_edge : float
            Edge density (10^19 m^-3)
        mode : str
            'L' or 'H' mode
        alpha_n : float
            Profile shape parameter

        Returns
        -------
        n : ndarray
            Density profile (10^19 m^-3)
        """
        psi = self.psi_norm

        if mode == 'L':
            n = n_edge + (n0 - n_edge) * (1 - psi**alpha_n)

        elif mode == 'H':
            n_ped = 0.5 * n0
            ped_width = 0.05
            ped_position = 0.9

            n_core = n_ped + (n0 - n_ped) * (1 - (psi/ped_position)**alpha_n)
            n_core = np.where(psi < ped_position, n_core, n_ped)

            pedestal_mask = (psi >= ped_position) & (psi <= ped_position + ped_width)
            n_drop = n_ped + (n_edge - n_ped) * ((psi - ped_position) / ped_width)
            n = np.where(pedestal_mask, n_drop, n_core)
            n = np.where(psi > ped_position + ped_width, n_edge, n)

        return n

    def pressure_profile(self, T, n):
        """
        Compute pressure from temperature and density.

        p = 2 * n * k_B * T (factor 2 for electrons + ions)

        Parameters
        ----------
        T : ndarray
            Temperature (keV)
        n : ndarray
            Density (10^19 m^-3)

        Returns
        -------
        p : ndarray
            Pressure (kPa)
        """
        # Convert to SI units
        T_J = T * 1e3 * E_CHARGE  # keV -> J
        n_SI = n * 1e19  # 10^19 m^-3 -> m^-3

        p = 2 * n_SI * T_J  # Pa
        return p / 1e3  # Convert to kPa

    def safety_factor_profile(self, q0, q_edge, mode='monotonic'):
        """
        Generate safety factor profile.

        Parameters
        ----------
        q0 : float
            Central safety factor
        q_edge : float
            Edge safety factor
        mode : str
            'monotonic' or 'reversed' (for advanced scenarios)

        Returns
        -------
        q : ndarray
            Safety factor profile
        """
        psi = self.psi_norm

        if mode == 'monotonic':
            # Standard monotonic q profile
            q = q0 + (q_edge - q0) * psi**2

        elif mode == 'reversed':
            # Reversed shear (for ITB scenarios)
            q_min = 1.5
            psi_min = 0.4
            q = np.where(psi < psi_min,
                        q0 + (q_min - q0) * (psi / psi_min)**2,
                        q_min + (q_edge - q_min) * ((psi - psi_min) / (1 - psi_min))**2)

        return q

    def current_density_profile(self, q, B0, R0, a):
        """
        Compute current density from safety factor.

        J ~ dq/dψ (simplified relation)

        Parameters
        ----------
        q : ndarray
            Safety factor
        B0 : float
            Toroidal field (T)
        R0 : float
            Major radius (m)
        a : float
            Minor radius (m)

        Returns
        -------
        J : ndarray
            Current density (MA/m^2)
        """
        psi = self.psi_norm
        r = psi * a  # Approximate minor radius

        # J_phi ~ (r*B0)/(R0*q) for circular geometry
        J = np.zeros_like(psi)
        J[1:] = (r[1:] * B0) / (R0 * q[1:] + 1e-10)
        J = J / (a**2)  # Normalize

        # Convert to MA/m^2
        J = J * 10  # Scaling factor

        return J


def compute_beta_parameters(p, B0, R0, a, I_p):
    """
    Compute dimensionless beta parameters.

    Parameters
    ----------
    p : ndarray
        Pressure profile (Pa)
    B0 : float
        Toroidal field (T)
    R0 : float
        Major radius (m)
    a : float
        Minor radius (m)
    I_p : float
        Plasma current (MA)

    Returns
    -------
    dict
        Dictionary with beta_N, beta_p, beta_toroidal, li
    """
    # Volume-averaged pressure
    p_SI = p * 1e3  # kPa -> Pa
    p_avg = np.mean(p_SI)

    # Toroidal beta
    beta_toroidal = 2 * MU_0 * p_avg / B0**2

    # Poloidal beta (simplified)
    B_p = MU_0 * I_p * 1e6 / (2 * np.pi * a)  # Poloidal field estimate
    beta_p = 2 * MU_0 * p_avg / B_p**2

    # Normalized beta
    beta_N = beta_toroidal / (I_p / (a * B0))

    # Internal inductance (simplified, assuming circular cross-section)
    li = 1.0  # Typical value

    return {
        'beta_toroidal': beta_toroidal * 100,  # Convert to %
        'beta_p': beta_p,
        'beta_N': beta_N,
        'li': li
    }


def plot_profiles(profiles_L, profiles_H, psi_norm):
    """
    Plot comparison of L-mode and H-mode profiles.

    Parameters
    ----------
    profiles_L : dict
        L-mode profiles
    profiles_H : dict
        H-mode profiles
    psi_norm : ndarray
        Normalized flux coordinate
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Temperature
    axes[0, 0].plot(psi_norm, profiles_L['T'], 'b-', linewidth=2, label='L-mode')
    axes[0, 0].plot(psi_norm, profiles_H['T'], 'r-', linewidth=2, label='H-mode')
    axes[0, 0].set_xlabel('Normalized flux ψ_N', fontsize=11)
    axes[0, 0].set_ylabel('Temperature (keV)', fontsize=11)
    axes[0, 0].set_title('Temperature Profile', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Density
    axes[0, 1].plot(psi_norm, profiles_L['n'], 'b-', linewidth=2, label='L-mode')
    axes[0, 1].plot(psi_norm, profiles_H['n'], 'r-', linewidth=2, label='H-mode')
    axes[0, 1].set_xlabel('Normalized flux ψ_N', fontsize=11)
    axes[0, 1].set_ylabel('Density (10¹⁹ m⁻³)', fontsize=11)
    axes[0, 1].set_title('Density Profile', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Pressure
    axes[0, 2].plot(psi_norm, profiles_L['p'], 'b-', linewidth=2, label='L-mode')
    axes[0, 2].plot(psi_norm, profiles_H['p'], 'r-', linewidth=2, label='H-mode')
    axes[0, 2].set_xlabel('Normalized flux ψ_N', fontsize=11)
    axes[0, 2].set_ylabel('Pressure (kPa)', fontsize=11)
    axes[0, 2].set_title('Pressure Profile', fontsize=12, fontweight='bold')
    axes[0, 2].legend(fontsize=10)
    axes[0, 2].grid(True, alpha=0.3)

    # Safety factor
    axes[1, 0].plot(psi_norm, profiles_L['q'], 'b-', linewidth=2, label='L-mode')
    axes[1, 0].plot(psi_norm, profiles_H['q'], 'r-', linewidth=2, label='H-mode')
    axes[1, 0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].axhline(y=2, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Normalized flux ψ_N', fontsize=11)
    axes[1, 0].set_ylabel('Safety factor q', fontsize=11)
    axes[1, 0].set_title('Safety Factor', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Current density
    axes[1, 1].plot(psi_norm, profiles_L['J'], 'b-', linewidth=2, label='L-mode')
    axes[1, 1].plot(psi_norm, profiles_H['J'], 'r-', linewidth=2, label='H-mode')
    axes[1, 1].set_xlabel('Normalized flux ψ_N', fontsize=11)
    axes[1, 1].set_ylabel('J (MA/m²)', fontsize=11)
    axes[1, 1].set_title('Current Density', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    # Pressure gradient (related to stability)
    dp_L = np.gradient(profiles_L['p'], psi_norm)
    dp_H = np.gradient(profiles_H['p'], psi_norm)
    axes[1, 2].plot(psi_norm, np.abs(dp_L), 'b-', linewidth=2, label='L-mode')
    axes[1, 2].plot(psi_norm, np.abs(dp_H), 'r-', linewidth=2, label='H-mode')
    axes[1, 2].set_xlabel('Normalized flux ψ_N', fontsize=11)
    axes[1, 2].set_ylabel('|dp/dψ| (kPa)', fontsize=11)
    axes[1, 2].set_title('Pressure Gradient', fontsize=12, fontweight='bold')
    axes[1, 2].legend(fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('Tokamak Radial Profiles: L-mode vs H-mode',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('tokamak_profiles.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    # Initialize profile generator
    profiles_gen = TokamakProfiles(n_points=200)

    # Tokamak parameters
    R0 = 1.8  # Major radius (m)
    a = 0.5   # Minor radius (m)
    B0 = 3.0  # Toroidal field (T)
    I_p = 1.0  # Plasma current (MA)

    # Profile parameters
    T0 = 10.0  # Central temperature (keV)
    T_edge = 0.1  # Edge temperature (keV)
    n0 = 8.0   # Central density (10^19 m^-3)
    n_edge = 1.0  # Edge density (10^19 m^-3)
    q0 = 1.0   # Central q
    q_edge = 4.0  # Edge q

    print("Tokamak Profile Analysis")
    print("=" * 60)
    print(f"Device parameters:")
    print(f"  Major radius R0: {R0:.2f} m")
    print(f"  Minor radius a: {a:.2f} m")
    print(f"  Toroidal field B0: {B0:.1f} T")
    print(f"  Plasma current I_p: {I_p:.1f} MA")
    print()

    # Generate L-mode profiles
    T_L = profiles_gen.temperature_profile(T0, T_edge, mode='L', alpha_T=2.0)
    n_L = profiles_gen.density_profile(n0, n_edge, mode='L', alpha_n=0.5)
    p_L = profiles_gen.pressure_profile(T_L, n_L)
    q_L = profiles_gen.safety_factor_profile(q0, q_edge, mode='monotonic')
    J_L = profiles_gen.current_density_profile(q_L, B0, R0, a)

    # Generate H-mode profiles
    T_H = profiles_gen.temperature_profile(T0, T_edge, mode='H', alpha_T=2.0)
    n_H = profiles_gen.density_profile(n0, n_edge, mode='H', alpha_n=0.5)
    p_H = profiles_gen.pressure_profile(T_H, n_H)
    q_H = profiles_gen.safety_factor_profile(q0, q_edge, mode='monotonic')
    J_H = profiles_gen.current_density_profile(q_H, B0, R0, a)

    profiles_L = {'T': T_L, 'n': n_L, 'p': p_L, 'q': q_L, 'J': J_L}
    profiles_H = {'T': T_H, 'n': n_H, 'p': p_H, 'q': q_H, 'J': J_H}

    # Compute beta parameters
    beta_L = compute_beta_parameters(p_L, B0, R0, a, I_p)
    beta_H = compute_beta_parameters(p_H, B0, R0, a, I_p)

    print("L-mode parameters:")
    print(f"  β_toroidal: {beta_L['beta_toroidal']:.3f}%")
    print(f"  β_poloidal: {beta_L['beta_p']:.3f}")
    print(f"  β_N: {beta_L['beta_N']:.3f}")
    print(f"  li: {beta_L['li']:.2f}")
    print()

    print("H-mode parameters:")
    print(f"  β_toroidal: {beta_H['beta_toroidal']:.3f}%")
    print(f"  β_poloidal: {beta_H['beta_p']:.3f}")
    print(f"  β_N: {beta_H['beta_N']:.3f}")
    print(f"  li: {beta_H['li']:.2f}")
    print()

    # Plot profiles
    plot_profiles(profiles_L, profiles_H, profiles_gen.psi_norm)
    print("Profile comparison plot saved as 'tokamak_profiles.png'")


if __name__ == '__main__':
    main()
