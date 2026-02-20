#!/usr/bin/env python3
"""
Parker Solar Wind Model

This module implements the Parker solar wind solution, which explains how
the solar corona expands to create the solar wind that fills the heliosphere.

Key features:
- Isothermal Parker solution with critical point
- Transonic flow: subsonic → supersonic transition at r_c
- Multiple solution branches: breeze, wind, accretion
- Comparison with observations
- Extension to non-isothermal case

The critical point occurs where the flow velocity equals the local sound speed.

Author: MHD Course Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve


class ParkerSolarWind:
    """
    Parker solar wind model solver.

    Attributes:
        M_sun (float): Solar mass (kg)
        R_sun (float): Solar radius (m)
        T_corona (float): Coronal temperature (K)
        n_base (float): Base density (m^-3)
        gamma (float): Polytropic index
    """

    def __init__(self, M_sun=1.989e30, R_sun=6.96e8,
                 T_corona=1.5e6, n_base=1e14, gamma=1.0):
        """
        Initialize Parker solar wind model.

        Parameters:
            M_sun (float): Solar mass (kg)
            R_sun (float): Solar radius (m)
            T_corona (float): Coronal temperature (K)
            n_base (float): Base number density (m^-3)
            gamma (float): Polytropic index (1.0 = isothermal)
        """
        self.M_sun = M_sun
        self.R_sun = R_sun
        self.T_corona = T_corona
        self.n_base = n_base
        self.gamma = gamma

        # Constants
        self.G = 6.674e-11  # Gravitational constant
        self.k_B = 1.38e-23  # Boltzmann constant
        self.m_p = 1.67e-27  # Proton mass

        # Sound speed (isothermal)
        self.c_s = np.sqrt(2 * self.k_B * T_corona / self.m_p)

        # Critical radius (sonic point)
        self.r_c = self.G * M_sun / (2 * self.c_s**2)

        # Escape velocity at base
        self.v_esc = np.sqrt(2 * self.G * M_sun / R_sun)

    def parker_equation_rhs(self, v, r):
        """
        Right-hand side of Parker wind equation.

        dv/dr = (2c_s²/r - GM/r²) / (v - c_s²/v)

        Parameters:
            v (float): Velocity (m/s)
            r (float): Radius (m)

        Returns:
            float: dv/dr
        """
        G = self.G
        M = self.M_sun
        c_s = self.c_s

        # Numerator
        numerator = 2 * c_s**2 / r - G * M / r**2

        # Denominator
        denominator = v - c_s**2 / v

        # Avoid division by zero near sonic point
        if abs(denominator) < 1e-10:
            return 0.0

        dv_dr = numerator / denominator

        return dv_dr

    def solve_parker_wind(self, r_array, v_init, solution_type='wind'):
        """
        Solve Parker wind equation.

        Parameters:
            r_array (ndarray): Radial grid
            v_init (float): Initial velocity at r_array[0]
            solution_type (str): 'wind', 'breeze', or 'accretion'

        Returns:
            ndarray: Velocity solution
        """
        # For wind solution: start just above critical point
        # For breeze: start below critical point and integrate inward

        if solution_type == 'wind':
            # Start slightly above sonic point, integrate outward
            r_start = self.r_c * 1.01
            v_start = self.c_s * 1.01

            r_integrate = r_array[r_array >= r_start]
            v_solution = odeint(lambda v, r: self.parker_equation_rhs(v[0], r),
                               [v_start], r_integrate)[:, 0]

            # Pad with NaN for r < r_c
            v_full = np.full_like(r_array, np.nan)
            v_full[r_array >= r_start] = v_solution

            return v_full

        elif solution_type == 'breeze':
            # Start at base, integrate outward (stays subsonic)
            v_solution = odeint(lambda v, r: self.parker_equation_rhs(v[0], r),
                               [v_init], r_array)[:, 0]
            return v_solution

        elif solution_type == 'accretion':
            # Integrate inward from infinity (supersonic infall)
            r_start = r_array[-1]
            v_start = -self.c_s * 2.0  # Negative = inward

            r_integrate = r_array[::-1]
            v_solution = odeint(lambda v, r: self.parker_equation_rhs(v[0], r),
                               [v_start], r_integrate)[:, 0]

            return v_solution[::-1]

        return np.zeros_like(r_array)

    def density_from_continuity(self, r, v, n0, r0):
        """
        Compute density from continuity equation.

        ρ(r) v(r) r² = ρ₀ v₀ r₀²

        Parameters:
            r (float or ndarray): Radius
            v (float or ndarray): Velocity
            n0 (float): Base density
            r0 (float): Base radius

        Returns:
            float or ndarray: Density
        """
        # Find velocity at r0
        idx = np.argmin(np.abs(r - r0))
        v0 = v[idx]

        # Continuity
        n = n0 * (v0 / v) * (r0 / r)**2

        return n

    def temperature_profile(self, r, T0, r0):
        """
        Temperature profile for polytropic case.

        T(r) = T₀ (ρ/ρ₀)^(γ-1)

        Parameters:
            r (ndarray): Radius
            T0 (float): Base temperature
            r0 (float): Base radius

        Returns:
            ndarray: Temperature
        """
        if self.gamma == 1.0:
            # Isothermal
            return np.full_like(r, T0)
        else:
            # Polytropic (simplified)
            return T0 * (r0 / r)**(2 * (self.gamma - 1))

    def plot_solutions(self):
        """
        Plot Parker wind solutions: wind, breeze, accretion.
        """
        # Radial grid from 1 R_sun to 100 R_sun
        r_array = np.linspace(self.R_sun, 100 * self.R_sun, 1000)

        # Solve for different solution types
        v_wind = self.solve_parker_wind(r_array, 0, 'wind')
        v_breeze = self.solve_parker_wind(r_array, 1e3, 'breeze')
        v_accretion = self.solve_parker_wind(r_array, 0, 'accretion')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Velocity solutions
        ax1.plot(r_array / self.R_sun, v_wind / 1e3, 'b-', linewidth=2,
                label='Wind (transonic)')
        ax1.plot(r_array / self.R_sun, v_breeze / 1e3, 'g--', linewidth=2,
                label='Breeze (subsonic)')
        ax1.plot(r_array / self.R_sun, np.abs(v_accretion) / 1e3, 'r:',
                linewidth=2, label='Accretion (infall)')

        # Sound speed
        ax1.axhline(y=self.c_s / 1e3, color='k', linestyle='--', alpha=0.5,
                   label=f'Sound speed ({self.c_s/1e3:.0f} km/s)')

        # Critical point
        ax1.axvline(x=self.r_c / self.R_sun, color='orange', linestyle='--',
                   alpha=0.5, label=f'Critical radius ({self.r_c/self.R_sun:.1f} R☉)')

        ax1.set_xlabel(r'Radius ($R_\odot$)', fontsize=12)
        ax1.set_ylabel('Velocity (km/s)', fontsize=12)
        ax1.set_title('Parker Solar Wind Solutions', fontsize=14)
        ax1.set_xlim([1, 100])
        ax1.set_ylim([0, 1000])
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Mach number
        M_wind = v_wind / self.c_s
        M_breeze = v_breeze / self.c_s

        ax2.semilogy(r_array / self.R_sun, M_wind, 'b-', linewidth=2,
                    label='Wind')
        ax2.semilogy(r_array / self.R_sun, M_breeze, 'g--', linewidth=2,
                    label='Breeze')
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5,
                   label='Sonic point (M=1)')
        ax2.axvline(x=self.r_c / self.R_sun, color='orange', linestyle='--',
                   alpha=0.5)

        ax2.set_xlabel(r'Radius ($R_\odot$)', fontsize=12)
        ax2.set_ylabel('Mach Number', fontsize=12)
        ax2.set_title('Mach Number vs Radius', fontsize=14)
        ax2.set_xlim([1, 100])
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        return fig

    def plot_with_observations(self):
        """
        Plot Parker wind solution with observational data points.
        """
        r_array = np.linspace(self.R_sun, 100 * self.R_sun, 1000)
        v_wind = self.solve_parker_wind(r_array, 0, 'wind')

        # Compute density and temperature
        n_wind = self.density_from_continuity(r_array, v_wind,
                                              self.n_base, self.R_sun)
        T_wind = self.temperature_profile(r_array, self.T_corona, self.R_sun)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Velocity
        ax1.plot(r_array / self.R_sun, v_wind / 1e3, 'b-', linewidth=2,
                label='Parker model')

        # Observational points (typical values)
        r_obs = np.array([1, 10, 50, 100])
        v_obs = np.array([0, 200, 350, 400])
        ax1.plot(r_obs, v_obs, 'ro', markersize=8, label='Observations')

        ax1.set_xlabel(r'Radius ($R_\odot$)', fontsize=11)
        ax1.set_ylabel('Velocity (km/s)', fontsize=11)
        ax1.set_title('Solar Wind Velocity', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Density
        ax2.semilogy(r_array / self.R_sun, n_wind, 'b-', linewidth=2)
        ax2.set_xlabel(r'Radius ($R_\odot$)', fontsize=11)
        ax2.set_ylabel(r'Density (m$^{-3}$)', fontsize=11)
        ax2.set_title('Number Density', fontsize=13)
        ax2.grid(True, alpha=0.3)

        # Temperature
        ax3.plot(r_array / self.R_sun, T_wind / 1e6, 'b-', linewidth=2)
        ax3.set_xlabel(r'Radius ($R_\odot$)', fontsize=11)
        ax3.set_ylabel('Temperature (MK)', fontsize=11)
        ax3.set_title('Temperature Profile', fontsize=13)
        ax3.grid(True, alpha=0.3)

        # Mass flux
        mass_flux = n_wind * self.m_p * v_wind * 4 * np.pi * r_array**2
        ax4.plot(r_array / self.R_sun, mass_flux / 1e9, 'b-', linewidth=2)
        ax4.set_xlabel(r'Radius ($R_\odot$)', fontsize=11)
        ax4.set_ylabel(r'Mass Flux ($10^9$ kg/s)', fontsize=11)
        ax4.set_title('Mass Loss Rate (should be constant)', fontsize=13)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def main():
    """
    Main function demonstrating Parker solar wind model.
    """
    print("=" * 60)
    print("Parker Solar Wind Model")
    print("=" * 60)

    # Create Parker wind model
    parker = ParkerSolarWind(
        M_sun=1.989e30,
        R_sun=6.96e8,
        T_corona=1.5e6,  # 1.5 MK
        n_base=1e14,     # 10^8 cm^-3
        gamma=1.0        # Isothermal
    )

    print(f"\nParameters:")
    print(f"  Coronal temperature: {parker.T_corona/1e6:.1f} MK")
    print(f"  Sound speed: {parker.c_s/1e3:.0f} km/s")
    print(f"  Critical radius: {parker.r_c/parker.R_sun:.2f} R☉")
    print(f"  Escape velocity at R☉: {parker.v_esc/1e3:.0f} km/s")

    # Plot solutions
    parker.plot_solutions()
    parker.plot_with_observations()

    plt.savefig('/tmp/parker_solar_wind.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to /tmp/parker_solar_wind.png")

    plt.show()


if __name__ == "__main__":
    main()
