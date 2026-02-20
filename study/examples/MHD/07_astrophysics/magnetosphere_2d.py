#!/usr/bin/env python3
"""
Simplified 2D Magnetosphere Model

This module creates a simplified 2D magnetosphere model showing the interaction
between Earth's dipole magnetic field and the solar wind.

Key features:
- Dipole field inside magnetopause
- Uniform solar wind flow (potential flow approximation)
- Chapman-Ferraro magnetopause boundary (pressure balance)
- Visualization of field lines, stagnation point, and cusps

The magnetopause is where magnetic pressure balances solar wind ram pressure:
    B²/(2μ₀) = ½ρ_sw v_sw²

Author: MHD Course Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt


class Magnetosphere2D:
    """
    2D magnetosphere model with dipole field and solar wind.

    Attributes:
        B_dipole (float): Dipole moment strength (T·m³)
        R_planet (float): Planet radius (m)
        v_sw (float): Solar wind velocity (m/s)
        n_sw (float): Solar wind density (m^-3)
        rho_sw (float): Solar wind mass density (kg/m³)
    """

    def __init__(self, B_dipole=3.1e-5, R_planet=6.4e6,
                 v_sw=4e5, n_sw=5e6):
        """
        Initialize magnetosphere model.

        Parameters:
            B_dipole (float): Dipole field at equator at R_planet (T)
            R_planet (float): Planet radius (m)
            v_sw (float): Solar wind velocity (m/s)
            n_sw (float): Solar wind number density (m^-3)
        """
        self.B_dipole = B_dipole
        self.R_planet = R_planet
        self.v_sw = v_sw
        self.n_sw = n_sw

        # Constants
        self.mu_0 = 4 * np.pi * 1e-7
        self.m_p = 1.67e-27

        # Solar wind mass density
        self.rho_sw = n_sw * self.m_p

        # Dipole moment: M = B_eq * R³
        self.M_dipole = B_dipole * R_planet**3

        # Compute magnetopause standoff distance
        self.R_mp = self.calculate_magnetopause_distance()

    def calculate_magnetopause_distance(self):
        """
        Calculate magnetopause standoff distance from pressure balance.

        B²/(2μ₀) = ½ρ_sw v_sw²

        At subsolar point: B = M/(R_mp)³

        Returns:
            float: Magnetopause distance (m)
        """
        # Dynamic pressure
        P_ram = 0.5 * self.rho_sw * self.v_sw**2

        # Magnetic pressure at distance R: P_mag = B²/(2μ₀) = M²/(2μ₀ R⁶)
        # Solve: M²/(2μ₀ R_mp⁶) = P_ram

        R_mp = (self.M_dipole**2 / (2 * self.mu_0 * P_ram))**(1/6)

        return R_mp

    def dipole_field(self, x, y):
        """
        Compute dipole magnetic field components.

        B = (μ₀/4π) × (3(m·r)r/r⁵ - m/r³)

        For dipole in z-direction in x-y plane:
            B_x = 3μ₀M xy / (4π r⁵)
            B_y = μ₀M(3y² - r²) / (4π r⁵)

        Parameters:
            x, y (ndarray): Coordinates (m)

        Returns:
            tuple: (B_x, B_y) field components (T)
        """
        r = np.sqrt(x**2 + y**2)

        # Avoid singularity at origin
        r = np.maximum(r, 0.1 * self.R_planet)

        # Dipole field (simplified 2D version)
        # In x-y plane with dipole in z-direction
        B_x = (3 * self.M_dipole * x * y) / (r**5)
        B_y = self.M_dipole * (3 * y**2 - r**2) / (r**5)

        return B_x, B_y

    def is_inside_magnetopause(self, x, y):
        """
        Check if point is inside magnetopause.

        Magnetopause shape: Chapman-Ferraro approximation
        r(θ) ≈ R_mp (2/(1+cos(θ)))^(1/6)

        Parameters:
            x, y (ndarray): Coordinates

        Returns:
            ndarray: Boolean mask
        """
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        # Chapman-Ferraro shape (approximate)
        r_mp_theta = self.R_mp * (2 / (1 + np.cos(theta)))**(1/6)

        # Points inside magnetopause
        inside = r < r_mp_theta

        return inside

    def magnetopause_boundary(self, theta_array):
        """
        Compute magnetopause boundary shape.

        Parameters:
            theta_array (ndarray): Angles from 0 to 2π

        Returns:
            tuple: (x, y) coordinates of boundary
        """
        # Chapman-Ferraro model
        r_mp = self.R_mp * (2 / (1 + np.cos(theta_array)))**(1/6)

        x = r_mp * np.cos(theta_array)
        y = r_mp * np.sin(theta_array)

        return x, y

    def plot_magnetosphere(self):
        """
        Plot 2D magnetosphere structure.
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Grid
        x_max = 15 * self.R_planet
        y_max = 15 * self.R_planet
        nx, ny = 150, 150

        x = np.linspace(-5 * self.R_planet, x_max, nx)
        y = np.linspace(-y_max, y_max, ny)
        X, Y = np.meshgrid(x, y)

        # Compute magnetic field
        B_x, B_y = self.dipole_field(X, Y)
        B_mag = np.sqrt(B_x**2 + B_y**2)

        # Mask outside magnetopause
        inside = self.is_inside_magnetopause(X, Y)

        # Plot field magnitude (inside magnetopause only)
        B_plot = np.where(inside, B_mag, np.nan)
        im = ax.contourf(X / self.R_planet, Y / self.R_planet,
                        np.log10(B_plot * 1e9), levels=20,
                        cmap='plasma', alpha=0.6)
        plt.colorbar(im, ax=ax, label=r'$\log_{10}(B)$ [nT]')

        # Plot field lines (streamplot)
        # Sample on coarser grid for clarity
        skip = 5
        ax.streamplot(X[::skip, ::skip] / self.R_planet,
                     Y[::skip, ::skip] / self.R_planet,
                     B_x[::skip, ::skip],
                     B_y[::skip, ::skip],
                     color='white', linewidth=0.5, density=1.0,
                     arrowsize=0.8)

        # Plot magnetopause
        theta = np.linspace(0, 2*np.pi, 200)
        x_mp, y_mp = self.magnetopause_boundary(theta)
        ax.plot(x_mp / self.R_planet, y_mp / self.R_planet,
               'r-', linewidth=2.5, label='Magnetopause')

        # Plot Earth
        earth = plt.Circle((0, 0), 1, color='blue', alpha=0.8,
                          label='Planet')
        ax.add_patch(earth)

        # Mark stagnation point (subsolar point)
        ax.plot(self.R_mp / self.R_planet, 0, 'yo', markersize=12,
               label=f'Stagnation point ({self.R_mp/self.R_planet:.1f} R)')

        # Mark magnetic cusps (approximate)
        cusp_angle = np.pi / 3  # ~60 degrees
        theta_cusp = np.array([cusp_angle, -cusp_angle])
        x_cusp, y_cusp = self.magnetopause_boundary(theta_cusp)
        ax.plot(x_cusp / self.R_planet, y_cusp / self.R_planet,
               'go', markersize=10, label='Cusps')

        # Solar wind direction
        ax.arrow(-4, 12, 3, 0, head_width=1, head_length=0.5,
                fc='orange', ec='orange', linewidth=2)
        ax.text(-2.5, 13, 'Solar Wind', fontsize=12, color='orange',
               weight='bold')

        ax.set_xlabel(r'X ($R_{\rm planet}$)', fontsize=12)
        ax.set_ylabel(r'Y ($R_{\rm planet}$)', fontsize=12)
        ax.set_title('2D Magnetosphere Model\n(Dipole + Solar Wind)',
                    fontsize=14, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim([-5, 15])
        ax.set_ylim([-15, 15])

        plt.tight_layout()
        return fig

    def plot_pressure_profiles(self):
        """
        Plot magnetic and ram pressure along x-axis.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Array along x-axis (y=0)
        x_array = np.linspace(0.5 * self.R_planet, 20 * self.R_planet, 200)
        y_array = np.zeros_like(x_array)

        # Magnetic field and pressure
        B_x, B_y = self.dipole_field(x_array, y_array)
        B_mag = np.sqrt(B_x**2 + B_y**2)
        P_mag = B_mag**2 / (2 * self.mu_0)

        # Solar wind ram pressure
        P_ram = 0.5 * self.rho_sw * self.v_sw**2

        # Plot pressures
        ax1.loglog(x_array / self.R_planet, P_mag * 1e9, 'b-',
                  linewidth=2, label='Magnetic pressure')
        ax1.axhline(y=P_ram * 1e9, color='r', linestyle='--',
                   linewidth=2, label='Solar wind ram pressure')
        ax1.axvline(x=self.R_mp / self.R_planet, color='g',
                   linestyle='--', alpha=0.7, label='Magnetopause')

        ax1.set_xlabel(r'Distance ($R_{\rm planet}$)', fontsize=11)
        ax1.set_ylabel('Pressure (nPa)', fontsize=11)
        ax1.set_title('Pressure Balance along X-axis', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Plot beta = P_gas / P_mag
        # Assuming thermal pressure ~ ram pressure
        beta = P_ram / P_mag

        ax2.loglog(x_array / self.R_planet, beta, 'purple',
                  linewidth=2)
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5,
                   label='β = 1')
        ax2.axvline(x=self.R_mp / self.R_planet, color='g',
                   linestyle='--', alpha=0.7, label='Magnetopause')

        ax2.set_xlabel(r'Distance ($R_{\rm planet}$)', fontsize=11)
        ax2.set_ylabel(r'Plasma Beta $\beta$', fontsize=11)
        ax2.set_title(r'Plasma Beta ($\beta = P_{\rm ram}/P_{\rm mag}$)',
                     fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        return fig


def main():
    """
    Main function demonstrating 2D magnetosphere model.
    """
    print("=" * 60)
    print("2D Magnetosphere Model")
    print("=" * 60)

    # Create magnetosphere (Earth parameters)
    mag = Magnetosphere2D(
        B_dipole=3.1e-5,  # 31,000 nT at equator
        R_planet=6.4e6,   # Earth radius
        v_sw=4e5,         # 400 km/s
        n_sw=5e6          # 5 cm^-3
    )

    print(f"\nParameters:")
    print(f"  Dipole field at surface: {mag.B_dipole*1e9:.0f} nT")
    print(f"  Solar wind velocity: {mag.v_sw/1e3:.0f} km/s")
    print(f"  Solar wind density: {mag.n_sw/1e6:.1f} cm^-3")
    print(f"  Ram pressure: {0.5*mag.rho_sw*mag.v_sw**2*1e9:.2f} nPa")
    print(f"\nMagnetopause standoff distance: {mag.R_mp/mag.R_planet:.1f} R_planet")

    # Plot magnetosphere
    mag.plot_magnetosphere()

    # Plot pressure profiles
    mag.plot_pressure_profiles()

    plt.savefig('/tmp/magnetosphere_2d.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to /tmp/magnetosphere_2d.png")

    plt.show()


if __name__ == "__main__":
    main()
