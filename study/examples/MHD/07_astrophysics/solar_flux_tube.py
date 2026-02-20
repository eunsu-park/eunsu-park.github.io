#!/usr/bin/env python3
"""
Solar Flux Tube Buoyancy Model

This module simulates the rise of a magnetic flux tube through the solar
convection zone due to magnetic buoyancy.

Physical principles:
- Magnetic pressure reduces internal gas pressure and density
- Buoyant force: F_b = (ρ_ext - ρ_int)g × Volume
- Drag force: F_d = -C_D × ρ_ext × v² × Area
- Flux tube expands as it rises (pressure balance)

The model tracks:
- Trajectory of flux tube from base of convection zone to surface
- Rise velocity evolution
- Magnetic field strength variation with height
- Comparison with observed solar active region emergence

Author: MHD Course Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class SolarFluxTube:
    """
    Thin flux tube model in stratified solar atmosphere.

    Attributes:
        R_sun (float): Solar radius (m)
        g_sun (float): Solar surface gravity (m/s²)
        depth_base (float): Depth of convection zone base (m)
        T_base (float): Temperature at base (K)
        rho_base (float): Density at base (kg/m³)
        B0 (float): Initial magnetic field strength (T)
        tube_radius (float): Initial flux tube radius (m)
    """

    def __init__(self, R_sun=6.96e8, g_sun=274.0,
                 depth_base=2.0e8, T_base=2e6, rho_base=200.0,
                 B0=1e5, tube_radius=1e7):
        """
        Initialize solar flux tube model.

        Parameters:
            R_sun (float): Solar radius (m)
            g_sun (float): Surface gravity (m/s²)
            depth_base (float): Convection zone depth (m)
            T_base (float): Temperature at base (K)
            rho_base (float): Density at base (kg/m³)
            B0 (float): Initial field strength (Gauss → Tesla)
            tube_radius (float): Initial tube radius (m)
        """
        self.R_sun = R_sun
        self.g_sun = g_sun
        self.depth_base = depth_base
        self.T_base = T_base
        self.rho_base = rho_base
        self.B0 = B0 * 1e-4  # Convert Gauss to Tesla
        self.tube_radius = tube_radius

        # Gas constant
        self.mu = 0.6  # Mean molecular weight
        self.m_p = 1.67e-27  # Proton mass (kg)
        self.k_B = 1.38e-23  # Boltzmann constant (J/K)
        self.R_gas = self.k_B / (self.mu * self.m_p)

        # Stratification scale height
        self.H = self.R_gas * self.T_base / self.g_sun

        # Drag coefficient
        self.C_D = 1.0

        # History
        self.time_history = []
        self.height_history = []
        self.velocity_history = []
        self.B_history = []
        self.radius_history = []

    def external_density(self, z):
        """
        External atmospheric density as function of height.

        Assumes exponential stratification: ρ(z) = ρ₀ exp(-z/H)

        Parameters:
            z (float): Height above base (m)

        Returns:
            float: Density (kg/m³)
        """
        return self.rho_base * np.exp(-z / self.H)

    def external_pressure(self, z):
        """
        External atmospheric pressure.

        Parameters:
            z (float): Height above base (m)

        Returns:
            float: Pressure (Pa)
        """
        rho = self.external_density(z)
        T = self.T_base * np.exp(-z / (2 * self.H))  # Simplified temp profile
        return rho * self.R_gas * T

    def internal_density(self, z, B):
        """
        Internal flux tube density from pressure balance.

        Total pressure balance: p_int + B²/(2μ₀) = p_ext

        Parameters:
            z (float): Height (m)
            B (float): Magnetic field (T)

        Returns:
            float: Internal density (kg/m³)
        """
        mu_0 = 4 * np.pi * 1e-7  # Permeability
        p_ext = self.external_pressure(z)
        p_mag = B**2 / (2 * mu_0)

        # Internal gas pressure
        p_int = p_ext - p_mag

        if p_int <= 0:
            # Flux tube has expanded to very low density
            return 1e-3

        # Assuming same temperature inside and outside
        T = self.T_base * np.exp(-z / (2 * self.H))
        rho_int = p_int / (self.R_gas * T)

        return max(rho_int, 1e-3)

    def buoyancy_force(self, z, B):
        """
        Compute buoyancy force per unit mass.

        Parameters:
            z (float): Height (m)
            B (float): Magnetic field (T)

        Returns:
            float: Buoyancy acceleration (m/s²)
        """
        rho_ext = self.external_density(z)
        rho_int = self.internal_density(z, B)

        # Buoyancy acceleration
        g_eff = self.g_sun * (1 - rho_int / rho_ext)

        return g_eff

    def drag_force(self, z, v):
        """
        Compute drag force per unit mass.

        Parameters:
            z (float): Height (m)
            v (float): Velocity (m/s)

        Returns:
            float: Drag acceleration (m/s²)
        """
        rho_ext = self.external_density(z)

        # Drag per unit mass (assumes cylindrical tube)
        # F_d/m_tube = -C_D × (ρ_ext/ρ_int) × v²
        # Simplified: proportional to velocity
        drag_coeff = self.C_D * rho_ext / self.rho_base
        a_drag = -drag_coeff * v

        return a_drag

    def magnetic_field_evolution(self, z, B):
        """
        Magnetic field evolves due to tube expansion.

        Flux conservation: B × A = constant
        For thin tube: B ∝ ρ_int (from pressure balance)

        Parameters:
            z (float): Height (m)
            B (float): Current field (T)

        Returns:
            float: Updated field (T)
        """
        rho_int = self.internal_density(z, B)
        B_new = self.B0 * np.sqrt(rho_int / self.rho_base)

        return B_new

    def rhs(self, state, t):
        """
        Right-hand side for flux tube motion.

        state = [z, v]
        dz/dt = v
        dv/dt = g_buoyancy + a_drag

        Parameters:
            state (list): [height, velocity]
            t (float): Time

        Returns:
            list: Time derivatives
        """
        z, v = state

        # Current magnetic field
        B = self.magnetic_field_evolution(z, self.B0)

        # Forces
        a_buoy = self.buoyancy_force(z, B)
        a_drag = self.drag_force(z, v)

        dz_dt = v
        dv_dt = a_buoy + a_drag

        return [dz_dt, dv_dt]

    def run_simulation(self, t_end=1e6):
        """
        Simulate flux tube rise.

        Parameters:
            t_end (float): End time (s)
        """
        print("Simulating solar flux tube rise...")
        print(f"Initial depth: {self.depth_base/1e8:.1f} × 10^8 m")
        print(f"Initial field: {self.B0*1e4:.0f} Gauss")
        print(f"Scale height: {self.H/1e8:.2f} × 10^8 m")

        # Initial state: at base, zero velocity
        state0 = [0.0, 0.0]

        # Time array
        t_array = np.linspace(0, t_end, 1000)

        # Integrate
        solution = odeint(self.rhs, state0, t_array)

        self.time_history = t_array
        self.height_history = solution[:, 0]
        self.velocity_history = solution[:, 1]

        # Compute B and radius history
        self.B_history = []
        self.radius_history = []

        for z in self.height_history:
            B = self.magnetic_field_evolution(z, self.B0)
            self.B_history.append(B)

            # Radius from flux conservation
            rho_int = self.internal_density(z, B)
            r = self.tube_radius * np.sqrt(self.rho_base / rho_int)
            self.radius_history.append(r)

        self.B_history = np.array(self.B_history)
        self.radius_history = np.array(self.radius_history)

        # Find emergence time
        emergence_idx = np.argmax(self.height_history >= self.depth_base)
        if emergence_idx > 0:
            t_emerge = self.time_history[emergence_idx]
            print(f"\nFlux tube emerges at t = {t_emerge/86400:.1f} days")
            print(f"Emergence velocity: {self.velocity_history[emergence_idx]:.1f} m/s")
            print(f"Surface field: {self.B_history[emergence_idx]*1e4:.0f} Gauss")

    def plot_trajectory(self):
        """
        Plot flux tube trajectory and properties.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Convert to convenient units
        t_days = self.time_history / 86400
        z_Mm = self.height_history / 1e6
        B_gauss = self.B_history * 1e4
        r_km = self.radius_history / 1e3

        # Height vs time
        ax1.plot(t_days, z_Mm, 'b-', linewidth=2)
        ax1.axhline(y=self.depth_base/1e6, color='r', linestyle='--',
                   label='Surface', linewidth=2)
        ax1.set_xlabel('Time (days)', fontsize=11)
        ax1.set_ylabel('Height (Mm)', fontsize=11)
        ax1.set_title('Flux Tube Rise Trajectory', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Velocity vs height
        ax2.plot(z_Mm, self.velocity_history, 'g-', linewidth=2)
        ax2.set_xlabel('Height (Mm)', fontsize=11)
        ax2.set_ylabel('Velocity (m/s)', fontsize=11)
        ax2.set_title('Rise Velocity vs Height', fontsize=13)
        ax2.grid(True, alpha=0.3)

        # Magnetic field vs height
        ax3.plot(z_Mm, B_gauss, 'r-', linewidth=2)
        ax3.set_xlabel('Height (Mm)', fontsize=11)
        ax3.set_ylabel('Magnetic Field (Gauss)', fontsize=11)
        ax3.set_title('Magnetic Field Evolution', fontsize=13)
        ax3.grid(True, alpha=0.3)

        # Tube radius vs height
        ax4.plot(z_Mm, r_km, 'm-', linewidth=2)
        ax4.set_xlabel('Height (Mm)', fontsize=11)
        ax4.set_ylabel('Tube Radius (km)', fontsize=11)
        ax4.set_title('Flux Tube Expansion', fontsize=13)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def main():
    """
    Main function demonstrating solar flux tube rise.
    """
    print("=" * 60)
    print("Solar Flux Tube Buoyancy Simulation")
    print("=" * 60)

    # Create flux tube model
    tube = SolarFluxTube(
        R_sun=6.96e8,
        g_sun=274.0,
        depth_base=2.0e8,  # 200 Mm convection zone
        T_base=2e6,        # 2 MK at base
        rho_base=200.0,    # kg/m³
        B0=1e5,            # 100,000 Gauss
        tube_radius=1e7    # 10,000 km
    )

    # Run simulation (about 30 days)
    tube.run_simulation(t_end=3e6)

    # Plot results
    tube.plot_trajectory()

    plt.savefig('/tmp/solar_flux_tube.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to /tmp/solar_flux_tube.png")

    plt.show()


if __name__ == "__main__":
    main()
