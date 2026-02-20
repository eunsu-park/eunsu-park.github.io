#!/usr/bin/env python3
"""
Tokamak Equilibrium Calculation

This module performs simplified tokamak equilibrium calculations, computing
key parameters for magnetic confinement fusion.

Key concepts:
- Safety factor q(r): measures field line winding
- Shafranov shift: outward displacement of flux surfaces due to pressure
- Beta values: ratio of plasma to magnetic pressure
- Troyon beta limit: operational constraint

The safety factor is crucial for stability:
    q = r B_toroidal / (R B_poloidal)

Author: MHD Course Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz


class TokamakEquilibrium:
    """
    Simplified tokamak equilibrium solver.

    Attributes:
        R0 (float): Major radius (m)
        a (float): Minor radius (m)
        B_tor (float): Toroidal field at R0 (T)
        I_p (float): Plasma current (MA)
        beta_0 (float): Central beta value
    """

    def __init__(self, R0=3.0, a=1.0, B_tor=5.0, I_p=15.0, beta_0=0.02):
        """
        Initialize tokamak parameters.

        Parameters:
            R0 (float): Major radius (m)
            a (float): Minor radius (m)
            B_tor (float): Toroidal field (T)
            I_p (float): Plasma current (MA)
            beta_0 (float): Central beta
        """
        self.R0 = R0
        self.a = a
        self.B_tor = B_tor
        self.I_p = I_p * 1e6  # Convert to Amperes
        self.beta_0 = beta_0

        # Compute derived quantities
        self.epsilon = a / R0  # Inverse aspect ratio
        self.mu_0 = 4 * np.pi * 1e-7

        # Grid
        self.nr = 100
        self.r = np.linspace(0, a, self.nr)

    def current_density_profile(self, r, alpha=2.0):
        """
        Parameterized current density profile.

        j(r) = j_0 (1 - (r/a)²)^α

        Parameters:
            r (ndarray): Minor radius
            alpha (float): Profile parameter

        Returns:
            ndarray: Current density (A/m²)
        """
        rho = r / self.a
        j = (1 - rho**2)**alpha

        # Normalize to give total current I_p
        # I_p = ∫ j(r) 2πr dr
        integral = 2 * np.pi * np.trapz(j * r, r)
        j_0 = self.I_p / integral

        return j_0 * j

    def pressure_profile(self, r, alpha_p=2.0):
        """
        Pressure profile.

        p(r) = p_0 (1 - (r/a)²)^α_p

        Parameters:
            r (ndarray): Minor radius
            alpha_p (float): Pressure profile parameter

        Returns:
            ndarray: Pressure (Pa)
        """
        rho = r / self.a
        p_norm = (1 - rho**2)**alpha_p

        # p_0 from beta_0
        # β = 2μ₀p/B²
        p_0 = self.beta_0 * self.B_tor**2 / (2 * self.mu_0)

        return p_0 * p_norm

    def compute_q_profile(self):
        """
        Compute safety factor q(r).

        q(r) = (r B_tor) / (R₀ B_pol)
        where B_pol comes from enclosed current.

        Returns:
            ndarray: Safety factor profile
        """
        r = self.r
        j = self.current_density_profile(r)

        # Poloidal field from Ampere's law
        # B_pol(r) = μ₀ I_enclosed / (2π r)

        # Compute enclosed current
        I_enclosed = np.zeros_like(r)
        for i in range(1, len(r)):
            I_enclosed[i] = 2 * np.pi * np.trapz(j[:i+1] * r[:i+1], r[:i+1])

        # Poloidal field
        B_pol = np.zeros_like(r)
        B_pol[1:] = self.mu_0 * I_enclosed[1:] / (2 * np.pi * r[1:])
        B_pol[0] = B_pol[1]  # Avoid singularity

        # Safety factor
        q = r * self.B_tor / (self.R0 * B_pol)
        q[0] = q[1]  # Fix q(0)

        return q, B_pol

    def compute_shafranov_shift(self):
        """
        Compute Shafranov shift Δ(r).

        Simplified formula:
        Δ(r) ≈ (r²/R₀) × (β_pol + l_i/2)

        where β_pol is poloidal beta and l_i is internal inductance.

        Returns:
            ndarray: Shafranov shift
        """
        r = self.r
        p = self.pressure_profile(r)
        q, B_pol = self.compute_q_profile()

        # Poloidal beta
        beta_pol_avg = np.trapz(p * r, r) / np.trapz(0.5 * B_pol**2 / self.mu_0 * r, r)

        # Internal inductance (approximate)
        l_i = 0.5  # Typical value

        # Shafranov shift
        Delta = (r**2 / self.R0) * (beta_pol_avg + l_i / 2)

        return Delta, beta_pol_avg

    def compute_beta_values(self):
        """
        Compute various beta values.

        Returns:
            dict: Beta values
        """
        r = self.r
        p = self.pressure_profile(r)
        q, B_pol = self.compute_q_profile()

        # Volume-averaged pressure
        p_avg = np.trapz(p * r, r) / np.trapz(r, r)

        # Toroidal beta
        beta_tor = 2 * self.mu_0 * p_avg / self.B_tor**2

        # Poloidal beta
        B_pol_avg = np.sqrt(np.trapz(B_pol**2 * r, r) / np.trapz(r, r))
        beta_pol = 2 * self.mu_0 * p_avg / B_pol_avg**2

        # Total beta
        B_total_avg = np.sqrt(self.B_tor**2 + B_pol_avg**2)
        beta_total = 2 * self.mu_0 * p_avg / B_total_avg**2

        # Normalized beta (Troyon)
        beta_N = beta_tor * 100 * self.a * self.B_tor / (self.I_p / 1e6)

        return {
            'beta_tor': beta_tor,
            'beta_pol': beta_pol,
            'beta_total': beta_total,
            'beta_N': beta_N
        }

    def check_troyon_limit(self):
        """
        Check Troyon beta limit: β_N < β_{N,max} ≈ 3

        Returns:
            bool: True if within limit
        """
        betas = self.compute_beta_values()
        beta_N_max = 3.0

        within_limit = betas['beta_N'] < beta_N_max

        return within_limit, betas['beta_N'], beta_N_max

    def plot_equilibrium(self):
        """
        Plot equilibrium profiles and flux surfaces.
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        r = self.r
        q, B_pol = self.compute_q_profile()
        p = self.pressure_profile(r)
        j = self.current_density_profile(r)
        Delta, beta_pol = self.compute_shafranov_shift()

        # Safety factor
        ax1.plot(r / self.a, q, 'b-', linewidth=2)
        ax1.axhline(y=1, color='r', linestyle='--', alpha=0.5,
                   label='q=1 (sawtooth)')
        ax1.axhline(y=2, color='orange', linestyle='--', alpha=0.5,
                   label='q=2 (m=2 kink)')
        ax1.set_xlabel(r'$r/a$', fontsize=11)
        ax1.set_ylabel('Safety Factor q', fontsize=11)
        ax1.set_title('Safety Factor Profile', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.text(0.05, 0.95, f'q(0) = {q[0]:.2f}\nq(a) = {q[-1]:.2f}',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Pressure and current
        ax2_twin = ax2.twinx()
        ax2.plot(r / self.a, p / 1e3, 'b-', linewidth=2, label='Pressure')
        ax2_twin.plot(r / self.a, j / 1e6, 'r--', linewidth=2, label='Current density')
        ax2.set_xlabel(r'$r/a$', fontsize=11)
        ax2.set_ylabel('Pressure (kPa)', fontsize=11, color='b')
        ax2_twin.set_ylabel(r'Current Density (MA/m$^2$)', fontsize=11, color='r')
        ax2.set_title('Pressure and Current Profiles', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor='b')
        ax2_twin.tick_params(axis='y', labelcolor='r')

        # Flux surfaces with Shafranov shift
        theta = np.linspace(0, 2*np.pi, 100)
        colors = plt.cm.viridis(np.linspace(0, 1, 10))

        for i, r_flux in enumerate(np.linspace(0.1, 0.9, 10) * self.a):
            idx = np.argmin(np.abs(r - r_flux))
            shift = Delta[idx]

            R = self.R0 + shift + r_flux * np.cos(theta)
            Z = r_flux * np.sin(theta)

            ax3.plot(R, Z, color=colors[i], linewidth=1.5)

        # First wall
        R_wall = self.R0 + self.a * np.cos(theta)
        Z_wall = self.a * np.sin(theta)
        ax3.plot(R_wall, Z_wall, 'k-', linewidth=2.5, label='First wall')

        ax3.set_xlabel('R (m)', fontsize=11)
        ax3.set_ylabel('Z (m)', fontsize=11)
        ax3.set_title(f'Flux Surfaces with Shafranov Shift\nΔ(a) = {Delta[-1]*100:.1f} cm',
                     fontsize=13)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)

        # Beta values
        betas = self.compute_beta_values()
        within_limit, beta_N, beta_N_max = self.check_troyon_limit()

        beta_names = ['β_tor', 'β_pol', 'β_total', 'β_N']
        beta_vals = [betas['beta_tor']*100, betas['beta_pol']*100,
                    betas['beta_total']*100, betas['beta_N']]
        colors_beta = ['blue', 'red', 'green', 'purple']

        bars = ax4.bar(beta_names, beta_vals, color=colors_beta, alpha=0.7)
        ax4.axhline(y=beta_N_max, color='orange', linestyle='--',
                   linewidth=2, label=f'Troyon limit (β_N={beta_N_max})')
        ax4.set_ylabel('Value (%)', fontsize=11)
        ax4.set_title('Beta Values', fontsize=13)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend(fontsize=10)

        # Add values on bars
        for bar, val in zip(bars, beta_vals):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=10)

        # Status text
        status = "WITHIN LIMIT" if within_limit else "EXCEEDS LIMIT"
        color = 'green' if within_limit else 'red'
        ax4.text(0.5, 0.95, f'Troyon: {status}',
                transform=ax4.transAxes, ha='center', va='top',
                fontsize=11, weight='bold', color=color,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        return fig


def main():
    """
    Main function demonstrating tokamak equilibrium calculation.
    """
    print("=" * 60)
    print("Tokamak Equilibrium Calculation")
    print("=" * 60)

    # ITER-like parameters
    tokamak = TokamakEquilibrium(
        R0=6.2,      # Major radius (m)
        a=2.0,       # Minor radius (m)
        B_tor=5.3,   # Toroidal field (T)
        I_p=15.0,    # Plasma current (MA)
        beta_0=0.025 # Central beta
    )

    print(f"\nParameters:")
    print(f"  Major radius R₀ = {tokamak.R0:.1f} m")
    print(f"  Minor radius a = {tokamak.a:.1f} m")
    print(f"  Aspect ratio A = {tokamak.R0/tokamak.a:.1f}")
    print(f"  Toroidal field = {tokamak.B_tor:.1f} T")
    print(f"  Plasma current = {tokamak.I_p/1e6:.1f} MA")

    # Compute equilibrium
    q, B_pol = tokamak.compute_q_profile()
    betas = tokamak.compute_beta_values()
    within_limit, beta_N, beta_N_max = tokamak.check_troyon_limit()

    print(f"\nEquilibrium properties:")
    print(f"  q(0) = {q[0]:.2f}")
    print(f"  q(a) = {q[-1]:.2f}")
    print(f"  β_toroidal = {betas['beta_tor']*100:.2f}%")
    print(f"  β_poloidal = {betas['beta_pol']*100:.2f}%")
    print(f"  β_N = {betas['beta_N']:.2f}")
    print(f"\nTroyon limit check:")
    print(f"  β_N = {beta_N:.2f} < {beta_N_max:.1f}: {within_limit}")

    # Plot
    tokamak.plot_equilibrium()

    plt.savefig('/tmp/tokamak_equilibrium.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to /tmp/tokamak_equilibrium.png")

    plt.show()


if __name__ == "__main__":
    main()
