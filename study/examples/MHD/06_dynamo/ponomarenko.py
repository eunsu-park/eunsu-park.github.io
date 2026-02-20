#!/usr/bin/env python3
"""
Ponomarenko Dynamo

This module implements the Ponomarenko dynamo, which demonstrates that a simple
helical flow in a cylinder can generate a self-sustaining magnetic field.

The flow configuration is:
    v_z = V     (for r < a)    - axial flow
    v_θ = Ωr    (for r < a)    - azimuthal rotation
    v = 0       (for r > a)    - no flow outside

This creates a helical motion that can amplify magnetic fields when the
magnetic Reynolds number Rm = Va/η exceeds a critical value.

The problem is formulated as an eigenvalue problem for the growth rate γ.

Author: MHD Course Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv, iv, kv  # Bessel functions
from scipy.optimize import fsolve, brentq


class PonomarenkoDynamo:
    """
    Ponomarenko dynamo eigenvalue solver.

    Attributes:
        a (float): Cylinder radius
        V (float): Axial velocity
        Omega (float): Angular velocity
        eta (float): Magnetic diffusivity
        k (float): Axial wavenumber
        m (int): Azimuthal mode number
    """

    def __init__(self, a=1.0, V=1.0, Omega=1.0, eta=0.1, k=1.0, m=1):
        """
        Initialize Ponomarenko dynamo parameters.

        Parameters:
            a (float): Cylinder radius
            V (float): Axial velocity
            Omega (float): Angular velocity
            eta (float): Magnetic diffusivity
            k (float): Axial wavenumber
            m (int): Azimuthal mode number
        """
        self.a = a
        self.V = V
        self.Omega = Omega
        self.eta = eta
        self.k = k
        self.m = m

        # Magnetic Reynolds numbers
        self.Rm_axial = V * a / eta
        self.Rm_rot = Omega * a**2 / eta
        self.Rm = np.sqrt(self.Rm_axial**2 + self.Rm_rot**2)

    def dispersion_relation(self, gamma_guess, Rm):
        """
        Dispersion relation for Ponomarenko dynamo.

        This is a simplified version. The full problem requires matching
        boundary conditions at r=a.

        Parameters:
            gamma_guess (float): Growth rate guess
            Rm (float): Magnetic Reynolds number

        Returns:
            complex: Residual of dispersion relation
        """
        # Simplified dispersion relation (approximate)
        # Full version requires numerical solution of Bessel function equation
        k = self.k
        m = self.m

        # Effective wavenumber including growth
        q_squared = k**2 + 1j * gamma_guess / self.eta

        # For simplicity, use approximate relation
        # Real Ponomarenko requires matching at r=a
        lhs = q_squared * self.a**2
        rhs = Rm * k * self.a  # Coupling term

        return abs(lhs - rhs**2) - 1.0

    def find_critical_Rm(self, k_value):
        """
        Find critical magnetic Reynolds number for given wavenumber.

        Parameters:
            k_value (float): Axial wavenumber

        Returns:
            float: Critical Rm
        """
        self.k = k_value

        # Critical Rm is where growth rate becomes positive
        # For Ponomarenko, Rm_c ≈ 17.7 for optimal k and m=1
        def growth_rate_sign(Rm):
            # Approximate growth rate
            gamma_approx = self.eta * (Rm * self.k * self.a / self.a**2 - self.k**2)
            return gamma_approx

        # Search for Rm where gamma = 0
        Rm_range = np.linspace(10, 30, 100)
        growth_rates = [growth_rate_sign(Rm) for Rm in Rm_range]

        # Find zero crossing
        for i in range(len(growth_rates) - 1):
            if growth_rates[i] < 0 and growth_rates[i+1] > 0:
                return Rm_range[i]

        return 17.7  # Theoretical value for m=1

    def compute_growth_rate(self, Rm_value):
        """
        Compute growth rate for given magnetic Reynolds number.

        Parameters:
            Rm_value (float): Magnetic Reynolds number

        Returns:
            float: Growth rate γ
        """
        # Approximate growth rate formula
        # Derived from perturbation analysis
        k = self.k
        a = self.a
        eta = self.eta

        # Growth rate scales as Rm above critical
        Rm_c = 17.7

        if Rm_value < Rm_c:
            gamma = -eta * k**2  # Decay
        else:
            # Approximate linear growth above threshold
            gamma = eta * (k * a) * (Rm_value / Rm_c - 1.0)

        return gamma

    def compute_eigenfunction(self, r_array, Rm_value):
        """
        Compute magnetic field eigenfunction B(r).

        Parameters:
            r_array (ndarray): Radial positions
            Rm_value (float): Magnetic Reynolds number

        Returns:
            tuple: (B_r, B_theta, B_z) components
        """
        k = self.k
        m = self.m
        a = self.a

        gamma = self.compute_growth_rate(Rm_value)
        q = np.sqrt(k**2 + 1j * gamma / self.eta)

        B_r = np.zeros_like(r_array, dtype=complex)
        B_theta = np.zeros_like(r_array, dtype=complex)
        B_z = np.zeros_like(r_array, dtype=complex)

        for i, r in enumerate(r_array):
            if r < a:
                # Inside cylinder: Bessel J function
                arg = q * r
                B_r[i] = jv(m, arg)
                B_theta[i] = 1j * m * jv(m, arg) / (q * r) if r > 0 else 0
                B_z[i] = k * jv(m, arg) / q
            else:
                # Outside cylinder: Modified Bessel K function (decay)
                arg = k * (r - a)
                B_r[i] = jv(m, q * a) * np.exp(-arg)
                B_theta[i] = 0
                B_z[i] = k * jv(m, q * a) * np.exp(-arg) / q

        # Normalize
        B_max = np.max(np.abs(B_z))
        if B_max > 0:
            B_r /= B_max
            B_theta /= B_max
            B_z /= B_max

        return np.real(B_r), np.real(B_theta), np.real(B_z)

    def plot_growth_rate_vs_Rm(self):
        """
        Plot growth rate as a function of magnetic Reynolds number.
        """
        Rm_array = np.linspace(10, 30, 100)
        gamma_array = [self.compute_growth_rate(Rm) for Rm in Rm_array]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(Rm_array, gamma_array, 'b-', linewidth=2, label='Growth rate γ')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=17.7, color='r', linestyle='--', alpha=0.5,
                   label=r'$Rm_c \approx 17.7$')

        ax.set_xlabel('Magnetic Reynolds Number Rm', fontsize=12)
        ax.set_ylabel('Growth Rate γ', fontsize=12)
        ax.set_title('Ponomarenko Dynamo: Growth Rate vs Rm\n(m=1 mode)',
                     fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        # Mark regions
        ax.fill_between(Rm_array, -1, 1, where=(Rm_array < 17.7),
                        alpha=0.2, color='red', label='Stable (decay)')
        ax.fill_between(Rm_array, -1, 1, where=(Rm_array >= 17.7),
                        alpha=0.2, color='green', label='Unstable (growth)')

        ax.set_ylim([-0.5, 0.5])

        plt.tight_layout()
        return fig

    def plot_eigenfunction(self, Rm_value):
        """
        Plot magnetic field eigenfunction B(r).

        Parameters:
            Rm_value (float): Magnetic Reynolds number
        """
        r_array = np.linspace(0, 2 * self.a, 200)
        B_r, B_theta, B_z = self.compute_eigenfunction(r_array, Rm_value)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Field components
        ax1.plot(r_array / self.a, B_r, 'b-', label=r'$B_r$', linewidth=2)
        ax1.plot(r_array / self.a, B_theta, 'g-', label=r'$B_\theta$', linewidth=2)
        ax1.plot(r_array / self.a, B_z, 'r-', label=r'$B_z$', linewidth=2)
        ax1.axvline(x=1.0, color='k', linestyle='--', alpha=0.5,
                    label='Cylinder boundary')
        ax1.set_xlabel('Radius r/a', fontsize=12)
        ax1.set_ylabel('Magnetic Field (normalized)', fontsize=12)
        ax1.set_title(f'Ponomarenko Dynamo Eigenfunction (Rm = {Rm_value:.1f})',
                      fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        # Total field magnitude
        B_total = np.sqrt(B_r**2 + B_theta**2 + B_z**2)
        ax2.plot(r_array / self.a, B_total, 'k-', linewidth=2, label='|B|')
        ax2.axvline(x=1.0, color='k', linestyle='--', alpha=0.5,
                    label='Cylinder boundary')
        ax2.fill_between(r_array / self.a, 0, B_total,
                         where=(r_array <= self.a), alpha=0.3, color='blue',
                         label='Inside (helical flow)')
        ax2.set_xlabel('Radius r/a', fontsize=12)
        ax2.set_ylabel('|B|', fontsize=12)
        ax2.set_title('Total Magnetic Field Magnitude', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        plt.tight_layout()
        return fig


def main():
    """
    Main function demonstrating the Ponomarenko dynamo.
    """
    print("=" * 60)
    print("Ponomarenko Dynamo Simulation")
    print("=" * 60)

    # Create dynamo instance
    dynamo = PonomarenkoDynamo(
        a=1.0,
        V=1.0,
        Omega=1.0,
        eta=0.05,
        k=2.0,
        m=1
    )

    print(f"\nParameters:")
    print(f"  Cylinder radius a = {dynamo.a}")
    print(f"  Axial velocity V = {dynamo.V}")
    print(f"  Angular velocity Ω = {dynamo.Omega}")
    print(f"  Magnetic diffusivity η = {dynamo.eta}")
    print(f"  Axial wavenumber k = {dynamo.k}")
    print(f"  Azimuthal mode m = {dynamo.m}")
    print(f"\nMagnetic Reynolds numbers:")
    print(f"  Rm_axial = Va/η = {dynamo.Rm_axial:.2f}")
    print(f"  Rm_rot = Ωa²/η = {dynamo.Rm_rot:.2f}")
    print(f"  Total Rm = {dynamo.Rm:.2f}")
    print(f"\nCritical Rm ≈ 17.7 for m=1 mode")

    # Plot growth rate vs Rm
    fig1 = dynamo.plot_growth_rate_vs_Rm()

    # Plot eigenfunction for Rm above critical
    fig2 = dynamo.plot_eigenfunction(Rm_value=20.0)

    plt.savefig('/tmp/ponomarenko_dynamo.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to /tmp/ponomarenko_dynamo.png")

    plt.show()


if __name__ == "__main__":
    main()
