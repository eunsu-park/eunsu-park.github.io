#!/usr/bin/env python3
"""
Turbulent Small-Scale Dynamo

This module simulates a small-scale (turbulent) dynamo where magnetic fields
are amplified by turbulent motions at scales smaller than the energy injection
scale.

Key features:
- Kazantsev spectrum: E_B(k) ∝ k^{3/2} during kinematic phase
- Exponential growth phase followed by saturation
- Stochastic flow model (Ornstein-Uhlenbeck process)
- Magnetic energy evolution and spectral analysis

The small-scale dynamo is important in astrophysical contexts such as
the interstellar medium and early universe.

Author: MHD Course Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class TurbulentDynamo:
    """
    Small-scale turbulent dynamo model.

    This uses a simplified stochastic differential equation approach
    to model magnetic field amplification by turbulent flows.

    Attributes:
        n_modes (int): Number of Fourier modes
        k_modes (ndarray): Wavenumber array
        Re (float): Reynolds number
        Rm (float): Magnetic Reynolds number
        Pm (float): Magnetic Prandtl number (Rm/Re)
    """

    def __init__(self, n_modes=50, k_min=1.0, k_max=50.0,
                 Re=1000.0, Pm=1.0):
        """
        Initialize turbulent dynamo.

        Parameters:
            n_modes (int): Number of Fourier modes
            k_min (float): Minimum wavenumber
            k_max (float): Maximum wavenumber
            Re (float): Reynolds number
            Pm (float): Magnetic Prandtl number
        """
        self.n_modes = n_modes
        self.k_modes = np.logspace(np.log10(k_min), np.log10(k_max), n_modes)
        self.Re = Re
        self.Pm = Pm
        self.Rm = Re * Pm

        # Turbulent velocity amplitude (from Kolmogorov)
        self.u_rms = 1.0
        self.eta = self.u_rms / self.Rm  # Magnetic diffusivity
        self.nu = self.u_rms / self.Re   # Kinematic viscosity

        # Eddy turnover time
        self.tau_eddy = 1.0

        # Initialize magnetic energy spectrum
        self.E_B = 1e-6 * np.ones(n_modes)  # Seed field

        # Time history
        self.time_history = []
        self.energy_history = []
        self.spectrum_history = []

    def kolmogorov_spectrum(self, k):
        """
        Kolmogorov turbulent energy spectrum E_v(k) ∝ k^{-5/3}.

        Parameters:
            k (ndarray): Wavenumber

        Returns:
            ndarray: Velocity energy spectrum
        """
        k0 = 1.0  # Energy injection scale
        epsilon = 1.0  # Energy dissipation rate
        C_K = 1.5  # Kolmogorov constant

        E_v = C_K * epsilon**(2/3) * k**(-5/3)

        # Add exponential cutoff at dissipation scale
        k_d = (epsilon / self.nu**3)**(1/4)  # Kolmogorov scale
        E_v *= np.exp(-k / k_d)

        return E_v

    def kazantsev_spectrum(self, k):
        """
        Kazantsev magnetic energy spectrum E_B(k) ∝ k^{3/2}.

        This is the theoretical prediction for small-scale dynamo
        in the kinematic regime.

        Parameters:
            k (ndarray): Wavenumber

        Returns:
            ndarray: Magnetic energy spectrum
        """
        # Kazantsev spectrum
        k0 = self.k_modes[0]
        C_K = 0.1

        E_B = C_K * (k / k0)**(3/2)

        # Cutoff at resistive scale
        k_eta = np.sqrt(self.u_rms / self.eta)
        E_B *= np.exp(-k / k_eta)

        return E_B

    def growth_rate(self, k):
        """
        Compute growth rate γ(k) for each mode.

        In kinematic regime: γ(k) ≈ constant (Lyapunov exponent)
        With diffusion: γ(k) = γ₀ - η k²

        Parameters:
            k (ndarray): Wavenumber

        Returns:
            ndarray: Growth rate
        """
        # Lyapunov exponent (stretching rate)
        gamma_0 = 0.1 * self.u_rms / self.tau_eddy

        # Diffusive damping
        gamma = gamma_0 - self.eta * k**2

        return gamma

    def rhs(self, E_B, t):
        """
        Right-hand side for magnetic energy evolution.

        dE_B/dt = 2γ(k)E_B - saturation term

        Parameters:
            E_B (ndarray): Magnetic energy spectrum
            t (float): Time

        Returns:
            ndarray: Time derivative
        """
        gamma = self.growth_rate(self.k_modes)

        # Velocity energy for saturation
        E_v = self.kolmogorov_spectrum(self.k_modes)

        # Kinematic growth
        dE_dt = 2 * gamma * E_B

        # Saturation: when E_B ~ E_v, growth slows
        # Simple saturation model
        saturation_factor = E_v / (E_v + E_B)
        dE_dt *= saturation_factor

        # Add stochastic forcing (simplified)
        forcing = 1e-4 * np.random.randn(self.n_modes)
        dE_dt += forcing

        return dE_dt

    def run_simulation(self, t_end=50.0, n_steps=500):
        """
        Run the turbulent dynamo simulation.

        Parameters:
            t_end (float): End time
            n_steps (int): Number of time steps
        """
        t_array = np.linspace(0, t_end, n_steps)

        print("Running turbulent dynamo simulation...")
        print(f"Magnetic Reynolds number Rm = {self.Rm:.1f}")
        print(f"Magnetic Prandtl number Pm = {self.Pm:.1f}")

        # Storage
        self.time_history = []
        self.energy_history = []
        self.spectrum_history = []

        # Time stepping (using manual Euler due to stochasticity)
        dt = t_end / n_steps
        E_B = self.E_B.copy()

        for i, t in enumerate(t_array):
            # Store data
            self.time_history.append(t)
            self.energy_history.append(np.sum(E_B))
            self.spectrum_history.append(E_B.copy())

            # Euler step
            dE_dt = self.rhs(E_B, t)
            E_B += dt * dE_dt

            # Ensure positivity
            E_B = np.maximum(E_B, 1e-10)

            if i % 100 == 0:
                total_energy = np.sum(E_B)
                print(f"Step {i}/{n_steps}, t={t:.2f}, E_mag={total_energy:.4e}")

        self.E_B = E_B
        self.spectrum_history = np.array(self.spectrum_history)
        print("Simulation complete!")

    def plot_energy_evolution(self):
        """
        Plot magnetic energy evolution.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Compute growth rate
        growth_rate_theory = self.growth_rate(self.k_modes[0])

        ax.semilogy(self.time_history, self.energy_history, 'b-',
                    linewidth=2, label='Total magnetic energy')

        # Theoretical exponential growth
        if len(self.time_history) > 0:
            E0 = self.energy_history[0]
            t_array = np.array(self.time_history)
            E_theory = E0 * np.exp(2 * 0.1 * t_array / self.tau_eddy)
            ax.semilogy(t_array, E_theory, 'r--', alpha=0.6,
                       label='Exponential growth (kinematic)')

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Total Magnetic Energy', fontsize=12)
        ax.set_title('Turbulent Dynamo: Magnetic Energy Growth', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        # Mark phases
        ax.text(0.5, 0.95, 'Kinematic phase → Saturation',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_spectrum_evolution(self):
        """
        Plot magnetic energy spectrum evolution.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Spectrum at different times
        n_times = min(5, len(self.spectrum_history))
        time_indices = np.linspace(0, len(self.spectrum_history)-1,
                                    n_times, dtype=int)

        for idx in time_indices:
            t = self.time_history[idx]
            spectrum = self.spectrum_history[idx]
            ax1.loglog(self.k_modes, spectrum, '-', linewidth=2,
                      label=f't = {t:.1f}')

        # Theoretical spectra
        kazantsev = self.kazantsev_spectrum(self.k_modes)
        ax1.loglog(self.k_modes, kazantsev * np.max(self.spectrum_history[-1]) /
                  np.max(kazantsev), 'k--', linewidth=2,
                  alpha=0.6, label=r'Kazantsev $k^{3/2}$')

        ax1.set_xlabel('Wavenumber k', fontsize=12)
        ax1.set_ylabel(r'$E_B(k)$', fontsize=12)
        ax1.set_title('Magnetic Energy Spectrum Evolution', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Final spectrum with comparison
        E_v = self.kolmogorov_spectrum(self.k_modes)
        E_v *= np.max(self.E_B) / np.max(E_v)  # Normalize for comparison

        ax2.loglog(self.k_modes, self.E_B, 'b-', linewidth=2,
                  label='Magnetic energy')
        ax2.loglog(self.k_modes, E_v, 'r--', linewidth=2, alpha=0.6,
                  label=r'Velocity energy $k^{-5/3}$')
        ax2.set_xlabel('Wavenumber k', fontsize=12)
        ax2.set_ylabel('Energy', fontsize=12)
        ax2.set_title('Final State: Magnetic vs Kinetic Energy', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        plt.tight_layout()
        return fig


def main():
    """
    Main function demonstrating turbulent dynamo.
    """
    print("=" * 60)
    print("Turbulent Small-Scale Dynamo Simulation")
    print("=" * 60)

    # Create dynamo
    dynamo = TurbulentDynamo(
        n_modes=50,
        k_min=1.0,
        k_max=100.0,
        Re=1000.0,
        Pm=1.0
    )

    # Run simulation
    dynamo.run_simulation(t_end=50.0, n_steps=1000)

    # Plot results
    dynamo.plot_energy_evolution()
    dynamo.plot_spectrum_evolution()

    plt.savefig('/tmp/turbulent_dynamo.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to /tmp/turbulent_dynamo.png")

    plt.show()


if __name__ == "__main__":
    main()
