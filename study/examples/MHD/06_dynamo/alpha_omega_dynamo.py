#!/usr/bin/env python3
"""
Alpha-Omega Mean-Field Dynamo Model

This module implements a 1D α-Ω mean-field dynamo model, which is used to
understand magnetic field generation in rotating systems like stars and planets.

The governing equations are:
    ∂A/∂t = αB + η∂²A/∂r²    (poloidal field evolution)
    ∂B/∂t = S(∂A/∂r) + η∂²B/∂r²    (toroidal field evolution)

where:
    A = poloidal field potential
    B = toroidal field
    α = alpha effect (helicity parameter)
    η = magnetic diffusivity
    S = shear parameter (differential rotation, Ω-effect)

The dynamo number D = αS/η² determines whether the dynamo is self-sustaining.

Author: MHD Course Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class AlphaOmegaDynamo:
    """
    1D α-Ω mean-field dynamo model solver.

    Attributes:
        nr (int): Number of radial grid points
        r (ndarray): Radial grid
        dr (float): Grid spacing
        alpha (float): Alpha effect strength
        eta (float): Magnetic diffusivity
        shear (float): Shear parameter S
        dynamo_number (float): D = αS/η²
    """

    def __init__(self, nr=100, r_min=0.5, r_max=1.0, alpha=1.0,
                 eta=0.1, shear=10.0):
        """
        Initialize the dynamo model.

        Parameters:
            nr (int): Number of radial grid points
            r_min (float): Inner radius
            r_max (float): Outer radius
            alpha (float): Alpha effect strength
            eta (float): Magnetic diffusivity
            shear (float): Shear parameter
        """
        self.nr = nr
        self.r = np.linspace(r_min, r_max, nr)
        self.dr = self.r[1] - self.r[0]
        self.alpha = alpha
        self.eta = eta
        self.shear = shear
        self.dynamo_number = (alpha * shear) / (eta**2)

        # Initialize fields
        self.A = np.zeros(nr)  # Poloidal potential
        self.B = np.zeros(nr)  # Toroidal field

        # Time history for butterfly diagram
        self.time_history = []
        self.B_history = []

    def diffusion_operator(self):
        """
        Construct the finite difference diffusion operator ∂²/∂r².

        Returns:
            scipy.sparse matrix: Diffusion operator
        """
        # Second derivative with periodic boundary conditions
        diag_main = -2.0 * np.ones(self.nr) / self.dr**2
        diag_off = np.ones(self.nr - 1) / self.dr**2

        # Periodic boundaries
        data = [diag_main, diag_off, diag_off]
        offsets = [0, 1, -1]
        D = diags(data, offsets, shape=(self.nr, self.nr), format='csr')

        # Wrap around for periodic BC
        D[0, -1] = 1.0 / self.dr**2
        D[-1, 0] = 1.0 / self.dr**2

        return D

    def gradient(self, f):
        """
        Compute ∂f/∂r using centered differences.

        Parameters:
            f (ndarray): Field to differentiate

        Returns:
            ndarray: Gradient of f
        """
        df = np.zeros_like(f)
        df[1:-1] = (f[2:] - f[:-2]) / (2 * self.dr)
        # Periodic boundaries
        df[0] = (f[1] - f[-1]) / (2 * self.dr)
        df[-1] = (f[0] - f[-2]) / (2 * self.dr)
        return df

    def step_explicit(self, dt):
        """
        Time step using explicit Euler method.

        Parameters:
            dt (float): Time step
        """
        # Compute spatial derivatives
        d2A_dr2 = self.eta * self.diffusion_operator().dot(self.A)
        d2B_dr2 = self.eta * self.diffusion_operator().dot(self.B)
        dA_dr = self.gradient(self.A)

        # Update equations
        dA_dt = self.alpha * self.B + d2A_dr2
        dB_dt = self.shear * dA_dr + d2B_dr2

        # Euler step
        self.A += dt * dA_dt
        self.B += dt * dB_dt

    def step_implicit(self, dt):
        """
        Time step using implicit method (Crank-Nicolson).

        Parameters:
            dt (float): Time step
        """
        I = np.eye(self.nr)
        D = self.diffusion_operator().toarray()

        # Crank-Nicolson matrices
        LHS_A = I - 0.5 * dt * self.eta * D
        RHS_A = I + 0.5 * dt * self.eta * D

        LHS_B = I - 0.5 * dt * self.eta * D
        RHS_B = I + 0.5 * dt * self.eta * D

        # Source terms
        source_A = dt * self.alpha * self.B
        source_B = dt * self.shear * self.gradient(self.A)

        # Solve linear systems
        self.A = np.linalg.solve(LHS_A, RHS_A @ self.A + source_A)
        self.B = np.linalg.solve(LHS_B, RHS_B @ self.B + source_B)

    def set_initial_condition(self, mode='random'):
        """
        Set initial magnetic field perturbation.

        Parameters:
            mode (str): 'random' or 'sinusoidal'
        """
        if mode == 'random':
            self.A = 0.01 * np.random.randn(self.nr)
            self.B = 0.01 * np.random.randn(self.nr)
        elif mode == 'sinusoidal':
            self.A = 0.01 * np.sin(2 * np.pi * (self.r - self.r[0]) /
                                   (self.r[-1] - self.r[0]))
            self.B = 0.01 * np.cos(2 * np.pi * (self.r - self.r[0]) /
                                   (self.r[-1] - self.r[0]))

    def run_simulation(self, t_end=10.0, dt=0.01, save_interval=10):
        """
        Run the dynamo simulation.

        Parameters:
            t_end (float): End time
            dt (float): Time step
            save_interval (int): Save data every N steps
        """
        n_steps = int(t_end / dt)
        t = 0.0

        print(f"Running α-Ω dynamo simulation...")
        print(f"Dynamo number D = {self.dynamo_number:.2f}")
        print(f"Critical D ~ 10 for oscillatory dynamo")

        for step in range(n_steps):
            self.step_implicit(dt)
            t += dt

            # Save history for butterfly diagram
            if step % save_interval == 0:
                self.time_history.append(t)
                self.B_history.append(self.B.copy())

            if step % 1000 == 0:
                B_max = np.max(np.abs(self.B))
                print(f"Step {step}/{n_steps}, t={t:.2f}, max|B|={B_max:.4f}")

        self.B_history = np.array(self.B_history)
        print("Simulation complete!")

    def plot_butterfly_diagram(self):
        """
        Plot the butterfly diagram (time-radius diagram of toroidal field).
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Butterfly diagram
        T, R = np.meshgrid(self.time_history, self.r)
        im = ax1.contourf(T, R, self.B_history.T, levels=20, cmap='RdBu_r')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Radius r')
        ax1.set_title(f'Butterfly Diagram (Toroidal Field B)\nD = {self.dynamo_number:.2f}')
        plt.colorbar(im, ax=ax1, label='B')

        # Dynamo wave propagation
        ax1.text(0.02, 0.98,
                 'Dynamo wave propagates\ndue to α-Ω coupling',
                 transform=ax1.transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Time series at mid-radius
        mid_idx = self.nr // 2
        ax2.plot(self.time_history, self.B_history[:, mid_idx], 'b-', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('B')
        ax2.set_title(f'Toroidal Field at r = {self.r[mid_idx]:.2f}')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        plt.tight_layout()
        return fig

    def plot_field_profiles(self):
        """
        Plot current field profiles.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.r, self.A, 'b-', linewidth=2, label='A (poloidal)')
        ax1.set_xlabel('Radius r')
        ax1.set_ylabel('A')
        ax1.set_title('Poloidal Field Potential')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(self.r, self.B, 'r-', linewidth=2, label='B (toroidal)')
        ax2.set_xlabel('Radius r')
        ax2.set_ylabel('B')
        ax2.set_title('Toroidal Field')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        return fig


def main():
    """
    Main function demonstrating the α-Ω dynamo model.
    """
    # Create dynamo instance
    dynamo = AlphaOmegaDynamo(
        nr=100,
        r_min=0.5,
        r_max=1.0,
        alpha=1.0,
        eta=0.05,
        shear=15.0
    )

    # Set initial condition
    dynamo.set_initial_condition(mode='random')

    # Run simulation
    dynamo.run_simulation(t_end=20.0, dt=0.005, save_interval=20)

    # Plot results
    dynamo.plot_butterfly_diagram()
    dynamo.plot_field_profiles()

    plt.savefig('/tmp/alpha_omega_dynamo.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to /tmp/alpha_omega_dynamo.png")

    plt.show()


if __name__ == "__main__":
    main()
