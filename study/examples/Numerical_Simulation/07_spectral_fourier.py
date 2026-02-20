#!/usr/bin/env python3
"""
Fourier Spectral Method for PDEs
=================================

This module demonstrates the Fourier spectral method for solving partial
differential equations (PDEs) using Fast Fourier Transform (FFT).

Examples:
    1. 1D Heat Equation: u_t = nu * u_xx
    2. 1D Burgers Equation: u_t + u * u_x = nu * u_xx (nonlinear)

Key Concepts:
    - Spectral differentiation using FFT
    - Time integration with RK4
    - Dealiasing using the 3/2 rule for nonlinear terms
    - High-order accuracy in space

Author: Educational example for Numerical Simulation
License: MIT
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class FourierSpectralSolver:
    """
    Fourier spectral method solver for 1D PDEs with periodic boundary conditions.

    Attributes:
        N (int): Number of spatial grid points
        L (float): Domain length [0, L]
        nu (float): Viscosity/diffusion coefficient
        x (ndarray): Spatial grid points
        k (ndarray): Wavenumber array for spectral differentiation
    """

    def __init__(self, N, L, nu):
        """
        Initialize the spectral solver.

        Args:
            N (int): Number of grid points (should be even for dealiasing)
            L (float): Domain length
            nu (float): Viscosity coefficient
        """
        self.N = N
        self.L = L
        self.nu = nu

        # Spatial grid (periodic, exclude endpoint)
        self.x = np.linspace(0, L, N, endpoint=False)

        # Wavenumbers for FFT (properly ordered for fft)
        self.k = 2 * np.pi * fftfreq(N, d=L/N)

    def spectral_derivative(self, u_hat, order=1):
        """
        Compute spectral derivative in Fourier space.

        For a function u(x) with Fourier transform û(k):
            d^n u/dx^n  <-->  (ik)^n û(k)

        Args:
            u_hat (ndarray): Fourier coefficients of u
            order (int): Derivative order

        Returns:
            ndarray: Fourier coefficients of du/dx^order
        """
        return (1j * self.k)**order * u_hat

    def dealias_3_2(self, u_hat):
        """
        Apply 3/2 rule dealiasing to prevent aliasing errors in nonlinear terms.

        The 3/2 rule: zero out the middle third of Fourier modes when computing
        nonlinear products.

        Args:
            u_hat (ndarray): Fourier coefficients

        Returns:
            ndarray: Dealiased Fourier coefficients
        """
        u_hat_dealiased = u_hat.copy()
        # Zero out modes in middle third
        N = len(u_hat)
        u_hat_dealiased[N//3:(2*N)//3] = 0
        return u_hat_dealiased

    def heat_equation_rhs(self, u_hat):
        """
        Right-hand side for heat equation: u_t = nu * u_xx

        In Fourier space: û_t = -nu * k^2 * û

        Args:
            u_hat (ndarray): Fourier coefficients of u

        Returns:
            ndarray: Time derivative in Fourier space
        """
        return -self.nu * self.k**2 * u_hat

    def burgers_equation_rhs(self, u_hat, use_dealiasing=True):
        """
        Right-hand side for Burgers equation: u_t + u * u_x = nu * u_xx

        Args:
            u_hat (ndarray): Fourier coefficients of u
            use_dealiasing (bool): Whether to apply 3/2 rule dealiasing

        Returns:
            ndarray: Time derivative in Fourier space
        """
        # Linear term: diffusion
        diffusion = -self.nu * self.k**2 * u_hat

        # Nonlinear term: -u * u_x
        # Transform back to physical space
        u = ifft(u_hat).real

        # Compute u * du/dx in physical space
        u_x_hat = self.spectral_derivative(u_hat, order=1)
        u_x = ifft(u_x_hat).real
        nonlinear = -u * u_x

        # Transform to Fourier space and apply dealiasing
        nonlinear_hat = fft(nonlinear)
        if use_dealiasing:
            nonlinear_hat = self.dealias_3_2(nonlinear_hat)

        return diffusion + nonlinear_hat

    def rk4_step(self, u_hat, dt, equation='heat'):
        """
        Fourth-order Runge-Kutta time integration step.

        Args:
            u_hat (ndarray): Current Fourier coefficients
            dt (float): Time step
            equation (str): 'heat' or 'burgers'

        Returns:
            ndarray: Updated Fourier coefficients
        """
        if equation == 'heat':
            rhs = self.heat_equation_rhs
        elif equation == 'burgers':
            rhs = self.burgers_equation_rhs
        else:
            raise ValueError("equation must be 'heat' or 'burgers'")

        # RK4 stages
        k1 = rhs(u_hat)
        k2 = rhs(u_hat + 0.5 * dt * k1)
        k3 = rhs(u_hat + 0.5 * dt * k2)
        k4 = rhs(u_hat + dt * k3)

        # Update
        u_hat_new = u_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return u_hat_new

    def solve(self, u0, T, dt, equation='heat'):
        """
        Solve the PDE from t=0 to t=T.

        Args:
            u0 (ndarray): Initial condition in physical space
            T (float): Final time
            dt (float): Time step
            equation (str): 'heat' or 'burgers'

        Returns:
            tuple: (time_points, solution_history)
        """
        # Number of time steps
        Nt = int(T / dt)

        # Initialize
        u_hat = fft(u0)

        # Storage
        t_history = [0]
        u_history = [u0.copy()]

        # Time integration
        for n in range(Nt):
            u_hat = self.rk4_step(u_hat, dt, equation)

            # Store (every 10 steps to save memory)
            if (n + 1) % 10 == 0 or n == Nt - 1:
                u = ifft(u_hat).real
                t_history.append((n + 1) * dt)
                u_history.append(u.copy())

        return np.array(t_history), np.array(u_history)


def example_heat_equation():
    """
    Solve the 1D heat equation with a Gaussian initial condition.

    PDE: u_t = nu * u_xx, x in [0, 2π], periodic BCs
    IC:  u(x, 0) = exp(-10 * (x - π)^2)

    Analytical solution exists for comparison.
    """
    print("=" * 60)
    print("Example 1: Heat Equation")
    print("=" * 60)

    # Parameters
    N = 128          # Grid points
    L = 2 * np.pi    # Domain length
    nu = 0.1         # Diffusion coefficient
    T = 2.0          # Final time
    dt = 0.01        # Time step

    # Initialize solver
    solver = FourierSpectralSolver(N, L, nu)

    # Initial condition: Gaussian
    u0 = np.exp(-10 * (solver.x - np.pi)**2)

    # Solve
    print(f"Solving heat equation with N={N} points, dt={dt}")
    t_history, u_history = solver.solve(u0, T, dt, equation='heat')
    print(f"Computed {len(t_history)} time snapshots")

    # Analytical solution for comparison (Gaussian diffusion)
    def analytical_heat(x, t, nu):
        sigma0_sq = 1.0 / (2 * 10)  # Initial variance
        sigma_t_sq = sigma0_sq + 2 * nu * t
        return np.sqrt(sigma0_sq / sigma_t_sq) * np.exp(-((x - np.pi)**2) / (2 * sigma_t_sq))

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Evolution over time
    snapshot_indices = [0, len(t_history)//3, 2*len(t_history)//3, -1]
    for idx in snapshot_indices:
        ax1.plot(solver.x, u_history[idx], label=f't = {t_history[idx]:.2f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x, t)')
    ax1.set_title('Heat Equation Evolution (Spectral Method)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Comparison with analytical solution at final time
    u_analytical = analytical_heat(solver.x, t_history[-1], nu)
    ax2.plot(solver.x, u_history[-1], 'b-', label='Spectral', linewidth=2)
    ax2.plot(solver.x, u_analytical, 'r--', label='Analytical', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x, T)')
    ax2.set_title(f'Comparison at t = {T}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Compute error
    error = np.linalg.norm(u_history[-1] - u_analytical) / np.linalg.norm(u_analytical)
    print(f"Relative L2 error: {error:.2e}")

    plt.tight_layout()
    plt.savefig('/tmp/heat_equation_spectral.png', dpi=150)
    print("Saved plot to /tmp/heat_equation_spectral.png")
    plt.show()


def example_burgers_equation():
    """
    Solve the 1D viscous Burgers equation.

    PDE: u_t + u * u_x = nu * u_xx, x in [0, 2π], periodic BCs
    IC:  u(x, 0) = sin(x) + 0.5 * sin(2x)

    This demonstrates shock formation and dissipation.
    """
    print("\n" + "=" * 60)
    print("Example 2: Burgers Equation")
    print("=" * 60)

    # Parameters
    N = 256          # Grid points (more for shock resolution)
    L = 2 * np.pi    # Domain length
    nu = 0.05        # Viscosity
    T = 3.0          # Final time
    dt = 0.005       # Time step (smaller for stability)

    # Initialize solver
    solver = FourierSpectralSolver(N, L, nu)

    # Initial condition: smooth wave
    u0 = np.sin(solver.x) + 0.5 * np.sin(2 * solver.x)

    # Solve
    print(f"Solving Burgers equation with N={N} points, dt={dt}")
    t_history, u_history = solver.solve(u0, T, dt, equation='burgers')
    print(f"Computed {len(t_history)} time snapshots")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top: Evolution over time
    snapshot_indices = [0, len(t_history)//4, len(t_history)//2, 3*len(t_history)//4, -1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_indices)))

    for i, idx in enumerate(snapshot_indices):
        ax1.plot(solver.x, u_history[idx], color=colors[i],
                label=f't = {t_history[idx]:.2f}', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x, t)')
    ax1.set_title('Burgers Equation: Shock Formation and Dissipation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom: Space-time contour plot
    X, T_grid = np.meshgrid(solver.x, t_history)
    contour = ax2.contourf(X, T_grid, u_history, levels=20, cmap='RdBu_r')
    plt.colorbar(contour, ax=ax2, label='u(x, t)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('Space-Time Evolution')

    plt.tight_layout()
    plt.savefig('/tmp/burgers_equation_spectral.png', dpi=150)
    print("Saved plot to /tmp/burgers_equation_spectral.png")
    plt.show()


if __name__ == "__main__":
    print("Fourier Spectral Method for PDEs")
    print("=" * 60)
    print("This script demonstrates spectral methods using FFT for:")
    print("  1. Heat equation (linear diffusion)")
    print("  2. Burgers equation (nonlinear advection + diffusion)")
    print()

    # Run examples
    example_heat_equation()
    example_burgers_equation()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Spectral methods provide exponential convergence for smooth solutions")
    print("  - FFT enables O(N log N) computation of derivatives")
    print("  - Dealiasing (3/2 rule) prevents aliasing in nonlinear terms")
    print("  - RK4 time integration maintains high temporal accuracy")
    print("=" * 60)
