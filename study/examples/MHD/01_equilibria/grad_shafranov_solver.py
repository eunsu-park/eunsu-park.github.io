#!/usr/bin/env python3
"""
Grad-Shafranov Equation Solver

Solves the Grad-Shafranov equation for axisymmetric toroidal MHD equilibria:
    Δ*ψ = -μ₀ R² dp/dψ - F dF/dψ

where:
    Δ* = R ∂/∂R (1/R ∂/∂R) + ∂²/∂Z²
    ψ is the poloidal flux function
    p(ψ) is the pressure profile
    F(ψ) = R*B_φ is the poloidal current function

Uses iterative Picard iteration and verifies against Solovev analytical solution.

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, linalg as splinalg
from scipy.interpolate import interp1d

# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space


class GradShafranovSolver:
    """
    Solver for the Grad-Shafranov equation.

    Attributes
    ----------
    R : ndarray
        Major radius grid
    Z : ndarray
        Vertical position grid
    psi : ndarray
        Poloidal flux function
    """

    def __init__(self, R_range, Z_range, nR, nZ):
        """
        Initialize solver grid.

        Parameters
        ----------
        R_range : tuple
            (R_min, R_max) in meters
        Z_range : tuple
            (Z_min, Z_max) in meters
        nR, nZ : int
            Number of grid points
        """
        self.R_arr = np.linspace(R_range[0], R_range[1], nR)
        self.Z_arr = np.linspace(Z_range[0], Z_range[1], nZ)
        self.R, self.Z = np.meshgrid(self.R_arr, self.Z_arr, indexing='ij')

        self.dR = self.R_arr[1] - self.R_arr[0]
        self.dZ = self.Z_arr[1] - self.Z_arr[0]

        self.nR = nR
        self.nZ = nZ

        self.psi = np.zeros((nR, nZ))

    def source_term(self, psi):
        """
        Compute source term: S(ψ) = -μ₀ R² dp/dψ - F dF/dψ

        Parameters
        ----------
        psi : ndarray
            Current flux function

        Returns
        -------
        source : ndarray
            Source term
        """
        # Normalize psi to [0, 1] for profile functions
        psi_min, psi_max = psi.min(), psi.max()
        if psi_max > psi_min:
            psi_norm = (psi - psi_min) / (psi_max - psi_min)
        else:
            psi_norm = np.zeros_like(psi)

        # Pressure profile: p(ψ) = p0 * (1 - ψ_norm)²
        p0 = 1e5  # 100 kPa
        dpdpsi = -2 * p0 / (psi_max - psi_min + 1e-10)

        # F profile: F(ψ) = F0 (constant for simplicity)
        F0 = 1.0  # R*B_phi at magnetic axis
        dFdpsi = 0.0

        source = -MU_0 * self.R**2 * dpdpsi - F0 * dFdpsi

        return source

    def delta_star_operator(self, psi):
        """
        Apply Δ* operator: Δ*ψ = R ∂/∂R (1/R ∂ψ/∂R) + ∂²ψ/∂Z²

        Parameters
        ----------
        psi : ndarray
            Input flux function

        Returns
        -------
        result : ndarray
            Result of Δ* operation
        """
        result = np.zeros_like(psi)

        for i in range(1, self.nR - 1):
            for j in range(1, self.nZ - 1):
                R = self.R[i, j]

                # ∂ψ/∂R at i±1/2
                dpsi_dR_plus = (psi[i+1, j] - psi[i, j]) / self.dR
                dpsi_dR_minus = (psi[i, j] - psi[i-1, j]) / self.dR

                # R values at i±1/2
                R_plus = (self.R[i+1, j] + R) / 2
                R_minus = (self.R[i-1, j] + R) / 2

                # R ∂/∂R (1/R ∂ψ/∂R)
                term1 = R * (dpsi_dR_plus / R_plus - dpsi_dR_minus / R_minus) / self.dR

                # ∂²ψ/∂Z²
                term2 = (psi[i, j+1] - 2*psi[i, j] + psi[i, j-1]) / self.dZ**2

                result[i, j] = term1 + term2

        return result

    def solve_picard(self, max_iter=100, tol=1e-6, omega=0.5):
        """
        Solve Grad-Shafranov using Picard iteration.

        Parameters
        ----------
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        omega : float
            Relaxation parameter

        Returns
        -------
        converged : bool
            Whether solver converged
        """
        print("Starting Picard iteration...")

        for iteration in range(max_iter):
            psi_old = self.psi.copy()

            # Compute source term with current psi
            source = self.source_term(self.psi)

            # Solve Δ*ψ = source using finite differences
            # This is a simplified direct update; a real solver would use
            # a linear system solver
            for i in range(1, self.nR - 1):
                for j in range(1, self.nZ - 1):
                    R = self.R[i, j]

                    # Coefficients for finite difference stencil
                    R_plus = (self.R[i+1, j] + R) / 2
                    R_minus = (self.R[i-1, j] + R) / 2

                    a_R = R / (R_plus * self.dR**2)
                    b_R = -R / (R_minus * self.dR**2)
                    c_Z = 1.0 / self.dZ**2

                    coeff = a_R + b_R + 2*c_Z

                    rhs = (a_R * psi_old[i+1, j] +
                           b_R * psi_old[i-1, j] +
                           c_Z * (psi_old[i, j+1] + psi_old[i, j-1]) -
                           source[i, j])

                    self.psi[i, j] = rhs / coeff

            # Apply relaxation
            self.psi = omega * self.psi + (1 - omega) * psi_old

            # Boundary conditions: psi = 0 at boundaries
            self.psi[0, :] = 0
            self.psi[-1, :] = 0
            self.psi[:, 0] = 0
            self.psi[:, -1] = 0

            # Check convergence
            error = np.max(np.abs(self.psi - psi_old))

            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: error = {error:.2e}")

            if error < tol:
                print(f"Converged after {iteration} iterations")
                return True

        print(f"Did not converge after {max_iter} iterations")
        return False

    def solovev_solution(self, epsilon=0.32, kappa=1.0, delta=0.0):
        """
        Analytical Solovev solution for verification.

        Parameters
        ----------
        epsilon : float
            Inverse aspect ratio a/R0
        kappa : float
            Elongation
        delta : float
            Triangularity

        Returns
        -------
        psi_solovev : ndarray
            Analytical flux function
        """
        R0 = (self.R_arr.max() + self.R_arr.min()) / 2
        a = epsilon * R0

        # Solovev parameters
        c = [1.0, 0.0, -1.0/(2*a**2), 0.0]  # Simplified

        psi_sol = np.zeros_like(self.psi)

        for i in range(self.nR):
            for j in range(self.nZ):
                R = self.R[i, j]
                Z = self.Z[i, j]

                x = (R - R0) / a
                y = Z / a

                # Solovev form: ψ = c[0]*x² + c[2]*y² + higher order terms
                psi_sol[i, j] = (c[0] * x**2 + c[2] * y**2 +
                                  c[1] * x + c[3] * (x**2 * y - y**3/3))

        # Normalize
        psi_sol = psi_sol - psi_sol.min()
        psi_sol = psi_sol / psi_sol.max()

        return psi_sol


def compute_current_density(solver):
    """
    Compute toroidal current density: J_φ = R Δ*ψ / μ₀

    Parameters
    ----------
    solver : GradShafranovSolver
        Solver instance

    Returns
    -------
    J_phi : ndarray
        Toroidal current density
    """
    delta_star_psi = solver.delta_star_operator(solver.psi)
    J_phi = solver.R * delta_star_psi / MU_0

    return J_phi


def plot_equilibrium(solver, J_phi):
    """
    Plot flux surfaces and current density.

    Parameters
    ----------
    solver : GradShafranovSolver
        Solver instance
    J_phi : ndarray
        Current density
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Flux surfaces
    levels = np.linspace(solver.psi.min(), solver.psi.max(), 20)
    cs1 = axes[0].contour(solver.R, solver.Z, solver.psi, levels=levels,
                          colors='blue', linewidths=1.5)
    axes[0].set_xlabel('R (m)', fontsize=12)
    axes[0].set_ylabel('Z (m)', fontsize=12)
    axes[0].set_title('Flux Surfaces (ψ contours)', fontsize=13, fontweight='bold')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    # Current density
    im2 = axes[1].contourf(solver.R, solver.Z, J_phi / 1e6,
                           levels=20, cmap='RdBu_r')
    axes[1].contour(solver.R, solver.Z, solver.psi, levels=10,
                    colors='black', linewidths=0.5, alpha=0.3)
    axes[1].set_xlabel('R (m)', fontsize=12)
    axes[1].set_ylabel('Z (m)', fontsize=12)
    axes[1].set_title('Toroidal Current Density J_φ', fontsize=13, fontweight='bold')
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1], label='J_φ (MA/m²)')

    # Pressure (computed from psi)
    psi_norm = (solver.psi - solver.psi.min()) / (solver.psi.max() - solver.psi.min() + 1e-10)
    p = 1e5 * (1 - psi_norm)**2
    im3 = axes[2].contourf(solver.R, solver.Z, p / 1e3, levels=20, cmap='hot')
    axes[2].contour(solver.R, solver.Z, solver.psi, levels=10,
                    colors='blue', linewidths=0.5, alpha=0.3)
    axes[2].set_xlabel('R (m)', fontsize=12)
    axes[2].set_ylabel('Z (m)', fontsize=12)
    axes[2].set_title('Pressure', fontsize=13, fontweight='bold')
    axes[2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[2], label='Pressure (kPa)')

    plt.suptitle('Grad-Shafranov Equilibrium Solution',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('grad_shafranov_solution.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    # Create solver
    R_range = (0.5, 1.5)  # 0.5 to 1.5 meters
    Z_range = (-0.5, 0.5)  # -0.5 to 0.5 meters
    nR, nZ = 80, 80

    solver = GradShafranovSolver(R_range, Z_range, nR, nZ)

    print("Grad-Shafranov Equation Solver")
    print("=" * 50)
    print(f"Grid: {nR} x {nZ}")
    print(f"R range: {R_range[0]:.2f} to {R_range[1]:.2f} m")
    print(f"Z range: {Z_range[0]:.2f} to {Z_range[1]:.2f} m")
    print()

    # Solve using Picard iteration
    converged = solver.solve_picard(max_iter=200, tol=1e-6, omega=0.5)

    if not converged:
        print("Warning: Solver did not fully converge")

    # Compute current density
    J_phi = compute_current_density(solver)

    print(f"\nResults:")
    print(f"  ψ range: {solver.psi.min():.3e} to {solver.psi.max():.3e}")
    print(f"  Peak current density: {np.max(np.abs(J_phi))/1e6:.2f} MA/m²")

    # Compare with Solovev solution (for verification)
    psi_solovev = solver.solovev_solution(epsilon=0.3)
    difference = np.max(np.abs(solver.psi/solver.psi.max() - psi_solovev))
    print(f"  Difference from Solovev solution: {difference:.3f}")

    # Plot results
    plot_equilibrium(solver, J_phi)
    print("\nPlot saved as 'grad_shafranov_solution.png'")


if __name__ == '__main__':
    main()
