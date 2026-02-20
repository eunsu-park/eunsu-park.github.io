#!/usr/bin/env python3
"""
MHD Eigenvalue Solver

Solves the linearized MHD force operator eigenvalue problem:
    F·ξ = -ω² ρ ξ

where F is the force operator, ξ is the displacement, ω is the frequency,
and ρ is the mass density.

Discretizes the operator on a radial grid and solves the generalized
eigenvalue problem to find growth rates for unstable modes.

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.integrate import simpson


class MHDEigenvalueSolver:
    """
    Solver for MHD stability eigenvalue problem.

    Attributes
    ----------
    r : ndarray
        Radial grid
    n_points : int
        Number of grid points
    """

    def __init__(self, r_max=0.01, n_points=100):
        """
        Initialize solver.

        Parameters
        ----------
        r_max : float
            Maximum radius
        n_points : int
            Number of radial points
        """
        # Avoid r=0 for numerical stability
        self.r = np.linspace(r_max/n_points, r_max, n_points)
        self.r_max = r_max
        self.n_points = n_points
        self.dr = self.r[1] - self.r[0]

    def equilibrium(self, B_z, I_total, rho_0=1e-3):
        """
        Set up equilibrium profiles.

        Parameters
        ----------
        B_z : float
            Axial field (T)
        I_total : float
            Total current (A)
        rho_0 : float
            Central density (kg/m³)

        Returns
        -------
        dict
            Equilibrium quantities
        """
        mu_0 = 4 * np.pi * 1e-7

        # Parabolic current density
        J_z = 2 * I_total / (np.pi * self.r_max**2) * \
              (1 - (self.r/self.r_max)**2)

        # Azimuthal field
        B_theta = np.zeros_like(self.r)
        for i, ri in enumerate(self.r):
            # Approximate enclosed current
            r_frac = ri / self.r_max
            I_enc = I_total * (2*r_frac**2 - r_frac**4)
            B_theta[i] = mu_0 * I_enc / (2 * np.pi * ri)

        # Pressure from force balance
        p = np.zeros_like(self.r)
        p_edge = 100.0
        p[-1] = p_edge

        for i in range(len(self.r)-2, -1, -1):
            p[i] = p[i+1] - J_z[i] * B_theta[i] * self.dr

        # Density profile
        rho = rho_0 * (1 - (self.r/self.r_max)**2)

        return {
            'B_z': B_z,
            'B_theta': B_theta,
            'J_z': J_z,
            'p': p,
            'rho': rho
        }

    def construct_force_operator(self, eq, m):
        """
        Construct discretized force operator F.

        F = -∇(δp) + (1/μ₀)(∇×δB)×B + (1/μ₀)(∇×B)×δB

        Simplified for cylindrical geometry with mode number m.

        Parameters
        ----------
        eq : dict
            Equilibrium quantities
        m : int
            Poloidal mode number

        Returns
        -------
        F : ndarray
            Force operator matrix
        M : ndarray
            Mass matrix (diagonal)
        """
        mu_0 = 4 * np.pi * 1e-7
        n = self.n_points

        # Simplified operator for radial force balance
        # F_r ~ d²ξ/dr² + (1/r)dξ/dr - (m²/r²)ξ + (pressure and field terms)

        # Construct tridiagonal matrix for Laplacian
        main_diag = np.zeros(n)
        upper_diag = np.zeros(n-1)
        lower_diag = np.zeros(n-1)

        B_theta = eq['B_theta']
        B_z = eq['B_z']
        p = eq['p']
        rho = eq['rho']

        for i in range(1, n-1):
            r_i = self.r[i]

            # Laplacian coefficients
            main_diag[i] = -2/self.dr**2 - m**2/r_i**2
            upper_diag[i] = 1/self.dr**2 + 1/(2*r_i*self.dr)
            if i > 0:
                lower_diag[i-1] = 1/self.dr**2 - 1/(2*r_i*self.dr)

            # Magnetic tension term
            B_tot_sq = B_z**2 + B_theta[i]**2
            main_diag[i] += B_tot_sq / (mu_0 * r_i**2)

            # Pressure gradient term (stabilizing)
            if i < n-1 and i > 0:
                dp_dr = (p[i+1] - p[i-1]) / (2*self.dr)
                main_diag[i] -= dp_dr / r_i

        # Boundary conditions: ξ = 0 at boundaries
        main_diag[0] = 1.0
        main_diag[-1] = 1.0
        if n > 1:
            upper_diag[0] = 0.0
            lower_diag[-1] = 0.0

        # Construct sparse matrix
        F = diags([lower_diag, main_diag, upper_diag],
                  offsets=[-1, 0, 1],
                  shape=(n, n)).toarray()

        # Mass matrix (diagonal, proportional to density)
        M = diags([rho], offsets=[0], shape=(n, n)).toarray()

        return F, M

    def solve_eigenvalue_problem(self, F, M, n_eigs=6):
        """
        Solve generalized eigenvalue problem: F·ξ = λ M·ξ

        where λ = -ω².

        Parameters
        ----------
        F : ndarray
            Force operator
        M : ndarray
            Mass matrix
        n_eigs : int
            Number of eigenvalues to compute

        Returns
        -------
        eigenvalues : ndarray
            Eigenvalues (λ = -ω²)
        eigenvectors : ndarray
            Eigenvectors (displacement patterns)
        """
        # Solve standard eigenvalue problem: M^{-1}·F·ξ = λ·ξ
        # Add small regularization to M
        M_reg = M + np.eye(M.shape[0]) * 1e-10

        # Use numpy for small problems
        eigenvalues, eigenvectors = np.linalg.eig(
            np.linalg.solve(M_reg, F)
        )

        # Sort by eigenvalue (most unstable first)
        idx = np.argsort(eigenvalues.real)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues[:n_eigs], eigenvectors[:, :n_eigs]

    def compute_growth_rates(self, eigenvalues):
        """
        Compute growth rates from eigenvalues.

        For F·ξ = -ω²·M·ξ, if λ = -ω², then:
        - λ > 0: ω² < 0, oscillatory (stable)
        - λ < 0: ω² > 0, ω = ±sqrt(-λ), exponential growth

        Parameters
        ----------
        eigenvalues : ndarray
            Eigenvalues

        Returns
        -------
        gamma : ndarray
            Growth rates (real part of ω)
        """
        gamma = np.zeros(len(eigenvalues))

        for i, lam in enumerate(eigenvalues):
            if lam.real < 0:
                # Unstable: ω = ±i·sqrt(-λ) has real part
                gamma[i] = np.sqrt(-lam.real)
            else:
                # Stable or oscillatory
                gamma[i] = 0.0

        return gamma


def plot_eigenfunctions(solver, eigenvectors, eigenvalues, n_plot=4):
    """
    Plot eigenfunctions for unstable modes.

    Parameters
    ----------
    solver : MHDEigenvalueSolver
        Solver instance
    eigenvectors : ndarray
        Eigenvectors
    eigenvalues : ndarray
        Eigenvalues
    n_plot : int
        Number of modes to plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    r_cm = solver.r * 100  # Convert to cm

    for i in range(min(n_plot, len(eigenvalues))):
        ax = axes[i]

        # Eigenvector
        xi = eigenvectors[:, i].real
        # Normalize
        xi = xi / np.max(np.abs(xi))

        ax.plot(r_cm, xi, 'b-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Radius (cm)', fontsize=11)
        ax.set_ylabel('Normalized ξ_r', fontsize=11)

        # Growth rate
        gamma = np.sqrt(-eigenvalues[i].real) if eigenvalues[i].real < 0 else 0
        stability = "UNSTABLE" if gamma > 0 else "STABLE"
        color = "red" if gamma > 0 else "green"

        ax.set_title(f'Mode {i+1}: λ={eigenvalues[i].real:.2e}, γ={gamma:.2e}\n{stability}',
                    fontsize=11, fontweight='bold', color=color)
        ax.grid(True, alpha=0.3)

    plt.suptitle('MHD Eigenfunctions (Radial Displacement)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eigenfunctions.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_spectrum(eigenvalues, growth_rates):
    """
    Plot eigenvalue spectrum and growth rates.

    Parameters
    ----------
    eigenvalues : ndarray
        Eigenvalues
    growth_rates : ndarray
        Growth rates
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Eigenvalue spectrum
    ax1 = axes[0]
    ax1.scatter(eigenvalues.real, eigenvalues.imag,
               c=growth_rates, cmap='coolwarm',
               s=100, edgecolors='black', linewidths=1.5)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Re(λ)', fontsize=12)
    ax1.set_ylabel('Im(λ)', fontsize=12)
    ax1.set_title('Eigenvalue Spectrum', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add stability boundary
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2,
               label='Stability boundary')
    ax1.legend(fontsize=10)

    # Growth rates
    ax2 = axes[1]
    mode_numbers = np.arange(1, len(growth_rates) + 1)
    colors = ['red' if g > 1e-6 else 'green' for g in growth_rates]

    ax2.bar(mode_numbers, growth_rates, color=colors,
           alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Mode number', fontsize=12)
    ax2.set_ylabel('Growth rate γ (s⁻¹)', fontsize=12)
    ax2.set_title('Growth Rate Spectrum', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add threshold line
    if np.max(growth_rates) > 0:
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)

    plt.tight_layout()
    plt.savefig('eigenvalue_spectrum.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    # Initialize solver
    r_max = 0.01  # 1 cm
    solver = MHDEigenvalueSolver(r_max=r_max, n_points=100)

    # Equilibrium parameters
    B_z = 0.3  # T
    I_total = 50e3  # 50 kA
    rho_0 = 1e-3  # kg/m³

    print("MHD Eigenvalue Solver")
    print("=" * 60)
    print(f"Configuration: Z-pinch")
    print(f"  Radius: {r_max*100:.1f} cm")
    print(f"  Axial field: {B_z:.2f} T")
    print(f"  Current: {I_total/1e3:.1f} kA")
    print(f"  Grid points: {solver.n_points}")
    print()

    # Compute equilibrium
    eq = solver.equilibrium(B_z, I_total, rho_0)

    print("Equilibrium computed:")
    print(f"  Peak B_θ: {np.max(eq['B_theta'])*1e3:.2f} mT")
    print(f"  Peak pressure: {np.max(eq['p'])/1e3:.2f} kPa")
    print(f"  Peak density: {np.max(eq['rho'])*1e3:.2f} g/m³")
    print()

    # Solve eigenvalue problem for m=1 mode
    m = 1
    print(f"Solving eigenvalue problem for m={m} mode...")

    F, M = solver.construct_force_operator(eq, m)
    eigenvalues, eigenvectors = solver.solve_eigenvalue_problem(F, M, n_eigs=8)
    growth_rates = solver.compute_growth_rates(eigenvalues)

    print(f"\nEigenvalue results:")
    print(f"  {'Mode':<6} {'Re(λ)':<12} {'Im(λ)':<12} {'γ (s⁻¹)':<12} {'Status'}")
    print("  " + "-" * 60)

    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        gamma = growth_rates[i]
        status = "UNSTABLE" if gamma > 1e-6 else "STABLE"

        print(f"  {i+1:<6} {lam.real:<12.3e} {lam.imag:<12.3e} "
              f"{gamma:<12.3e} {status}")

    # Count unstable modes
    n_unstable = np.sum(growth_rates > 1e-6)
    print(f"\nNumber of unstable modes: {n_unstable}")

    if n_unstable > 0:
        max_gamma = np.max(growth_rates)
        max_idx = np.argmax(growth_rates)
        print(f"Most unstable mode: #{max_idx+1} with γ = {max_gamma:.3e} s⁻¹")
        print(f"Growth time: {1/max_gamma:.3e} s")

    # Plot results
    plot_eigenfunctions(solver, eigenvectors, eigenvalues, n_plot=4)
    print("\nEigenfunction plot saved as 'eigenfunctions.png'")

    plot_spectrum(eigenvalues, growth_rates)
    print("Eigenvalue spectrum plot saved as 'eigenvalue_spectrum.png'")


if __name__ == '__main__':
    main()
