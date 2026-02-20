#!/usr/bin/env python3
"""
Orszag-Tang Vortex MHD Test Problem

This is a standard benchmark for MHD codes that demonstrates the formation
of shocks, current sheets, and complex vortical structures from smooth
initial conditions.

Initial conditions:
    v = (-sin(y), sin(x), 0)
    B = (-sin(y), sin(2x), 0)
    ρ = γ², P = γ
    Domain: [0, 2π]² with periodic boundaries

The Orszag-Tang vortex tests:
- Shock capturing
- Current sheet formation
- Conservation properties
- Resolution of small-scale structures

Author: MHD Course Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt


class OrszagTangVortex:
    """
    Orszag-Tang vortex test problem.

    This provides initial conditions and analysis for the standard
    MHD benchmark test.
    """

    def __init__(self, nx=256, ny=256, gamma=5/3):
        """
        Initialize Orszag-Tang vortex.

        Parameters:
            nx, ny (int): Grid resolution
            gamma (float): Adiabatic index
        """
        self.nx = nx
        self.ny = ny
        self.gamma = gamma

        # Domain: [0, 2π]²
        self.xmin, self.xmax = 0.0, 2*np.pi
        self.ymin, self.ymax = 0.0, 2*np.pi

        # Grid
        x = np.linspace(self.xmin, self.xmax, nx, endpoint=False)
        y = np.linspace(self.ymin, self.ymax, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')

    def initial_conditions(self):
        """
        Set up Orszag-Tang initial conditions.

        Returns:
            dict: Primitive variables {rho, vx, vy, vz, Bx, By, Bz, P}
        """
        X, Y = self.X, self.Y

        # Density (constant)
        rho = self.gamma**2 * np.ones_like(X)

        # Velocity
        vx = -np.sin(Y)
        vy = np.sin(X)
        vz = np.zeros_like(X)

        # Magnetic field
        Bx = -np.sin(Y)
        By = np.sin(2*X)
        Bz = np.zeros_like(X)

        # Pressure (constant)
        P = self.gamma * np.ones_like(X)

        return {
            'rho': rho, 'vx': vx, 'vy': vy, 'vz': vz,
            'Bx': Bx, 'By': By, 'Bz': Bz, 'P': P
        }

    def compute_diagnostics(self, rho, vx, vy, vz, Bx, By, Bz, P):
        """
        Compute diagnostic quantities.

        Parameters:
            rho, vx, vy, vz, Bx, By, Bz, P: Primitive variables

        Returns:
            dict: Diagnostic quantities
        """
        # Current density: J = ∇×B
        dBx_dy = np.gradient(Bx, axis=1)
        dBy_dx = np.gradient(By, axis=0)
        Jz = dBy_dx - dBx_dy

        # Vorticity: ω = ∇×v
        dvx_dy = np.gradient(vx, axis=1)
        dvy_dx = np.gradient(vy, axis=0)
        omega_z = dvy_dx - dvx_dy

        # Magnetic pressure
        B_squared = Bx**2 + By**2 + Bz**2
        P_mag = 0.5 * B_squared

        # Total pressure
        P_total = P + P_mag

        # Mach number
        sound_speed = np.sqrt(self.gamma * P / rho)
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        mach = v_mag / sound_speed

        # Plasma beta
        beta = P / P_mag

        return {
            'current': Jz,
            'vorticity': omega_z,
            'P_mag': P_mag,
            'P_total': P_total,
            'mach': mach,
            'beta': beta,
            'B_mag': np.sqrt(B_squared)
        }

    def plot_solution(self, rho, vx, vy, vz, Bx, By, Bz, P, time=0.0):
        """
        Plot the solution with standard diagnostics.

        Parameters:
            rho, vx, vy, vz, Bx, By, Bz, P: Primitive variables
            time (float): Current time
        """
        diag = self.compute_diagnostics(rho, vx, vy, vz, Bx, By, Bz, P)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        ax = axes.flatten()

        # Density
        im0 = ax[0].contourf(self.X, self.Y, rho, levels=30, cmap='viridis')
        ax[0].set_title(f'Density (t={time:.3f})')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        plt.colorbar(im0, ax=ax[0])

        # Pressure
        im1 = ax[1].contourf(self.X, self.Y, P, levels=30, cmap='plasma')
        ax[1].set_title('Gas Pressure')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        plt.colorbar(im1, ax=ax[1])

        # Magnetic field magnitude
        im2 = ax[2].contourf(self.X, self.Y, diag['B_mag'], levels=30,
                            cmap='inferno')
        ax[2].set_title('|B|')
        ax[2].set_xlabel('x')
        ax[2].set_ylabel('y')
        plt.colorbar(im2, ax=ax[2])

        # Current density
        im3 = ax[3].contourf(self.X, self.Y, diag['current'], levels=30,
                            cmap='RdBu_r')
        ax[3].set_title(r'Current Density $J_z$')
        ax[3].set_xlabel('x')
        ax[3].set_ylabel('y')
        plt.colorbar(im3, ax=ax[3])

        # Vorticity
        im4 = ax[4].contourf(self.X, self.Y, diag['vorticity'], levels=30,
                            cmap='RdBu_r')
        ax[4].set_title(r'Vorticity $\omega_z$')
        ax[4].set_xlabel('x')
        ax[4].set_ylabel('y')
        plt.colorbar(im4, ax=ax[4])

        # Mach number
        im5 = ax[5].contourf(self.X, self.Y, diag['mach'], levels=30,
                            cmap='hot')
        ax[5].set_title('Mach Number')
        ax[5].set_xlabel('x')
        ax[5].set_ylabel('y')
        plt.colorbar(im5, ax=ax[5])

        plt.tight_layout()
        return fig


def main():
    """
    Demonstrate Orszag-Tang vortex initial conditions.
    """
    print("=" * 60)
    print("Orszag-Tang Vortex MHD Benchmark")
    print("=" * 60)

    # Create problem
    ot = OrszagTangVortex(nx=256, ny=256, gamma=5/3)

    # Get initial conditions
    ic = ot.initial_conditions()

    print("\nInitial conditions set up on 256×256 grid")
    print("Domain: [0, 2π]²")
    print(f"γ = {ot.gamma:.3f}")

    # Compute initial diagnostics
    diag = ot.compute_diagnostics(**ic)

    print(f"\nInitial state:")
    print(f"  ρ range: [{np.min(ic['rho']):.3f}, {np.max(ic['rho']):.3f}]")
    print(f"  P range: [{np.min(ic['P']):.3f}, {np.max(ic['P']):.3f}]")
    print(f"  |B| range: [{np.min(diag['B_mag']):.3f}, {np.max(diag['B_mag']):.3f}]")
    print(f"  |v| max: {np.max(np.sqrt(ic['vx']**2 + ic['vy']**2)):.3f}")

    # Plot initial conditions
    fig = ot.plot_solution(**ic, time=0.0)
    fig.suptitle('Orszag-Tang Vortex: Initial Conditions', fontsize=14,
                 weight='bold', y=0.995)

    plt.savefig('/tmp/orszag_tang_vortex.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to /tmp/orszag_tang_vortex.png")
    print("\nTo run full simulation, use with MHD solver (mhd_2d_ct.py)")

    plt.show()


if __name__ == "__main__":
    main()
