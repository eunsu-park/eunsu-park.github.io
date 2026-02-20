#!/usr/bin/env python3
"""
2D Magnetic Reconnection Simulation

Simplified 2D resistive MHD simulation of magnetic reconnection in a Harris current sheet.
This demonstrates the fundamental process by which magnetic field lines break and reconnect,
releasing stored magnetic energy as kinetic and thermal energy.

Key results:
- Harris current sheet evolves through resistive reconnection
- Formation of X-point and magnetic islands
- Current density concentrates at reconnection sites
- Magnetic energy converts to kinetic/thermal energy

Physics:
- Ideal MHD with resistivity: ∂B/∂t = ∇×(v×B) - ∇×(ηJ)
- Momentum: ρ∂v/∂t = J×B - ∇p
- Incompressible flow: ∇·v = 0

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from typing import Tuple


class MagneticReconnection2D:
    """
    2D Magnetic Reconnection Solver using simplified resistive MHD.

    Uses the vector potential formulation where B = ∇×A (with A = A_z ẑ in 2D).
    Evolution equations:
    - ∂A/∂t = -E_z = (v×B)_z - η J_z
    - ∂ω/∂t = ∇×(J×B/ρ) - ν∇²ω  (vorticity equation)
    where ω = ∇²ψ is the vorticity and v = ∇×ψ ŷ
    """

    def __init__(self, nx: int = 128, ny: int = 128,
                 Lx: float = 20.0, Ly: float = 10.0,
                 eta: float = 0.01, nu: float = 0.001):
        """
        Initialize the reconnection simulation.

        Args:
            nx, ny: Grid resolution
            Lx, Ly: Domain size
            eta: Magnetic resistivity (controls reconnection rate)
            nu: Kinematic viscosity (for numerical stability)
        """
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.eta = eta
        self.nu = nu

        # Grid spacing
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)

        # Coordinate arrays
        self.x = np.linspace(-Lx/2, Lx/2, nx)
        self.y = np.linspace(-Ly/2, Ly/2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Field arrays
        self.A = np.zeros((nx, ny))      # Vector potential (magnetic flux function)
        self.psi = np.zeros((nx, ny))    # Stream function (velocity potential)
        self.Bx = np.zeros((nx, ny))     # Magnetic field components
        self.By = np.zeros((nx, ny))
        self.Jz = np.zeros((nx, ny))     # Current density (out-of-plane)
        self.vx = np.zeros((nx, ny))     # Velocity components
        self.vy = np.zeros((nx, ny))

        # Energy tracking
        self.time = 0.0
        self.magnetic_energy = []
        self.kinetic_energy = []
        self.time_history = []

    def harris_current_sheet(self, B0: float = 1.0, width: float = 1.0,
                            perturbation: float = 0.1):
        """
        Initialize Harris current sheet equilibrium with perturbation.

        The Harris sheet is a classic 1D equilibrium with:
        - B_x(y) = B0 * tanh(y/width)
        - J_z = -B0/width * sech²(y/width)

        We add a small perturbation to trigger reconnection.

        Args:
            B0: Asymptotic magnetic field strength
            width: Current sheet thickness
            perturbation: Amplitude of initial perturbation
        """
        # Harris sheet: A = -B0 * width * ln(cosh(y/width))
        for i in range(self.nx):
            for j in range(self.ny):
                y = self.Y[i, j]
                x = self.X[i, j]

                # Base Harris profile
                self.A[i, j] = -B0 * width * np.log(np.cosh(y / width))

                # Add perturbation to seed reconnection
                # Symmetric tearing mode: δA ∝ cos(kx) * sech(y/width)
                k = 2.0 * np.pi / self.Lx
                self.A[i, j] += perturbation * B0 * np.cos(k * x) / np.cosh(y / width)

        # Initialize stream function (small or zero)
        self.psi = np.zeros_like(self.A)

        # Compute derived fields
        self._update_derived_fields()

    def _update_derived_fields(self):
        """Compute B, J, v from potentials A and psi."""
        # Magnetic field: B = ∇×A = (∂A/∂y, -∂A/∂x, 0)
        self.By, self.Bx = np.gradient(self.A, self.dx, self.dy)
        self.Bx = -self.Bx

        # Current density: J_z = -∇²A (Ampere's law in 2D)
        self.Jz = -self._laplacian(self.A)

        # Velocity: v = ∇×psi ŷ = (-∂psi/∂y, ∂psi/∂x, 0)
        self.vy, self.vx = np.gradient(self.psi, self.dx, self.dy)
        self.vx = -self.vx

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute 2D Laplacian using centered finite differences."""
        laplacian = np.zeros_like(field)

        laplacian[1:-1, 1:-1] = (
            (field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1]) / self.dx**2 +
            (field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]) / self.dy**2
        )

        return laplacian

    def _poisson_solve(self, rhs: np.ndarray, max_iter: int = 1000,
                      tol: float = 1e-5) -> np.ndarray:
        """
        Solve Poisson equation ∇²φ = rhs using Jacobi iteration.

        Used for: ∇²ψ = ω (vorticity → stream function)
        """
        phi = np.zeros_like(rhs)
        dx2 = self.dx**2
        dy2 = self.dy**2
        denom = 2.0 * (dx2 + dy2)

        for iteration in range(max_iter):
            phi_old = phi.copy()

            phi[1:-1, 1:-1] = (
                dx2 * (phi_old[1:-1, 2:] + phi_old[1:-1, :-2]) +
                dy2 * (phi_old[2:, 1:-1] + phi_old[:-2, 1:-1]) -
                dx2 * dy2 * rhs[1:-1, 1:-1]
            ) / denom

            # Boundary conditions: φ = 0 at boundaries
            phi[0, :] = phi[-1, :] = phi[:, 0] = phi[:, -1] = 0

            # Check convergence
            error = np.max(np.abs(phi - phi_old))
            if error < tol:
                break

        return phi

    def step(self, dt: float):
        """
        Advance the simulation by one time step using forward Euler.

        Evolution equations:
        1. Induction: ∂A/∂t = (v×B)_z - η J_z = v_x B_y - v_y B_x - η J_z
        2. Vorticity: ∂ω/∂t = (∇×(J×B))_z - ν∇²ω
        """
        # Compute vorticity
        omega = -self._laplacian(self.psi)

        # 1. Advance vector potential A
        # ∂A/∂t = E_z = v×B - ηJ
        vxB = self.vx * self.By - self.vy * self.Bx
        dA_dt = vxB - self.eta * self.Jz

        self.A += dt * dA_dt

        # Boundary conditions for A (conducting walls)
        self.A[0, :] = self.A[-1, :] = self.A[:, 0] = self.A[:, -1] = 0

        # 2. Advance vorticity ω
        # J×B force term
        JxB_x = self.Jz * self.By
        JxB_y = -self.Jz * self.Bx

        # Curl of J×B (simplified, assuming constant density ρ=1)
        dJxB_y_dx = np.gradient(JxB_y, self.dx, axis=0)
        dJxB_x_dy = np.gradient(JxB_x, self.dy, axis=1)
        curl_JxB = dJxB_y_dx - dJxB_x_dy

        # Viscous term
        visc_omega = self.nu * self._laplacian(omega)

        domega_dt = curl_JxB + visc_omega
        omega += dt * domega_dt

        # 3. Recover stream function from vorticity: ∇²ψ = ω
        self.psi = self._poisson_solve(omega)

        # 4. Update all derived fields
        self._update_derived_fields()

        # Update time
        self.time += dt

        # Track energies
        self._compute_energies()

    def _compute_energies(self):
        """Compute magnetic and kinetic energies."""
        # Magnetic energy: ∫ B²/2 dV
        B_squared = self.Bx**2 + self.By**2
        E_mag = 0.5 * np.sum(B_squared) * self.dx * self.dy

        # Kinetic energy: ∫ ρv²/2 dV (ρ=1)
        v_squared = self.vx**2 + self.vy**2
        E_kin = 0.5 * np.sum(v_squared) * self.dx * self.dy

        self.magnetic_energy.append(E_mag)
        self.kinetic_energy.append(E_kin)
        self.time_history.append(self.time)

    def find_xpoint(self) -> Tuple[float, float]:
        """
        Find the X-point location (magnetic null point where B=0).

        Returns:
            (x, y) coordinates of X-point
        """
        # Find minimum of |B|
        B_magnitude = np.sqrt(self.Bx**2 + self.By**2)

        # Search in central region
        nx_mid = self.nx // 2
        ny_mid = self.ny // 2
        search_region = B_magnitude[nx_mid-20:nx_mid+20, ny_mid-20:ny_mid+20]

        i_min, j_min = np.unravel_index(search_region.argmin(), search_region.shape)
        i_min += nx_mid - 20
        j_min += ny_mid - 20

        return self.X[i_min, j_min], self.Y[i_min, j_min]


def visualize_reconnection(sim: MagneticReconnection2D,
                          save_prefix: str = "reconnection"):
    """
    Create comprehensive visualization of magnetic reconnection.

    Shows:
    1. Current density with magnetic field lines
    2. Velocity field
    3. Energy evolution
    """
    fig = plt.figure(figsize=(16, 10))

    # 1. Current density and magnetic field lines
    ax1 = plt.subplot(2, 3, 1)

    # Current density color map
    levels = np.linspace(-1.5, 1.5, 30)
    cf = ax1.contourf(sim.X, sim.Y, sim.Jz, levels=levels, cmap='RdBu_r', extend='both')
    plt.colorbar(cf, ax=ax1, label='Current density $J_z$')

    # Magnetic field lines (contours of A)
    A_levels = np.linspace(sim.A.min(), sim.A.max(), 20)
    ax1.contour(sim.X, sim.Y, sim.A, levels=A_levels, colors='black',
                linewidths=0.8, alpha=0.6)

    # Mark X-point
    try:
        x_xpt, y_xpt = sim.find_xpoint()
        ax1.plot(x_xpt, y_xpt, 'g*', markersize=15, label='X-point')
        ax1.legend()
    except:
        pass

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Current Density and B-field Lines (t={sim.time:.2f})')
    ax1.set_aspect('equal')

    # 2. Magnetic field vectors
    ax2 = plt.subplot(2, 3, 2)

    # Subsample for quiver plot
    skip = 4
    ax2.quiver(sim.X[::skip, ::skip], sim.Y[::skip, ::skip],
               sim.Bx[::skip, ::skip], sim.By[::skip, ::skip],
               alpha=0.7)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Magnetic Field Vectors')
    ax2.set_aspect('equal')

    # 3. Velocity field
    ax3 = plt.subplot(2, 3, 3)

    v_magnitude = np.sqrt(sim.vx**2 + sim.vy**2)
    cf3 = ax3.contourf(sim.X, sim.Y, v_magnitude, levels=20, cmap='plasma')
    plt.colorbar(cf3, ax=ax3, label='|v|')

    # Velocity streamlines
    ax3.streamplot(sim.x, sim.y, sim.vx.T, sim.vy.T,
                   color='white', linewidth=0.5, density=1.5, arrowsize=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Velocity Field')
    ax3.set_aspect('equal')

    # 4. Energy evolution
    ax4 = plt.subplot(2, 3, 4)

    times = np.array(sim.time_history)
    E_mag = np.array(sim.magnetic_energy)
    E_kin = np.array(sim.kinetic_energy)
    E_total = E_mag + E_kin

    ax4.plot(times, E_mag, 'b-', label='Magnetic', linewidth=2)
    ax4.plot(times, E_kin, 'r-', label='Kinetic', linewidth=2)
    ax4.plot(times, E_total, 'k--', label='Total', linewidth=1.5)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Energy')
    ax4.set_title('Energy Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Current density profile at y=0
    ax5 = plt.subplot(2, 3, 5)

    j_mid = sim.ny // 2
    ax5.plot(sim.x, sim.Jz[:, j_mid], 'b-', linewidth=2)
    ax5.set_xlabel('x')
    ax5.set_ylabel('$J_z(x, y=0)$')
    ax5.set_title('Current Sheet Profile')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # 6. Magnetic field profile at x=0
    ax6 = plt.subplot(2, 3, 6)

    i_mid = sim.nx // 2
    ax6.plot(sim.y, sim.Bx[i_mid, :], 'b-', label='$B_x$', linewidth=2)
    ax6.plot(sim.y, sim.By[i_mid, :], 'r-', label='$B_y$', linewidth=2)
    ax6.set_xlabel('y')
    ax6.set_ylabel('B(x=0, y)')
    ax6.set_title('Magnetic Field Profile')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'/opt/projects/01_Personal/03_Study/examples/Numerical_Simulation/10_projects/{save_prefix}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {save_prefix}.png")


def run_reconnection_simulation():
    """Run a complete magnetic reconnection simulation."""
    print("=" * 70)
    print("2D Magnetic Reconnection Simulation")
    print("=" * 70)

    # Initialize simulation
    print("\nInitializing Harris current sheet...")
    sim = MagneticReconnection2D(nx=128, ny=64, Lx=20.0, Ly=10.0,
                                  eta=0.01, nu=0.001)
    sim.harris_current_sheet(B0=1.0, width=1.0, perturbation=0.1)

    print(f"  Grid: {sim.nx} × {sim.ny}")
    print(f"  Domain: [{-sim.Lx/2:.1f}, {sim.Lx/2:.1f}] × [{-sim.Ly/2:.1f}, {sim.Ly/2:.1f}]")
    print(f"  Resistivity η = {sim.eta}")
    print(f"  Initial magnetic energy: {sim.magnetic_energy[0]:.4f}")

    # Time stepping
    dt = 0.01
    t_final = 50.0
    n_steps = int(t_final / dt)
    output_interval = 500

    print(f"\nTime integration:")
    print(f"  dt = {dt}, t_final = {t_final}")
    print(f"  Total steps: {n_steps}")

    # Initial state
    print("\nSaving initial state...")
    visualize_reconnection(sim, save_prefix="reconnection_t000")

    # Time evolution
    print("\nEvolving system...")
    for step in range(n_steps):
        sim.step(dt)

        if (step + 1) % output_interval == 0:
            print(f"  Step {step+1}/{n_steps}, t={sim.time:.2f}, "
                  f"E_mag={sim.magnetic_energy[-1]:.4f}, "
                  f"E_kin={sim.kinetic_energy[-1]:.4f}")

            # Save snapshot
            visualize_reconnection(sim,
                                  save_prefix=f"reconnection_t{int(sim.time):03d}")

    # Final state
    print("\nFinal state:")
    print(f"  Time: {sim.time:.2f}")
    print(f"  Magnetic energy: {sim.magnetic_energy[-1]:.4f} "
          f"(change: {sim.magnetic_energy[-1]-sim.magnetic_energy[0]:.4f})")
    print(f"  Kinetic energy: {sim.kinetic_energy[-1]:.4f}")
    print(f"  Total energy: {sim.magnetic_energy[-1]+sim.kinetic_energy[-1]:.4f}")

    # Find X-point
    try:
        x_xpt, y_xpt = sim.find_xpoint()
        print(f"  X-point location: ({x_xpt:.2f}, {y_xpt:.2f})")
    except:
        print("  X-point not clearly identified")

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)
    print("""
    Physical Interpretation:

    1. Initial Harris sheet: Anti-parallel magnetic field with thin current layer
    2. Perturbation triggers instability (tearing mode)
    3. X-point forms where field lines break and reconnect
    4. Magnetic energy converts to kinetic energy and heat
    5. Current sheet fragments into magnetic islands (plasmoids)

    Key Parameters:
    - Resistivity η: Controls reconnection rate (higher η → faster reconnection)
    - Perturbation: Seeds the instability
    - Aspect ratio: Affects island formation

    Applications:
    - Solar flares and coronal mass ejections
    - Magnetospheric substorms
    - Tokamak disruptions
    - Astrophysical jets
    """)


if __name__ == "__main__":
    run_reconnection_simulation()
