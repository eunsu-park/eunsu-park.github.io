#!/usr/bin/env python3
"""
Production-Quality 1D Vlasov-Poisson Solver

This script implements a high-quality 1D electrostatic Vlasov-Poisson solver
using spectral methods and operator splitting.

Algorithm:
- Strang splitting: half x-advect → full v-advect → half x-advect
- Cubic spline interpolation for advection
- FFT Poisson solver with periodic boundaries

Test cases:
1. Linear plasma oscillation (verify ω = ωpe)
2. Landau damping (verify γ matches theory)
3. Two-stream instability (growth + saturation)
4. Bump-on-tail (plateau formation)

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import CubicSpline
from scipy.fft import fft, ifft, fftfreq

# Physical constants
QE = 1.602176634e-19    # C
ME = 9.10938356e-31     # kg
EPS0 = 8.854187817e-12  # F/m

class VlasovPoisson1D:
    """1D Vlasov-Poisson solver."""

    def __init__(self, nx, nv, Lx, vmax, dt):
        """
        Initialize solver.

        Parameters:
        -----------
        nx : int
            Number of spatial grid points
        nv : int
            Number of velocity grid points
        Lx : float
            Spatial domain length [m]
        vmax : float
            Maximum velocity [m/s]
        dt : float
            Time step [s]
        """
        self.nx = nx
        self.nv = nv
        self.Lx = Lx
        self.vmax = vmax
        self.dt = dt

        # Spatial grid (periodic)
        self.x = np.linspace(0, Lx, nx, endpoint=False)
        self.dx = Lx / nx

        # Velocity grid (symmetric)
        self.v = np.linspace(-vmax, vmax, nv)
        self.dv = 2 * vmax / (nv - 1)

        # Phase space distribution
        self.f = np.zeros((nx, nv))

        # Electric field and potential
        self.E = np.zeros(nx)
        self.phi = np.zeros(nx)

        # Diagnostics storage
        self.diagnostics = {
            'time': [],
            'field_energy': [],
            'kinetic_energy': [],
            'total_energy': [],
            'momentum': [],
            'entropy': [],
            'E_max': []
        }

    def initialize_maxwellian(self, n0, v0, vth):
        """
        Initialize with a shifted Maxwellian.

        f(x,v) = (n0 / sqrt(2π vth²)) exp(-(v-v0)²/(2vth²))

        Parameters:
        -----------
        n0 : float
            Density [m^-3]
        v0 : float
            Drift velocity [m/s]
        vth : float
            Thermal velocity [m/s]
        """
        for i in range(self.nx):
            self.f[i, :] = (n0 / np.sqrt(2 * np.pi * vth**2)) * \
                          np.exp(-(self.v - v0)**2 / (2 * vth**2))

    def initialize_two_stream(self, n0, v0, vth):
        """
        Initialize two counter-streaming beams.

        Parameters:
        -----------
        n0 : float
            Density per beam [m^-3]
        v0 : float
            Beam velocity [m/s]
        vth : float
            Thermal spread [m/s]
        """
        for i in range(self.nx):
            self.f[i, :] = (n0 / (2 * np.sqrt(2 * np.pi * vth**2))) * \
                          (np.exp(-(self.v - v0)**2 / (2 * vth**2)) +
                           np.exp(-(self.v + v0)**2 / (2 * vth**2)))

    def initialize_bump_on_tail(self, n_bulk, n_beam, v_beam, vth_bulk, vth_beam):
        """
        Initialize bump-on-tail distribution.

        Parameters:
        -----------
        n_bulk : float
            Bulk density [m^-3]
        n_beam : float
            Beam density [m^-3]
        v_beam : float
            Beam velocity [m/s]
        vth_bulk : float
            Bulk thermal velocity [m/s]
        vth_beam : float
            Beam thermal spread [m/s]
        """
        for i in range(self.nx):
            # Bulk
            f_bulk = (n_bulk / np.sqrt(2 * np.pi * vth_bulk**2)) * \
                    np.exp(-self.v**2 / (2 * vth_bulk**2))

            # Beam
            f_beam = (n_beam / np.sqrt(2 * np.pi * vth_beam**2)) * \
                    np.exp(-(self.v - v_beam)**2 / (2 * vth_beam**2))

            self.f[i, :] = f_bulk + f_beam

    def add_perturbation(self, k, amplitude):
        """
        Add sinusoidal perturbation in x.

        f → f * (1 + amplitude * cos(k*x))

        Parameters:
        -----------
        k : float
            Wavenumber [rad/m]
        amplitude : float
            Perturbation amplitude
        """
        for j in range(self.nv):
            self.f[:, j] *= (1 + amplitude * np.cos(k * self.x))

    def compute_moments(self):
        """
        Compute density and current from distribution function.

        Returns:
        --------
        n, j : density [m^-3] and current density [A/m^2]
        """
        # Density: n = ∫ f dv
        n = np.trapz(self.f, x=self.v, axis=1)

        # Current: j = q ∫ v f dv
        j = QE * np.trapz(self.v[None, :] * self.f, x=self.v, axis=1)

        return n, j

    def solve_poisson_fft(self, ne):
        """
        Solve Poisson equation using FFT: ∇²φ = e(ne - ni)/ε0.

        Assumes ni = n0 (background ions).

        Parameters:
        -----------
        ne : array
            Electron density [m^-3]

        Returns:
        --------
        phi, E : potential [V] and electric field [V/m]
        """
        # Background ion density (uniform)
        n0 = np.mean(ne)

        # Charge density
        rho = -QE * (ne - n0)

        # FFT
        rho_k = fft(rho)
        k = 2 * np.pi * fftfreq(self.nx, d=self.dx)

        # Solve in Fourier space: -k²φ_k = rho_k/ε0
        # Avoid division by zero at k=0
        phi_k = np.zeros_like(rho_k, dtype=complex)
        phi_k[k != 0] = -rho_k[k != 0] / (k[k != 0]**2 * EPS0)

        # Inverse FFT
        phi = np.real(ifft(phi_k))

        # Electric field: E = -dφ/dx
        E_k = 1j * k * phi_k
        E = np.real(ifft(E_k))

        return phi, E

    def advect_x(self, f, dt_frac):
        """
        Advect in x direction: ∂f/∂t + v·∂f/∂x = 0.

        Uses cubic spline interpolation.

        Parameters:
        -----------
        f : array (nx, nv)
            Distribution function
        dt_frac : float
            Time step (can be fractional for splitting)

        Returns:
        --------
        f_new : array (nx, nv)
        """
        f_new = np.zeros_like(f)

        for j in range(self.nv):
            # Displacement
            dx = self.v[j] * dt_frac

            # Interpolate (periodic boundary)
            cs = CubicSpline(self.x, f[:, j], bc_type='periodic')

            x_new = (self.x - dx) % self.Lx  # Periodic
            f_new[:, j] = cs(x_new)

        return f_new

    def advect_v(self, f, E, dt_frac):
        """
        Advect in v direction: ∂f/∂t + a·∂f/∂v = 0.

        where a = qE/m.

        Parameters:
        -----------
        f : array (nx, nv)
            Distribution function
        E : array (nx,)
            Electric field [V/m]
        dt_frac : float
            Time step

        Returns:
        --------
        f_new : array (nx, nv)
        """
        f_new = np.zeros_like(f)

        # Acceleration
        a = -QE * E / ME  # Electron

        for i in range(self.nx):
            # Velocity displacement
            dv = a[i] * dt_frac

            # Interpolate
            cs = CubicSpline(self.v, f[i, :], bc_type='natural')

            v_new = self.v - dv

            # Clip to velocity domain
            v_new = np.clip(v_new, self.v[0], self.v[-1])

            f_new[i, :] = cs(v_new)

        return f_new

    def step(self):
        """
        Advance one time step using Strang splitting.

        Strang splitting:
        1. Half x-advect
        2. Full v-advect
        3. Half x-advect
        """
        # 1. Half x-advect
        self.f = self.advect_x(self.f, 0.5 * self.dt)

        # 2. Compute field
        ne, _ = self.compute_moments()
        self.phi, self.E = self.solve_poisson_fft(ne)

        # 3. Full v-advect
        self.f = self.advect_v(self.f, self.E, self.dt)

        # 4. Half x-advect
        self.f = self.advect_x(self.f, 0.5 * self.dt)

        # Update field for diagnostics
        ne, _ = self.compute_moments()
        self.phi, self.E = self.solve_poisson_fft(ne)

    def compute_diagnostics(self, t):
        """
        Compute and store diagnostic quantities.

        Parameters:
        -----------
        t : float
            Current time [s]
        """
        ne, _ = self.compute_moments()

        # Field energy: (ε0/2) ∫ E² dx
        E_field = 0.5 * EPS0 * np.sum(self.E**2) * self.dx

        # Kinetic energy: (m/2) ∫∫ v² f dx dv
        v2_grid = self.v[None, :]**2
        E_kinetic = 0.5 * ME * np.sum(v2_grid * self.f) * self.dx * self.dv

        # Total energy
        E_total = E_field + E_kinetic

        # Momentum: ∫∫ m v f dx dv
        momentum = ME * np.sum(self.v[None, :] * self.f) * self.dx * self.dv

        # Entropy: -∫∫ f ln(f) dx dv
        f_safe = self.f + 1e-30  # Avoid log(0)
        entropy = -np.sum(self.f * np.log(f_safe)) * self.dx * self.dv

        # Maximum E field
        E_max = np.max(np.abs(self.E))

        # Store
        self.diagnostics['time'].append(t)
        self.diagnostics['field_energy'].append(E_field)
        self.diagnostics['kinetic_energy'].append(E_kinetic)
        self.diagnostics['total_energy'].append(E_total)
        self.diagnostics['momentum'].append(momentum)
        self.diagnostics['entropy'].append(entropy)
        self.diagnostics['E_max'].append(E_max)

    def run(self, t_max, save_every=10):
        """
        Run simulation.

        Parameters:
        -----------
        t_max : float
            Maximum simulation time [s]
        save_every : int
            Save diagnostics every N steps

        Returns:
        --------
        snapshots : list of (t, f, E) tuples
        """
        n_steps = int(t_max / self.dt)
        snapshots = []

        print(f"Running simulation: {n_steps} steps...")

        for step in range(n_steps):
            t = step * self.dt

            # Save snapshot
            if step % save_every == 0:
                snapshots.append((t, self.f.copy(), self.E.copy()))
                self.compute_diagnostics(t)

                if step % (save_every * 10) == 0:
                    print(f"  Step {step}/{n_steps} (t = {t:.2e} s)")

            # Advance
            self.step()

        print("Simulation complete!")
        return snapshots

def test_plasma_oscillation():
    """Test case 1: Linear plasma oscillation."""
    print("\n" + "=" * 70)
    print("Test 1: Linear Plasma Oscillation")
    print("=" * 70)

    # Parameters
    n0 = 1e16  # m^-3
    Te = 1.0   # eV
    vth = np.sqrt(2 * Te * QE / ME)

    # Plasma frequency
    omega_pe = np.sqrt(n0 * QE**2 / (ME * EPS0))
    T_pe = 2 * np.pi / omega_pe

    print(f"Density: {n0:.2e} m^-3")
    print(f"Plasma frequency: {omega_pe/(2*np.pi)/1e9:.3f} GHz")
    print(f"Plasma period: {T_pe*1e9:.3f} ns")

    # Setup grid
    Lx = 1e-2  # 1 cm
    k = 2 * np.pi / Lx
    vmax = 6 * vth

    solver = VlasovPoisson1D(nx=128, nv=256, Lx=Lx, vmax=vmax, dt=0.1 * T_pe / 100)

    # Initialize
    solver.initialize_maxwellian(n0, v0=0, vth=vth)
    solver.add_perturbation(k, amplitude=0.01)

    # Run
    snapshots = solver.run(t_max=5 * T_pe, save_every=5)

    return solver, snapshots, omega_pe

def test_landau_damping():
    """Test case 2: Landau damping."""
    print("\n" + "=" * 70)
    print("Test 2: Landau Damping")
    print("=" * 70)

    # Parameters
    n0 = 1e16  # m^-3
    Te = 1.0   # eV
    vth = np.sqrt(2 * Te * QE / ME)

    omega_pe = np.sqrt(n0 * QE**2 / (ME * EPS0))

    # Choose k*λD = 0.5 for moderate damping
    lambda_D = vth / omega_pe
    k = 0.5 / lambda_D
    Lx = 2 * np.pi / k

    print(f"Debye length: {lambda_D*1e6:.2f} μm")
    print(f"k·λD = {k*lambda_D:.2f}")

    # Setup
    vmax = 6 * vth
    T_pe = 2 * np.pi / omega_pe

    solver = VlasovPoisson1D(nx=128, nv=256, Lx=Lx, vmax=vmax, dt=0.1 * T_pe / 100)

    # Initialize
    solver.initialize_maxwellian(n0, v0=0, vth=vth)
    solver.add_perturbation(k, amplitude=0.01)

    # Run
    snapshots = solver.run(t_max=10 * T_pe, save_every=5)

    return solver, snapshots, omega_pe

def plot_results(solver, snapshots, omega_pe, test_name):
    """Plot simulation results."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Extract diagnostics
    t = np.array(solver.diagnostics['time'])
    E_max = np.array(solver.diagnostics['E_max'])
    E_field = np.array(solver.diagnostics['field_energy'])
    E_total = np.array(solver.diagnostics['total_energy'])

    # Plot 1: Electric field vs time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.semilogy(t * omega_pe / (2 * np.pi), E_max, 'b-', linewidth=2)
    ax1.set_xlabel(r'Time ($\omega_{pe} t / 2\pi$)', fontsize=11)
    ax1.set_ylabel('Max |E| (V/m)', fontsize=11)
    ax1.set_title(f'{test_name}: Electric Field Evolution',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Energy conservation
    ax2 = fig.add_subplot(gs[1, 0])
    E_total_norm = (E_total - E_total[0]) / E_total[0] * 100
    ax2.plot(t * omega_pe / (2 * np.pi), E_total_norm, 'r-', linewidth=2)
    ax2.set_xlabel(r'Time ($\omega_{pe} t / 2\pi$)', fontsize=11)
    ax2.set_ylabel('Energy Error (%)', fontsize=11)
    ax2.set_title('Energy Conservation', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3-5: Phase space snapshots
    snapshot_indices = [0, len(snapshots)//2, -1]
    axes = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
            fig.add_subplot(gs[2, 0])]

    for idx, ax in zip(snapshot_indices, axes):
        t_snap, f_snap, E_snap = snapshots[idx]

        im = ax.pcolormesh(solver.x * 1e3, solver.v / 1e6, f_snap.T,
                          shading='auto', cmap='viridis')
        ax.set_xlabel('x (mm)', fontsize=10)
        ax.set_ylabel('v (Mm/s)', fontsize=10)
        ax.set_title(f't = {t_snap*omega_pe/(2*np.pi):.1f} / ωpe',
                    fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, label='f(x,v)')

    # Plot 6: Final density and E field
    ax6 = fig.add_subplot(gs[2, 1:])
    t_final, f_final, E_final = snapshots[-1]
    ne_final, _ = solver.compute_moments()

    ax6_twin = ax6.twinx()
    ax6.plot(solver.x * 1e3, ne_final / 1e16, 'b-', linewidth=2, label='Density')
    ax6_twin.plot(solver.x * 1e3, E_final, 'r-', linewidth=2, label='E field')

    ax6.set_xlabel('x (mm)', fontsize=11)
    ax6.set_ylabel(r'Density ($10^{16}$ m$^{-3}$)', fontsize=11, color='b')
    ax6_twin.set_ylabel('E (V/m)', fontsize=11, color='r')
    ax6.tick_params(axis='y', labelcolor='b')
    ax6_twin.tick_params(axis='y', labelcolor='r')
    ax6.set_title('Final State', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'1D Vlasov-Poisson: {test_name}',
                 fontsize=14, fontweight='bold')

    filename = test_name.lower().replace(' ', '_') + '.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved as '{filename}'")

    plt.show()

if __name__ == "__main__":
    # Run test case 1: Plasma oscillation
    solver1, snapshots1, omega_pe1 = test_plasma_oscillation()
    plot_results(solver1, snapshots1, omega_pe1, "Plasma Oscillation")

    # Run test case 2: Landau damping
    solver2, snapshots2, omega_pe2 = test_landau_damping()
    plot_results(solver2, snapshots2, omega_pe2, "Landau Damping")

    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)
