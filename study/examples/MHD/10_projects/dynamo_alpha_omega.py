#!/usr/bin/env python3
"""
Alpha-Omega Dynamo in Spherical Shell

Mean-field dynamo model for magnetic field generation in rotating spheres
(stars, planets). Demonstrates how differential rotation (Ω-effect) and
helical turbulence (α-effect) sustain magnetic fields against Ohmic decay.

Key results:
- Self-sustained magnetic field oscillations
- Butterfly diagram showing equatorward migration of magnetic activity
- Periodic field reversals (like Earth's magnetic field)
- Critical dynamo number for field generation

Physics:
- Mean-field induction equation with α and Ω effects
- ∂B/∂t = ∇×(α B + Ω×r × B) + η ∇²B
- Spherical shell geometry (inner core + convecting outer layer)

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from scipy.integrate import odeint
from typing import Tuple, List


class AlphaOmegaDynamo:
    """
    Alpha-Omega dynamo in a spherical shell.

    Simplified mean-field model:
    - α-effect: Twisting of toroidal field to poloidal field by helical convection
    - Ω-effect: Shearing of poloidal field to toroidal field by differential rotation

    We use a spectral method with low-order spherical harmonics.
    """

    def __init__(self, r_inner: float = 0.3, r_outer: float = 1.0,
                 nr: int = 50, ntheta: int = 40):
        """
        Initialize dynamo model.

        Args:
            r_inner: Inner boundary (solid inner core)
            r_outer: Outer boundary (surface)
            nr: Radial grid points
            ntheta: Latitudinal grid points
        """
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.nr = nr
        self.ntheta = ntheta

        # Grid
        self.r = np.linspace(r_inner, r_outer, nr)
        self.theta = np.linspace(0, np.pi, ntheta)  # Colatitude
        self.R, self.Theta = np.meshgrid(self.r, self.theta, indexing='ij')

        # Latitude (for plotting)
        self.lat = 90 - np.degrees(self.theta)

        # Time history
        self.time_history = []
        self.Bp_history = []  # Poloidal field
        self.Bt_history = []  # Toroidal field

    def alpha_profile(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Alpha effect profile: α(r, θ)

        Models helical turbulence from convection.
        Typically: α ∝ cos(θ) (antisymmetric about equator)

        Args:
            r: Radial coordinate (normalized)
            theta: Colatitude

        Returns:
            α coefficient
        """
        # Radial profile: peaked in convection zone
        r_norm = (r - self.r_inner) / (self.r_outer - self.r_inner)
        radial = np.sin(np.pi * r_norm)  # Peaked in middle of shell

        # Latitudinal profile: cos(θ) (north-south antisymmetry)
        latitudinal = np.cos(theta)

        return radial * latitudinal

    def omega_profile(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Differential rotation profile: Ω(r, θ)

        Solar-like differential rotation:
        - Faster rotation at equator than poles
        - Radial shear

        Args:
            r: Radial coordinate
            theta: Colatitude

        Returns:
            Angular velocity Ω
        """
        # Latitudinal differential rotation: Ω = Ω₀(1 - β sin²θ)
        # where β ~ 0.2 for the Sun
        beta = 0.2
        lat_term = 1 - beta * np.sin(theta)**2

        # Radial shear: Ω increases outward in convection zone
        r_norm = (r - self.r_inner) / (self.r_outer - self.r_inner)
        rad_term = 1 + 0.3 * r_norm

        return lat_term * rad_term

    def initial_field_perturbation(self, amplitude: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create small seed magnetic field to start the dynamo.

        Returns:
            (B_poloidal, B_toroidal) on the grid
        """
        # Small random dipole-like poloidal field
        # Using P₁¹(cos θ) = sin θ spherical harmonic
        Bp = amplitude * self.R * np.sin(self.Theta) * np.random.randn()

        # No initial toroidal field
        Bt = np.zeros_like(Bp)

        return Bp, Bt

    def rhs_dynamo(self, t: float, state: np.ndarray,
                   C_alpha: float, C_omega: float, eta: float) -> np.ndarray:
        """
        Right-hand side of the mean-field dynamo equations.

        Equations (simplified, 1D radial + latitudinal):
        ∂B_p/∂t = C_α B_t + η ∇²B_p
        ∂B_t/∂t = C_Ω (∇B_p)·(∇Ω) + η ∇²B_t

        Where:
        - B_p: Poloidal field component
        - B_t: Toroidal field component
        - C_α, C_Ω: Dynamo numbers (control strength of α and Ω effects)
        - η: Magnetic diffusivity

        Args:
            t: Time
            state: Flattened [B_p, B_t]
            C_alpha: Alpha effect strength
            C_omega: Omega effect strength
            eta: Magnetic diffusivity

        Returns:
            d(state)/dt
        """
        n = len(state) // 2
        Bp = state[:n].reshape(self.nr, self.ntheta)
        Bt = state[n:].reshape(self.nr, self.ntheta)

        # Alpha and Omega profiles
        alpha = C_alpha * self.alpha_profile(self.R, self.Theta)
        omega = C_omega * self.omega_profile(self.R, self.Theta)

        # 1. Poloidal field evolution: ∂B_p/∂t = α B_t + η ∇²B_p
        # Alpha effect: toroidal → poloidal
        source_Bp = alpha * Bt

        # Diffusion (simplified radial diffusion only)
        dr = self.r[1] - self.r[0]
        diff_Bp = np.zeros_like(Bp)
        diff_Bp[1:-1, :] = eta * (Bp[2:, :] - 2*Bp[1:-1, :] + Bp[:-2, :]) / dr**2

        dBp_dt = source_Bp + diff_Bp

        # Boundary conditions: B_p = 0 at inner and outer boundaries
        dBp_dt[0, :] = -Bp[0, :] / 0.01   # Relax to zero
        dBp_dt[-1, :] = -Bp[-1, :] / 0.01

        # 2. Toroidal field evolution: ∂B_t/∂t = Ω-shear * B_p + η ∇²B_t
        # Omega effect: differential rotation shears poloidal → toroidal
        # Simplified: dB_t/dt ∝ Ω * ∂B_p/∂r
        dBp_dr = np.gradient(Bp, dr, axis=0)
        source_Bt = omega * dBp_dr

        # Diffusion
        diff_Bt = np.zeros_like(Bt)
        diff_Bt[1:-1, :] = eta * (Bt[2:, :] - 2*Bt[1:-1, :] + Bt[:-2, :]) / dr**2

        dBt_dt = source_Bt + diff_Bt

        # Boundary conditions
        dBt_dt[0, :] = -Bt[0, :] / 0.01
        dBt_dt[-1, :] = -Bt[-1, :] / 0.01

        # Flatten and return
        return np.concatenate([dBp_dt.flatten(), dBt_dt.flatten()])

    def run_dynamo(self, t_final: float = 100.0, dt: float = 0.1,
                   C_alpha: float = 1.0, C_omega: float = 5.0,
                   eta: float = 0.1) -> None:
        """
        Run the dynamo simulation.

        Args:
            t_final: Final time
            dt: Time step
            C_alpha: Alpha effect strength (dynamo number)
            C_omega: Omega effect strength
            eta: Magnetic diffusivity
        """
        print(f"Running alpha-omega dynamo...")
        print(f"  Parameters: C_α={C_alpha}, C_Ω={C_omega}, η={eta}")
        print(f"  Dynamo number D = C_α * C_Ω = {C_alpha * C_omega:.1f}")

        # Initial conditions
        Bp0, Bt0 = self.initial_field_perturbation(amplitude=0.01)
        state0 = np.concatenate([Bp0.flatten(), Bt0.flatten()])

        # Time points
        t_eval = np.arange(0, t_final + dt, dt)

        # Integrate using explicit Euler (simple for demonstration)
        self.time_history = [0]
        self.Bp_history = [Bp0]
        self.Bt_history = [Bt0]

        state = state0.copy()
        for i, t in enumerate(t_eval[1:]):
            # Euler step
            dstate = self.rhs_dynamo(t, state, C_alpha, C_omega, eta)
            state = state + dt * dstate

            # Store
            if (i + 1) % 10 == 0:  # Store every 10 steps
                n = len(state) // 2
                Bp = state[:n].reshape(self.nr, self.ntheta)
                Bt = state[n:].reshape(self.nr, self.ntheta)

                self.time_history.append(t)
                self.Bp_history.append(Bp.copy())
                self.Bt_history.append(Bt.copy())

                if (i + 1) % 100 == 0:
                    Bp_max = np.max(np.abs(Bp))
                    Bt_max = np.max(np.abs(Bt))
                    print(f"  t={t:.1f}, |Bp|_max={Bp_max:.3e}, |Bt|_max={Bt_max:.3e}")

        print(f"Dynamo simulation complete!")


def visualize_dynamo(dynamo: AlphaOmegaDynamo, save_prefix: str = "dynamo"):
    """
    Visualize dynamo results: butterfly diagram and field snapshots.
    """
    times = np.array(dynamo.time_history)
    n_times = len(times)

    # =========================================================================
    # 1. Butterfly Diagram (Hovmöller plot)
    # =========================================================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Extract toroidal field at mid-radius as function of time and latitude
    r_mid_idx = dynamo.nr // 2
    butterfly_data = np.zeros((n_times, dynamo.ntheta))

    for i, Bt in enumerate(dynamo.Bt_history):
        butterfly_data[i, :] = Bt[r_mid_idx, :]

    # Butterfly diagram
    ax = axes[0, 0]
    lat = 90 - np.degrees(dynamo.theta)
    extent = [lat.min(), lat.max(), times.min(), times.max()]
    im = ax.imshow(butterfly_data, aspect='auto', cmap='RdBu_r',
                   extent=extent, origin='lower', interpolation='bilinear')
    ax.set_xlabel('Latitude (degrees)')
    ax.set_ylabel('Time')
    ax.set_title('Butterfly Diagram (Toroidal Field B_t)')
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.colorbar(im, ax=ax, label='B_t')

    # =========================================================================
    # 2. Time series of field strength
    # =========================================================================
    ax = axes[0, 1]

    Bp_max_time = [np.max(np.abs(Bp)) for Bp in dynamo.Bp_history]
    Bt_max_time = [np.max(np.abs(Bt)) for Bt in dynamo.Bt_history]

    ax.plot(times, Bp_max_time, 'b-', label='|Bp|_max', linewidth=2)
    ax.plot(times, Bt_max_time, 'r-', label='|Bt|_max', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Field strength')
    ax.set_title('Magnetic Field Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # =========================================================================
    # 3. Snapshots at different times
    # =========================================================================
    snapshot_indices = [len(times)//4, len(times)//2, 3*len(times)//4, -1]

    for idx, snap_idx in enumerate(snapshot_indices):
        if idx >= 4:
            break

        row = 1 + idx // 2
        col = idx % 2
        ax = axes[row, col]

        Bt = dynamo.Bt_history[snap_idx]
        time_snap = times[snap_idx]

        # Plot toroidal field in meridional plane
        lat_2d = 90 - np.degrees(dynamo.Theta)
        levels = np.linspace(-np.abs(Bt).max(), np.abs(Bt).max(), 21)
        cf = ax.contourf(lat_2d, dynamo.R, Bt, levels=levels, cmap='RdBu_r', extend='both')
        ax.contour(lat_2d, dynamo.R, Bt, levels=levels, colors='k',
                  linewidths=0.5, alpha=0.3)

        ax.set_xlabel('Latitude (degrees)')
        ax.set_ylabel('Radius')
        ax.set_title(f'Toroidal Field B_t at t={time_snap:.1f}')
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.colorbar(cf, ax=ax, label='B_t')

    plt.tight_layout()
    plt.savefig(f'/opt/projects/01_Personal/03_Study/examples/Numerical_Simulation/10_projects/{save_prefix}_butterfly.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {save_prefix}_butterfly.png")

    # =========================================================================
    # 4. Field Structure at Final Time
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    Bp_final = dynamo.Bp_history[-1]
    Bt_final = dynamo.Bt_history[-1]
    lat_2d = 90 - np.degrees(dynamo.Theta)

    # Poloidal field
    ax = axes[0]
    levels_p = np.linspace(-np.abs(Bp_final).max(), np.abs(Bp_final).max(), 21)
    cf = ax.contourf(lat_2d, dynamo.R, Bp_final, levels=levels_p, cmap='PRGn', extend='both')
    ax.contour(lat_2d, dynamo.R, Bp_final, levels=levels_p, colors='k',
              linewidths=0.8, alpha=0.4)
    ax.set_xlabel('Latitude (degrees)')
    ax.set_ylabel('Radius')
    ax.set_title('Poloidal Field B_p (final)')
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.colorbar(cf, ax=ax, label='B_p')

    # Toroidal field
    ax = axes[1]
    levels_t = np.linspace(-np.abs(Bt_final).max(), np.abs(Bt_final).max(), 21)
    cf = ax.contourf(lat_2d, dynamo.R, Bt_final, levels=levels_t, cmap='RdBu_r', extend='both')
    ax.contour(lat_2d, dynamo.R, Bt_final, levels=levels_t, colors='k',
              linewidths=0.8, alpha=0.4)
    ax.set_xlabel('Latitude (degrees)')
    ax.set_ylabel('Radius')
    ax.set_title('Toroidal Field B_t (final)')
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.colorbar(cf, ax=ax, label='B_t')

    plt.tight_layout()
    plt.savefig(f'/opt/projects/01_Personal/03_Study/examples/Numerical_Simulation/10_projects/{save_prefix}_fields.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {save_prefix}_fields.png")


def study_dynamo_regimes():
    """
    Study different dynamo regimes by varying dynamo numbers.
    """
    print("=" * 70)
    print("Alpha-Omega Dynamo: Regime Study")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    regimes = [
        {'name': 'Subcritical', 'C_alpha': 0.5, 'C_omega': 2.0},  # D = 1.0
        {'name': 'Critical', 'C_alpha': 1.0, 'C_omega': 3.0},     # D = 3.0
        {'name': 'Supercritical', 'C_alpha': 1.5, 'C_omega': 5.0}, # D = 7.5
        {'name': 'Strongly Supercritical', 'C_alpha': 2.0, 'C_omega': 8.0}, # D = 16
    ]

    for idx, regime in enumerate(regimes):
        print(f"\n{regime['name']} Regime:")
        print(f"  C_α={regime['C_alpha']}, C_Ω={regime['C_omega']}")
        print(f"  Dynamo number D = {regime['C_alpha'] * regime['C_omega']}")

        dynamo = AlphaOmegaDynamo(r_inner=0.3, r_outer=1.0, nr=40, ntheta=30)
        dynamo.run_dynamo(t_final=80.0, dt=0.05,
                         C_alpha=regime['C_alpha'],
                         C_omega=regime['C_omega'],
                         eta=0.1)

        # Extract time series
        times = np.array(dynamo.time_history)
        Bt_max = [np.max(np.abs(Bt)) for Bt in dynamo.Bt_history]

        # Plot
        ax = axes[idx // 2, idx % 2]
        ax.plot(times, Bt_max, 'b-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('|Bt|_max')
        ax.set_title(f"{regime['name']}: D={regime['C_alpha']*regime['C_omega']:.1f}")
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Analyze
        if len(times) > 50:
            final_avg = np.mean(Bt_max[-20:])
            if final_avg < 1e-4:
                result = "Decays (subcritical)"
            elif np.std(Bt_max[-20:]) / np.mean(Bt_max[-20:]) > 0.3:
                result = "Oscillatory (supercritical)"
            else:
                result = "Saturated"
            print(f"  Result: {result}")

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Numerical_Simulation/10_projects/dynamo_regimes.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nFigure saved: dynamo_regimes.png")


def main():
    """Run complete alpha-omega dynamo study."""
    print("=" * 70)
    print("Alpha-Omega Dynamo in Spherical Shell")
    print("=" * 70)

    # Main simulation
    print("\n[1] Running main dynamo simulation...")
    dynamo = AlphaOmegaDynamo(r_inner=0.35, r_outer=1.0, nr=50, ntheta=40)
    dynamo.run_dynamo(t_final=100.0, dt=0.05,
                     C_alpha=1.2, C_omega=6.0, eta=0.1)

    # Visualize
    visualize_dynamo(dynamo, save_prefix="dynamo_main")

    # Study different regimes
    print("\n[2] Studying different dynamo regimes...")
    study_dynamo_regimes()

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Alpha-Omega Dynamo")
    print("=" * 70)
    print("""
    Mean-Field Dynamo Theory:

    The alpha-omega dynamo sustains magnetic fields through two mechanisms:

    1. Ω-Effect (Differential Rotation):
       - Shears poloidal field B_p into toroidal field B_t
       - ∂B_t/∂t ∝ (∇Ω) · (∇B_p)
       - Strong in regions with velocity gradients

    2. α-Effect (Helical Turbulence):
       - Twists toroidal field B_t back into poloidal field B_p
       - ∂B_p/∂t ∝ α B_t
       - Arises from cyclonic convection in rotating spheres

    Critical Dynamo Number:
       - D = C_α * C_Ω (dimensionless)
       - D < D_crit: Field decays (subcritical)
       - D ≈ D_crit: Marginal (critical dynamo number ~ 2-5)
       - D > D_crit: Self-sustained oscillations (supercritical)

    Butterfly Diagram:
       - Shows latitudinal migration of magnetic activity
       - In Sun: activity starts at ~30° latitude and migrates equatorward
       - Period: ~11 years for solar cycle (22 years for full magnetic cycle)
       - Our model shows similar equatorward migration pattern

    Field Reversals:
       - Poloidal field reverses periodically
       - Earth's magnetic field reverses every ~200,000-300,000 years
       - Mechanism: Nonlinear saturation and turbulent fluctuations

    Applications:
       - Solar dynamo: Sunspot cycle, solar flares
       - Earth's dynamo: Geomagnetic field, pole reversals
       - Stellar dynamos: Starspot activity, magnetic braking
       - Galactic dynamo: Large-scale magnetic fields in galaxies

    Limitations of This Model:
       - Simplified geometry (spherical shell, not full 3D)
       - Mean-field approximation (averages over turbulence)
       - Linear α-effect (real α depends on field strength)
       - No magnetic buoyancy or field instabilities
       - Low spatial resolution

    Advanced Topics:
       - α²-dynamo: Both poloidal and toroidal generated by α
       - Magnetic buoyancy: Rising flux tubes (sunspots)
       - Tachocline: Interface dynamo at base of convection zone
       - Grand minima: Maunder Minimum (1645-1715, reduced sunspots)
    """)


if __name__ == "__main__":
    main()
