#!/usr/bin/env python3
"""
Tokamak Stability Analysis

Analysis of MHD stability in a tokamak plasma confinement device.
Computes safety factor profiles, checks Kruskal-Shafranov criterion,
and estimates tearing mode stability.

Key results:
- Safety factor q(r) profile for different current distributions
- Kruskal-Shafranov limit: q(a) > 2-3 for stability
- Tearing mode stability parameter Δ'
- Rational surfaces where resonant modes can develop

Physics:
- Safety factor: q = (r B_φ) / (R₀ B_θ) ≈ (r/R₀)(B_φ/B_θ)
- Kruskal-Shafranov: Prevents external kink modes
- Tearing modes: m,n modes at rational surfaces q = m/n

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, cumtrapz
from scipy.interpolate import interp1d
from typing import Tuple, Callable


class TokamakEquilibrium:
    """
    Tokamak equilibrium and stability analysis.

    Simple cylindrical model with:
    - Major radius R₀
    - Minor radius a
    - Toroidal field B_φ(r)
    - Poloidal field B_θ(r) from current profile j(r)
    """

    def __init__(self, R0: float = 1.0, a: float = 0.3,
                 B0: float = 2.0, I_p: float = 1.0e6):
        """
        Initialize tokamak parameters.

        Args:
            R0: Major radius (m)
            a: Minor radius (m)
            B0: Toroidal field at R0 (T)
            I_p: Total plasma current (A)
        """
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.I_p = I_p

        # Constants
        self.mu0 = 4 * np.pi * 1e-7  # Permeability of free space

    def safety_factor(self, r: np.ndarray, j_profile: Callable) -> np.ndarray:
        """
        Compute safety factor q(r).

        In cylindrical approximation:
        q(r) ≈ (r * B_φ) / (R₀ * B_θ(r))

        where B_θ(r) is computed from current profile via Ampere's law:
        B_θ(r) = (μ₀/2πr) ∫₀ʳ j(r') 2πr' dr' = (μ₀/r) ∫₀ʳ j(r') r' dr'

        Args:
            r: Radial coordinate array
            j_profile: Current density function j(r)

        Returns:
            Safety factor q(r)
        """
        # Compute current density
        j = j_profile(r)

        # Integrate to get enclosed current I(r)
        # I(r) = ∫₀ʳ j(r') 2πr' dr'
        integrand = j * r
        I_enclosed = 2 * np.pi * cumtrapz(integrand, r, initial=0)

        # Poloidal magnetic field from Ampere's law
        # B_θ(r) = μ₀ I(r) / (2π r)
        B_theta = np.zeros_like(r)
        B_theta[1:] = self.mu0 * I_enclosed[1:] / (2 * np.pi * r[1:])
        B_theta[0] = 0  # On axis

        # Toroidal field (approximately constant in simple model)
        B_phi = self.B0 * np.ones_like(r)

        # Safety factor
        q = np.zeros_like(r)
        q[1:] = (r[1:] * B_phi[1:]) / (self.R0 * B_theta[1:])

        # On-axis limit: q(0) ≈ 2 B₀ / (μ₀ R₀ j₀)
        if j[0] > 0:
            q[0] = 2 * B_phi[0] / (self.mu0 * self.R0 * j[0])
        else:
            q[0] = q[1]

        return q

    def kruskal_shafranov_criterion(self, q_a: float) -> Tuple[bool, str]:
        """
        Check Kruskal-Shafranov criterion for external kink stability.

        Criterion: q(a) > q_crit ≈ 2-3

        More precisely:
        - q(a) > 2/(m-1) for m-th mode stability
        - For m=2 (most dangerous): q(a) > 2
        - Empirically, q(a) > 3 provides good stability margin

        Args:
            q_a: Safety factor at plasma edge

        Returns:
            (stable, message)
        """
        if q_a > 3.0:
            return True, f"Stable (q(a)={q_a:.2f} > 3)"
        elif q_a > 2.0:
            return True, f"Marginally stable (q(a)={q_a:.2f}, 2 < q < 3)"
        else:
            return False, f"Unstable to kink (q(a)={q_a:.2f} < 2)"

    def find_rational_surfaces(self, r: np.ndarray, q: np.ndarray,
                              m_max: int = 5, n: int = 1) -> list:
        """
        Find rational surfaces where q = m/n.

        These are locations where resonant MHD modes can develop.

        Args:
            r: Radial coordinate
            q: Safety factor profile
            m_max: Maximum poloidal mode number
            n: Toroidal mode number (typically n=1)

        Returns:
            List of (m, n, r_res) tuples
        """
        rational_surfaces = []

        # Interpolate q(r) for finding roots
        q_interp = interp1d(r, q, kind='cubic', bounds_error=False, fill_value='extrapolate')

        for m in range(2, m_max + 1):
            q_res = m / n

            # Check if resonance exists in domain
            if q.min() < q_res < q.max():
                # Find radius where q(r) = m/n
                # Binary search
                r_search = r[(q > q_res * 0.9) & (q < q_res * 1.1)]
                if len(r_search) > 0:
                    # Refine using interpolation
                    r_fine = np.linspace(r_search.min(), r_search.max(), 1000)
                    q_fine = q_interp(r_fine)
                    idx = np.argmin(np.abs(q_fine - q_res))
                    r_res = r_fine[idx]

                    rational_surfaces.append((m, n, r_res))

        return rational_surfaces

    def tearing_mode_delta_prime(self, r: np.ndarray, q: np.ndarray,
                                  m: int, n: int) -> float:
        """
        Estimate tearing mode stability parameter Δ'.

        Δ' measures the jump in dψ/dr across the rational surface.
        - Δ' > 0: Unstable (tearing mode grows)
        - Δ' < 0: Stable (current profile resists tearing)

        Simplified estimate using current gradient:
        Δ' ≈ (2μ₀ R₀ / B_φ) * (dj/dr) / q' at q = m/n

        This is a crude approximation; real calculation requires
        solving the Grad-Shafranov equation and matching inner/outer solutions.

        Args:
            r: Radial coordinate
            q: Safety factor
            m, n: Mode numbers

        Returns:
            Δ' estimate (normalized)
        """
        # Find rational surface
        q_res = m / n
        rational_surfaces = self.find_rational_surfaces(r, q, m_max=m, n=n)

        if not any(surf[0] == m for surf in rational_surfaces):
            return 0.0  # No resonance

        r_res = next(surf[2] for surf in rational_surfaces if surf[0] == m)

        # Compute q' = dq/dr at resonance
        dq_dr = np.gradient(q, r)
        q_prime_interp = interp1d(r, dq_dr, kind='linear')
        q_prime_res = q_prime_interp(r_res)

        # Simplified Δ' estimate
        # Positive if current profile is peaked (dj/dr < 0 at r_res)
        # This is a very rough proxy
        delta_prime = -1.0 / (r_res * q_prime_res + 1e-10)  # Normalized

        return delta_prime


# =============================================================================
# Current Profile Models
# =============================================================================

def parabolic_current(alpha: float = 2.0):
    """
    Parabolic current profile: j(r) = j₀ (1 - (r/a)^α)

    Args:
        alpha: Peaking parameter (α=1: linear, α=2: parabolic)

    Returns:
        Function j(r) normalized to integrate to 1
    """
    def j_func(r: np.ndarray, a: float = 0.3) -> np.ndarray:
        rho = r / a
        j = (1 - rho**alpha)
        j[r > a] = 0
        # Normalize
        norm = np.trapz(j * r, r) * 2 * np.pi
        return j / (norm + 1e-10)

    return j_func


def hollow_current(r_peak: float = 0.5, width: float = 0.1):
    """
    Hollow current profile (off-axis peak).

    j(r) = j₀ exp(-(r - r_peak)² / (2*width²))

    Hollow current profiles can be unstable to tearing modes.

    Returns:
        Function j(r)
    """
    def j_func(r: np.ndarray, a: float = 0.3) -> np.ndarray:
        j = np.exp(-(r - r_peak * a)**2 / (2 * (width * a)**2))
        j[r > a] = 0
        # Normalize
        norm = np.trapz(j * r, r) * 2 * np.pi
        return j / (norm + 1e-10)

    return j_func


def bootstrap_current(alpha: float = 1.5, beta: float = 2.0):
    """
    Bootstrap current profile (typical in advanced tokamaks).

    Peaked off-axis due to pressure gradient.

    j(r) = j₀ (r/a)^α (1 - (r/a)^β)

    Returns:
        Function j(r)
    """
    def j_func(r: np.ndarray, a: float = 0.3) -> np.ndarray:
        rho = r / a
        j = rho**alpha * (1 - rho**beta)
        j[r > a] = 0
        j[r == 0] = 0
        # Normalize
        norm = np.trapz(j * r, r) * 2 * np.pi
        return j / (norm + 1e-10)

    return j_func


# =============================================================================
# Analysis and Visualization
# =============================================================================

def compare_current_profiles():
    """Compare different current profiles and their q-profiles."""
    print("=" * 70)
    print("Tokamak Current Profiles and Safety Factor Analysis")
    print("=" * 70)

    # Setup
    tokamak = TokamakEquilibrium(R0=1.0, a=0.3, B0=2.0, I_p=1.0e6)
    r = np.linspace(0, tokamak.a, 200)

    # Current profiles to compare
    profiles = {
        'Parabolic (α=2)': parabolic_current(alpha=2.0),
        'Parabolic (α=1)': parabolic_current(alpha=1.0),
        'Hollow': hollow_current(r_peak=0.6, width=0.15),
        'Bootstrap': bootstrap_current(alpha=1.5, beta=2.0)
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Results storage
    results = {}

    for idx, (name, j_func) in enumerate(profiles.items()):
        print(f"\n{name}:")
        print("-" * 40)

        # Compute profiles
        j = j_func(r, tokamak.a)
        q = tokamak.safety_factor(r, lambda x: j_func(x, tokamak.a))

        # Store results
        results[name] = {'r': r, 'j': j, 'q': q}

        # Safety factor at edge
        q_a = q[-1]
        print(f"  q(0) = {q[0]:.2f}")
        print(f"  q(a) = {q_a:.2f}")

        # Kruskal-Shafranov
        stable, msg = tokamak.kruskal_shafranov_criterion(q_a)
        print(f"  Kruskal-Shafranov: {msg}")

        # Rational surfaces
        rational = tokamak.find_rational_surfaces(r, q, m_max=5, n=1)
        print(f"  Rational surfaces q=m/n:")
        for m, n, r_res in rational:
            delta_prime = tokamak.tearing_mode_delta_prime(r, q, m, n)
            stability = "unstable" if delta_prime > 0 else "stable"
            print(f"    m/n={m}/{n} at r={r_res:.3f} m (r/a={r_res/tokamak.a:.2f}), "
                  f"Δ'={delta_prime:.2f} ({stability})")

        # Current density profile
        ax = axes[0, 0]
        ax.plot(r / tokamak.a, j / j.max(), label=name, linewidth=2)

        # Safety factor profile
        ax = axes[0, 1]
        ax.plot(r / tokamak.a, q, label=name, linewidth=2)

    # Finalize current density plot
    ax = axes[0, 0]
    ax.set_xlabel('r/a (normalized radius)')
    ax.set_ylabel('j(r) / j_max (normalized)')
    ax.set_title('Current Density Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Finalize safety factor plot
    ax = axes[0, 1]
    ax.set_xlabel('r/a')
    ax.set_ylabel('q(r)')
    ax.set_title('Safety Factor Profiles')
    ax.axhline(y=2, color='r', linestyle='--', label='q=2 (KS limit)')
    ax.axhline(y=3, color='orange', linestyle='--', label='q=3')
    # Mark rational surfaces
    for m in [2, 3, 4]:
        ax.axhline(y=m, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.text(0.02, m, f'q={m}', fontsize=8, color='gray')
    ax.set_ylim([0, 8])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Detailed view: Parabolic vs Hollow
    ax = axes[1, 0]
    for name in ['Parabolic (α=2)', 'Hollow']:
        r_plot = results[name]['r']
        q_plot = results[name]['q']
        ax.plot(r_plot / tokamak.a, q_plot, label=name, linewidth=2.5)

        # Mark rational surfaces
        rational = tokamak.find_rational_surfaces(r_plot, q_plot, m_max=5, n=1)
        for m, n, r_res in rational:
            ax.plot(r_res / tokamak.a, m/n, 'o', markersize=8)
            ax.text(r_res / tokamak.a + 0.02, m/n, f'{m}/{n}', fontsize=9)

    ax.set_xlabel('r/a')
    ax.set_ylabel('q(r)')
    ax.set_title('Rational Surfaces (m/n resonances)')
    ax.axhline(y=2, color='r', linestyle='--', alpha=0.5)
    ax.set_ylim([1, 6])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Shear profile: s = (r/q) dq/dr
    ax = axes[1, 1]
    for name in ['Parabolic (α=2)', 'Bootstrap']:
        r_plot = results[name]['r']
        q_plot = results[name]['q']

        # Magnetic shear
        dq_dr = np.gradient(q_plot, r_plot)
        shear = (r_plot / (q_plot + 1e-10)) * dq_dr

        ax.plot(r_plot / tokamak.a, shear, label=name, linewidth=2)

    ax.set_xlabel('r/a')
    ax.set_ylabel('Magnetic shear s')
    ax.set_title('Magnetic Shear Profile')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Numerical_Simulation/10_projects/tokamak_stability.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nFigure saved: tokamak_stability.png")


def stability_diagram():
    """
    Create stability diagram: q(a) vs q(0) with stability boundaries.
    """
    print("\n" + "=" * 70)
    print("Tokamak Stability Diagram")
    print("=" * 70)

    tokamak = TokamakEquilibrium(R0=1.0, a=0.3, B0=2.0, I_p=1.0e6)
    r = np.linspace(0, tokamak.a, 200)

    # Scan over different peaking parameters
    alphas = np.linspace(0.5, 4.0, 30)
    q0_values = []
    qa_values = []
    stable_flags = []

    for alpha in alphas:
        j_func = parabolic_current(alpha=alpha)
        q = tokamak.safety_factor(r, lambda x: j_func(x, tokamak.a))

        q0_values.append(q[0])
        qa_values.append(q[-1])

        stable, _ = tokamak.kruskal_shafranov_criterion(q[-1])
        stable_flags.append(stable)

    q0_values = np.array(q0_values)
    qa_values = np.array(qa_values)
    stable_flags = np.array(stable_flags)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by stability
    stable_idx = stable_flags == True
    unstable_idx = stable_flags == False

    ax.scatter(q0_values[stable_idx], qa_values[stable_idx],
              c='green', s=50, alpha=0.7, label='Stable', zorder=3)
    ax.scatter(q0_values[unstable_idx], qa_values[unstable_idx],
              c='red', s=50, alpha=0.7, label='Unstable (kink)', zorder=3)

    # Stability boundaries
    ax.axhline(y=2, color='red', linestyle='--', linewidth=2, label='q(a)=2 (KS limit)')
    ax.axhline(y=3, color='orange', linestyle='--', linewidth=2, label='q(a)=3')

    # Optimal region
    ax.fill_between([0, 10], 2, 3, alpha=0.1, color='orange', label='Marginal')
    ax.fill_between([0, 10], 3, 10, alpha=0.1, color='green', label='Safe region')

    ax.set_xlabel('q(0) (on-axis safety factor)', fontsize=12)
    ax.set_ylabel('q(a) (edge safety factor)', fontsize=12)
    ax.set_title('Tokamak Stability Diagram (Kruskal-Shafranov)', fontsize=14)
    ax.set_xlim([0.5, 3])
    ax.set_ylim([1.5, 6])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Numerical_Simulation/10_projects/tokamak_stability_diagram.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure saved: tokamak_stability_diagram.png")

    print("\nKey findings:")
    print(f"  - Stable configurations: q(a) > 2-3")
    print(f"  - Typical range: q(0) ≈ 1-2, q(a) ≈ 3-5")
    print(f"  - Lower q(0) (peaked current) → higher q(a) needed for stability")


def main():
    """Run complete tokamak stability analysis."""
    compare_current_profiles()
    stability_diagram()

    print("\n" + "=" * 70)
    print("Summary: Tokamak MHD Stability")
    print("=" * 70)
    print("""
    Key Stability Criteria:

    1. Kruskal-Shafranov Criterion (External Kink):
       - q(a) > 2: Stable to m=2 kink mode
       - q(a) > 3: Safe operating margin
       - Prevents large-scale disruptions

    2. Rational Surfaces (Tearing Modes):
       - Occur where q = m/n (m,n integers)
       - Most dangerous: m/n = 2/1, 3/2
       - Δ' > 0 → unstable (islands grow)
       - Hollow current profiles more susceptible

    3. Current Profile Effects:
       - Parabolic (broad): High q(a), good stability
       - Peaked: Low q(a), may violate KS criterion
       - Hollow: Tearing mode instabilities
       - Bootstrap: Advanced scenarios, needs careful control

    4. Magnetic Shear s = (r/q) dq/dr:
       - Positive shear (standard): Generally stable
       - Negative shear (reversed): Can stabilize turbulence
       - Low/zero shear: Enhanced confinement, but instabilities

    Operational Implications:
    - ITER design: q(a) ≈ 3, q(0) ≈ 1 (safety margin)
    - Real-time control needed to maintain q-profile
    - Advanced scenarios explore low-shear, high-β regimes
    - Disruption avoidance critical for machine protection

    Limitations of This Analysis:
    - Cylindrical approximation (real tokamaks are toroidal)
    - No finite pressure effects (β)
    - No kinetic effects
    - Simplified Δ' calculation
    """)


if __name__ == "__main__":
    main()
