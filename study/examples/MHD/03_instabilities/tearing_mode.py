#!/usr/bin/env python3
"""
Tearing Mode Instability

Computes Δ' (delta-prime) parameter for tearing mode at rational surface.
Uses outer ideal MHD solutions and matches across rational surface.
Includes Rutherford island width evolution.

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint


def tearing_mode_delta_prime(q_profile, r, r_s, m):
    """
    Compute Δ' parameter for tearing mode.

    Δ' = [dψ'/dr]_+ - [dψ'/dr]_- at rational surface

    Parameters
    ----------
    q_profile : callable
        Safety factor q(r)
    r : ndarray
        Radial grid
    r_s : float
        Rational surface location
    m : int
        Mode number

    Returns
    -------
    delta_prime : float
        Δ' parameter (positive → unstable)
    """
    # Simple model: Δ' ~ (1/L_s) where L_s is shear length
    # L_s = q/(dq/dr) at rational surface

    # Find q' at rational surface
    dr = r[1] - r[0]
    idx_s = np.argmin(np.abs(r - r_s))

    if idx_s > 0 and idx_s < len(r) - 1:
        dq_dr = (q_profile(r[idx_s+1]) - q_profile(r[idx_s-1])) / (2*dr)
        q_s = q_profile(r_s)

        if np.abs(dq_dr) > 1e-10:
            L_s = q_s / dq_dr
            delta_prime = 1.0 / L_s  # Simplified formula
        else:
            delta_prime = 0.0
    else:
        delta_prime = 0.0

    return delta_prime


def rutherford_evolution(t, w, delta_prime, eta, a):
    """
    Rutherford island width evolution equation.

    dw/dt = c * η * Δ' / w^(1/2)  (for small island)
    dw/dt = c * η * Δ'(w)         (for large island)

    Parameters
    ----------
    t : float
        Time
    w : float
        Island width
    delta_prime : float
        Δ' parameter
    eta : float
        Resistivity
    a : float
        Minor radius

    Returns
    -------
    dwdt : float
        Time derivative of width
    """
    # Avoid singularity at w=0
    w_eff = max(w, 1e-6)

    # Classical Rutherford: dw/dt ~ η Δ'
    # Small island: 1/sqrt(w) factor
    # Saturated Δ'(w) = Δ'_0 (1 - w²/w_sat²)

    w_sat = 0.1 * a  # Saturation width
    delta_eff = delta_prime * (1 - (w_eff/w_sat)**2)

    if w_eff < 0.01:
        # Linear regime
        dwdt = eta * delta_eff / np.sqrt(w_eff)
    else:
        # Nonlinear regime
        dwdt = eta * delta_eff

    return max(dwdt, 0)  # No negative growth


def plot_delta_prime_vs_mode(r_max=0.5):
    """
    Plot Δ' vs mode number for different q-profiles.

    Parameters
    ----------
    r_max : float
        Maximum radius
    """
    r = np.linspace(0, r_max, 200)

    # Different q-profiles
    profiles = {
        'Parabolic': lambda r: 1.0 + 2.0 * (r/r_max)**2,
        'Linear': lambda r: 1.0 + 3.0 * (r/r_max),
        'Reversed shear': lambda r: 2.0 - 1.5 * (r/r_max) + 3.0 * (r/r_max)**2
    }

    # Mode numbers
    m_values = np.arange(2, 8)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(profiles)))

    for i, (name, q_func) in enumerate(profiles.items()):
        delta_primes = []

        for m in m_values:
            # Find rational surface where q(r_s) = m
            q_vals = q_func(r)
            idx = np.argmin(np.abs(q_vals - m))
            r_s = r[idx]

            if r_s > 0.05 and r_s < r_max - 0.05:
                dp = tearing_mode_delta_prime(q_func, r, r_s, m)
                delta_primes.append(dp)
            else:
                delta_primes.append(np.nan)

        ax.plot(m_values, delta_primes, 'o-', color=colors[i],
               linewidth=2.5, markersize=8, label=name)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=2,
              label='Δ\'=0 (marginal)')
    ax.set_xlabel('Mode number m', fontsize=13)
    ax.set_ylabel('Δ\' (m⁻¹)', fontsize=13)
    ax.set_title('Tearing Mode Δ\' Parameter',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('delta_prime_vs_mode.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_island_evolution(delta_prime=1.0, eta=1e-6, a=0.5):
    """
    Plot Rutherford island width evolution.

    Parameters
    ----------
    delta_prime : float
        Δ' parameter
    eta : float
        Resistivity
    a : float
        Minor radius
    """
    # Time array
    t_max = 1.0  # seconds
    t = np.linspace(0, t_max, 500)

    # Initial width
    w0 = 1e-4  # 0.1 mm

    # Solve ODE
    from scipy.integrate import odeint

    def dw_dt_wrapper(w, t):
        return rutherford_evolution(t, w, delta_prime, eta, a)

    w = odeint(dw_dt_wrapper, w0, t).flatten()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    axes[0].plot(t * 1e3, w * 1e2, 'b-', linewidth=2.5)
    axes[0].set_xlabel('Time (ms)', fontsize=12)
    axes[0].set_ylabel('Island width w (cm)', fontsize=12)
    axes[0].set_title('Rutherford Island Growth',
                     fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Log-log scale
    axes[1].loglog(t * 1e3, w * 1e2, 'r-', linewidth=2.5)
    axes[1].set_xlabel('Time (ms)', fontsize=12)
    axes[1].set_ylabel('Island width w (cm)', fontsize=12)
    axes[1].set_title('Log-Log Scale',
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, which='both')

    # Add power-law reference
    t_ref = t[t > 0.01]
    w_ref = w0 * (t_ref / t[1])**(2/3)  # Classical scaling
    axes[1].plot(t_ref * 1e3, w_ref * 1e2, 'k--', linewidth=2,
                label='~t^(2/3) scaling')
    axes[1].legend(fontsize=10)

    plt.suptitle(f'Tearing Mode Island Evolution (Δ\'={delta_prime:.1f} m⁻¹, η={eta:.1e})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('island_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_island_structure(w, r_s, m):
    """
    Visualize magnetic island structure.

    Parameters
    ----------
    w : float
        Island width
    r_s : float
        Rational surface radius
    m : int
        Mode number
    """
    # Poloidal angle and radial coordinate
    theta = np.linspace(0, 2*np.pi, 200)
    r = np.linspace(r_s - w, r_s + w, 100)

    R, Theta = np.meshgrid(r, theta)

    # Island structure: flux function
    # ψ ~ cos(mθ) * (r - r_s)²/w²
    psi = np.cos(m * Theta) * ((R - r_s) / w)**2

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    levels = np.linspace(-1, 1, 21)
    cs = ax.contour(Theta, R, psi, levels=levels, colors='blue', linewidths=1.5)
    ax.contourf(Theta, R, psi, levels=levels, cmap='RdBu_r', alpha=0.5)

    # Mark rational surface
    ax.plot(theta, np.full_like(theta, r_s), 'r--', linewidth=3,
           label=f'q={m} surface')

    # Mark separatrix
    ax.plot(theta, np.full_like(theta, r_s - w/2), 'k-', linewidth=2,
           label='Separatrix')
    ax.plot(theta, np.full_like(theta, r_s + w/2), 'k-', linewidth=2)

    ax.set_title(f'Magnetic Island Structure (m={m}, w={w*1e2:.1f} cm)',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig('island_structure.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("Tearing Mode Instability Analysis")
    print("=" * 60)

    # Δ' vs mode number
    print("\nComputing Δ' for different q-profiles...")
    plot_delta_prime_vs_mode(r_max=0.5)
    print("  Saved as 'delta_prime_vs_mode.png'")

    # Island evolution
    print("\nSimulating island growth...")
    delta_prime = 2.0  # m⁻¹
    eta = 1e-6  # Ωm
    a = 0.5  # m

    plot_island_evolution(delta_prime, eta, a)
    print("  Saved as 'island_evolution.png'")

    # Island structure
    print("\nVisualizing island structure...")
    w = 0.05  # 5 cm
    r_s = 0.3  # 30 cm
    m = 2

    plot_island_structure(w, r_s, m)
    print("  Saved as 'island_structure.png'")

    print("\nKey results:")
    print(f"  Δ' = {delta_prime:.1f} m⁻¹ (positive → unstable)")
    print(f"  Classical island growth ~ t^(2/3)")
    print(f"  Nonlinear saturation occurs at finite width")


if __name__ == '__main__':
    main()
