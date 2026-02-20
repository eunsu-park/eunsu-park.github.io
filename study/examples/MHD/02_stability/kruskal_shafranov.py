#!/usr/bin/env python3
"""
Kruskal-Shafranov Stability Criterion

Analyzes the Kruskal-Shafranov criterion for kink instability:
    q(a) > m/n

where q(a) is the safety factor at the plasma edge, and (m,n) are
the poloidal and toroidal mode numbers.

Computes stability boundaries and maximum stable current vs aspect ratio.

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson


def current_profile(r, a, I_total, profile='parabolic', alpha=2.0):
    """
    Define current density profile.

    Parameters
    ----------
    r : array_like
        Radial positions
    a : float
        Plasma radius
    I_total : float
        Total plasma current
    profile : str
        Profile type: 'uniform', 'parabolic', 'peaked'
    alpha : float
        Profile peaking factor

    Returns
    -------
    J : ndarray
        Current density
    """
    if profile == 'uniform':
        J_0 = I_total / (np.pi * a**2)
        J = np.where(r <= a, J_0, 0.0)

    elif profile == 'parabolic':
        J_0 = (alpha + 1) * I_total / (np.pi * a**2)
        J = np.where(r <= a, J_0 * (1 - (r/a)**2)**alpha, 0.0)

    elif profile == 'peaked':
        # Highly peaked on axis
        J_0 = 3 * I_total / (np.pi * a**2)
        J = np.where(r <= a, J_0 * np.exp(-3*(r/a)**2), 0.0)

    else:
        raise ValueError(f"Unknown profile: {profile}")

    return J


def safety_factor_from_current(r, J, R0, B0, a):
    """
    Compute safety factor q(r) from current density.

    For circular cross-section:
        q(r) = (r * B_φ) / (R0 * B_θ)

    where B_θ(r) = μ₀ I(r) / (2π r)

    Parameters
    ----------
    r : ndarray
        Radial positions
    J : ndarray
        Current density
    R0 : float
        Major radius
    B0 : float
        Toroidal field on axis
    a : float
        Minor radius

    Returns
    -------
    q : ndarray
        Safety factor
    """
    mu_0 = 4 * np.pi * 1e-7

    q = np.zeros_like(r)

    for i, ri in enumerate(r):
        if ri < 1e-10:
            q[i] = 0.0
        else:
            # Enclosed current
            r_int = r[r <= ri]
            J_int = J[r <= ri]
            if len(r_int) > 1:
                I_enclosed = 2 * np.pi * simpson(r_int * J_int, x=r_int)
            else:
                I_enclosed = 0.0

            # Poloidal field
            B_theta = mu_0 * I_enclosed / (2 * np.pi * ri)

            # Toroidal field (assume ~ 1/R)
            B_phi = B0 * R0 / (R0 + ri * np.cos(0))  # At outboard midplane

            # Safety factor
            q[i] = ri * B_phi / (R0 * B_theta + 1e-10)

    return q


def kruskal_shafranov_criterion(q_edge, m, n):
    """
    Check Kruskal-Shafranov criterion.

    Criterion: q(a) > m/n for stability

    Parameters
    ----------
    q_edge : float
        Safety factor at edge
    m : int
        Poloidal mode number
    n : int
        Toroidal mode number

    Returns
    -------
    bool
        True if stable
    """
    return q_edge > m / n


def maximum_stable_current(R0, a, B0, m, n, profile='parabolic'):
    """
    Compute maximum stable current for given mode (m,n).

    Parameters
    ----------
    R0 : float
        Major radius
    a : float
        Minor radius
    B0 : float
        Toroidal field
    m, n : int
        Mode numbers
    profile : str
        Current profile type

    Returns
    -------
    I_max : float
        Maximum stable current
    """
    # Binary search for maximum current
    I_min, I_max = 1e3, 10e6  # Search range: 1 kA to 10 MA
    tolerance = 1e3  # 1 kA

    r = np.linspace(0, a, 200)

    while I_max - I_min > tolerance:
        I_test = (I_min + I_max) / 2

        J = current_profile(r, a, I_test, profile=profile)
        q = safety_factor_from_current(r, J, R0, B0, a)
        q_edge = q[-1]

        if kruskal_shafranov_criterion(q_edge, m, n):
            # Stable, can increase current
            I_min = I_test
        else:
            # Unstable, decrease current
            I_max = I_test

    return I_min


def plot_q_profile(r, q, a, m_values=[1, 2, 3], n=1):
    """
    Plot q profile with stability boundaries.

    Parameters
    ----------
    r : ndarray
        Radial grid
    q : ndarray
        Safety factor
    a : float
        Minor radius
    m_values : list
        Mode numbers to show
    n : int
        Toroidal mode number
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Normalized radius
    r_norm = r / a

    # Plot q profile
    ax.plot(r_norm, q, 'b-', linewidth=3, label='q(r)')

    # Plot stability boundaries
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    for i, m in enumerate(m_values):
        q_crit = m / n
        ax.axhline(y=q_crit, color=colors[i % len(colors)],
                  linestyle='--', linewidth=2,
                  label=f'q = {m}/{n} (m={m}, n={n})')

    # Mark q at edge
    q_edge = q[-1]
    ax.plot(1.0, q_edge, 'ro', markersize=12, label=f'q(a) = {q_edge:.2f}')

    ax.set_xlabel('Normalized radius r/a', fontsize=13)
    ax.set_ylabel('Safety factor q', fontsize=13)
    ax.set_title('Safety Factor Profile with Stability Boundaries',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, np.max(q) * 1.2])

    plt.tight_layout()
    plt.savefig('q_profile.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_stability_diagram(R0, B0, profile='parabolic'):
    """
    Plot stability diagram: I_max vs aspect ratio.

    Parameters
    ----------
    R0 : float
        Major radius
    B0 : float
        Toroidal field
    profile : str
        Current profile type
    """
    # Scan over aspect ratio
    epsilon_values = np.linspace(0.1, 0.5, 20)  # a/R0
    a_values = epsilon_values * R0

    # Compute max current for different modes
    modes = [(1, 1), (2, 1), (3, 1), (3, 2)]
    I_max_data = {mode: [] for mode in modes}

    print("Computing stability boundaries...")
    for a in a_values:
        for mode in modes:
            m, n = mode
            I_max = maximum_stable_current(R0, a, B0, m, n, profile=profile)
            I_max_data[mode].append(I_max / 1e6)  # Convert to MA

    # Plot
    fig, ax = plt.subplots(figsize=(11, 7))

    colors = ['red', 'blue', 'green', 'purple']
    markers = ['o', 's', '^', 'D']

    for i, mode in enumerate(modes):
        m, n = mode
        ax.plot(epsilon_values, I_max_data[mode],
               color=colors[i], marker=markers[i], markersize=7,
               linewidth=2, label=f'm/n = {m}/{n}')

    ax.set_xlabel('Inverse aspect ratio ε = a/R₀', fontsize=13)
    ax.set_ylabel('Maximum stable current I_p (MA)', fontsize=13)
    ax.set_title('Kruskal-Shafranov Stability Limit',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    # Add text box with parameters
    textstr = f'R₀ = {R0:.1f} m\nB₀ = {B0:.1f} T\nProfile: {profile}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('stability_boundary.png', dpi=150, bbox_inches='tight')
    plt.show()


def parameter_scan(R0, a, B0, profile='parabolic'):
    """
    Scan current and plot marginal stability curves.

    Parameters
    ----------
    R0 : float
        Major radius
    a : float
        Minor radius
    B0 : float
        Toroidal field
    profile : str
        Current profile type
    """
    # Current scan
    I_values = np.linspace(0.1e6, 5e6, 30)  # 0.1 to 5 MA
    r = np.linspace(0, a, 200)

    q_edge_values = []

    for I in I_values:
        J = current_profile(r, a, I, profile=profile)
        q = safety_factor_from_current(r, J, R0, B0, a)
        q_edge_values.append(q[-1])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(I_values / 1e6, q_edge_values, 'b-', linewidth=3,
           label='q(a) vs I_p')

    # Mark stability boundaries
    modes = [(1, 1), (2, 1), (3, 1), (3, 2)]
    colors = ['red', 'orange', 'green', 'purple']

    for i, (m, n) in enumerate(modes):
        q_crit = m / n
        ax.axhline(y=q_crit, color=colors[i], linestyle='--',
                  linewidth=2, label=f'm/n = {m}/{n}')

    ax.set_xlabel('Plasma current I_p (MA)', fontsize=13)
    ax.set_ylabel('Edge safety factor q(a)', fontsize=13)
    ax.set_title('Marginal Stability: q(a) vs Plasma Current',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    # Shade unstable regions
    for i, (m, n) in enumerate(modes):
        q_crit = m / n
        if i == 0:  # Only shade below q=1 for clarity
            ax.fill_between(I_values / 1e6, 0, q_crit,
                           where=(np.array(q_edge_values) < q_crit),
                           alpha=0.2, color='red',
                           label='Unstable region')

    plt.tight_layout()
    plt.savefig('marginal_stability.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    # Tokamak parameters
    R0 = 1.5  # Major radius (m)
    a = 0.5   # Minor radius (m)
    B0 = 2.5  # Toroidal field (T)
    I_p = 1.0e6  # Plasma current (A) = 1 MA
    profile = 'parabolic'

    print("Kruskal-Shafranov Stability Analysis")
    print("=" * 60)
    print(f"Tokamak parameters:")
    print(f"  Major radius R₀: {R0:.2f} m")
    print(f"  Minor radius a: {a:.2f} m")
    print(f"  Aspect ratio A = R₀/a: {R0/a:.2f}")
    print(f"  Toroidal field B₀: {B0:.2f} T")
    print(f"  Plasma current I_p: {I_p/1e6:.2f} MA")
    print(f"  Current profile: {profile}")
    print()

    # Compute q profile
    r = np.linspace(0, a, 200)
    J = current_profile(r, a, I_p, profile=profile)
    q = safety_factor_from_current(r, J, R0, B0, a)

    q_0 = q[10]  # Near axis (avoiding r=0 singularity)
    q_edge = q[-1]

    print(f"Safety factor:")
    print(f"  q(0) ≈ {q_0:.2f}")
    print(f"  q(a) = {q_edge:.2f}")
    print()

    # Check stability for different modes
    modes = [(1, 1), (2, 1), (3, 1), (3, 2), (4, 1)]
    print("Stability check (Kruskal-Shafranov criterion):")
    print(f"  {'Mode (m,n)':<12} {'q_crit':<8} {'Stable?'}")
    print("  " + "-" * 35)

    for m, n in modes:
        stable = kruskal_shafranov_criterion(q_edge, m, n)
        q_crit = m / n
        status = "✓ STABLE" if stable else "✗ UNSTABLE"
        print(f"  ({m},{n}){' '*(10-len(f'({m},{n})'))} {q_crit:<8.2f} {status}")

    print()

    # Plot q profile
    plot_q_profile(r, q, a, m_values=[1, 2, 3], n=1)
    print("q profile plot saved as 'q_profile.png'")

    # Plot stability boundary vs aspect ratio
    plot_stability_diagram(R0, B0, profile=profile)
    print("Stability boundary plot saved as 'stability_boundary.png'")

    # Parameter scan
    parameter_scan(R0, a, B0, profile=profile)
    print("Marginal stability plot saved as 'marginal_stability.png'")


if __name__ == '__main__':
    main()
