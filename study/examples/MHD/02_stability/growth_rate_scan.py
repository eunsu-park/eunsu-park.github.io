#!/usr/bin/env python3
"""
Growth Rate Scan Over Mode Numbers

Scans over (m,n) mode numbers and computes growth rate for each using
simplified stability analysis. Creates 2D heatmap of growth rates to
identify the most dangerous modes.

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson


def compute_growth_rate_simple(m, n, eq, r):
    """
    Compute growth rate using simplified dispersion relation.

    For MHD instabilities, growth rate scales roughly as:
    γ² ~ (pressure drive - magnetic stabilization) / inertia

    Parameters
    ----------
    m, n : int
        Mode numbers
    eq : dict
        Equilibrium quantities
    r : ndarray
        Radial grid

    Returns
    -------
    gamma : float
        Growth rate (s⁻¹)
    """
    mu_0 = 4 * np.pi * 1e-7

    B_theta = eq['B_theta']
    B_z = eq['B_z']
    p = eq['p']
    rho = eq['rho']

    # Wave number
    k = np.sqrt(m**2 + n**2) / r[-1]

    # Pressure gradient (destabilizing)
    dp_dr = np.gradient(p, r)
    pressure_drive = -np.mean(dp_dr[dp_dr < 0])  # Average negative gradient

    # Magnetic stabilization
    B_tot_sq = B_z**2 + np.mean(B_theta**2)
    magnetic_stabilization = k**2 * B_tot_sq / mu_0

    # Average density for inertia
    rho_avg = np.mean(rho)

    # Growth rate estimate
    gamma_sq = (pressure_drive - magnetic_stabilization) / (rho_avg + 1e-10)

    if gamma_sq > 0:
        gamma = np.sqrt(gamma_sq)
    else:
        gamma = 0.0

    # Scale by mode number (higher m,n typically have lower growth)
    gamma = gamma / (1 + 0.1 * (m + n))

    return gamma


def scan_mode_numbers(eq, r, m_max=8, n_max=8):
    """
    Scan over mode numbers and compute growth rates.

    Parameters
    ----------
    eq : dict
        Equilibrium
    r : ndarray
        Radial grid
    m_max, n_max : int
        Maximum mode numbers

    Returns
    -------
    gamma_map : ndarray
        2D array of growth rates
    m_arr, n_arr : ndarray
        Mode number arrays
    """
    m_arr = np.arange(0, m_max + 1)
    n_arr = np.arange(1, n_max + 1)

    gamma_map = np.zeros((len(m_arr), len(n_arr)))

    for i, m in enumerate(m_arr):
        for j, n in enumerate(n_arr):
            m_eff = max(m, 1)  # Avoid m=0
            gamma_map[i, j] = compute_growth_rate_simple(m_eff, n, eq, r)

    return gamma_map, m_arr, n_arr


def create_equilibrium(r):
    """
    Create Z-pinch equilibrium.

    Parameters
    ----------
    r : ndarray
        Radial grid

    Returns
    -------
    eq : dict
        Equilibrium quantities
    """
    mu_0 = 4 * np.pi * 1e-7
    r_max = r[-1]

    # Parameters
    B_z = 0.4  # T
    I_total = 60e3  # 60 kA

    # Current density
    J_z = 2 * I_total / (np.pi * r_max**2) * (1 - (r/r_max)**2)

    # Azimuthal field
    B_theta = np.zeros_like(r)
    for i, ri in enumerate(r):
        if ri > 1e-10:
            r_frac = ri / r_max
            I_enc = I_total * (2*r_frac**2 - r_frac**4)
            B_theta[i] = mu_0 * I_enc / (2 * np.pi * ri)

    # Pressure
    p = np.zeros_like(r)
    p_edge = 100.0
    p[-1] = p_edge

    for i in range(len(r)-2, -1, -1):
        dr = r[i+1] - r[i]
        p[i] = p[i+1] - J_z[i] * B_theta[i] * dr

    # Density
    rho_0 = 1e-3  # kg/m³
    rho = rho_0 * (1 - (r/r_max)**2)

    return {
        'B_z': B_z,
        'B_theta': B_theta,
        'J_z': J_z,
        'p': p,
        'rho': rho
    }


def plot_growth_rate_heatmap(gamma_map, m_arr, n_arr):
    """
    Plot 2D heatmap of growth rates.

    Parameters
    ----------
    gamma_map : ndarray
        Growth rate map
    m_arr, n_arr : ndarray
        Mode numbers
    """
    fig, ax = plt.subplots(figsize=(11, 8))

    # Create heatmap
    im = ax.imshow(gamma_map, cmap='hot', aspect='auto',
                   extent=[n_arr[0]-0.5, n_arr[-1]+0.5,
                          m_arr[-1]+0.5, m_arr[0]-0.5],
                   interpolation='bilinear')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Growth rate γ (s⁻¹)', fontsize=13)

    # Find and mark most dangerous mode
    max_idx = np.unravel_index(np.argmax(gamma_map), gamma_map.shape)
    m_max = m_arr[max_idx[0]]
    n_max = n_arr[max_idx[1]]
    gamma_max = gamma_map[max_idx]

    ax.plot(n_max, m_max, 'c*', markersize=25, markeredgecolor='white',
           markeredgewidth=2, label=f'Most unstable: ({m_max},{n_max})')

    # Contour lines
    if np.max(gamma_map) > 0:
        levels = np.linspace(0, np.max(gamma_map), 6)[1:]
        cs = ax.contour(n_arr, m_arr, gamma_map, levels=levels,
                       colors='cyan', linewidths=1.5, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=9, fmt='%.1e')

    ax.set_xlabel('Toroidal mode number n', fontsize=13)
    ax.set_ylabel('Poloidal mode number m', fontsize=13)
    ax.set_title('MHD Growth Rate Map: γ(m,n)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')

    # Grid
    ax.set_xticks(n_arr)
    ax.set_yticks(m_arr)
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig('growth_rate_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_growth_vs_mode(gamma_map, m_arr, n_arr):
    """
    Plot growth rate vs mode number for different n.

    Parameters
    ----------
    gamma_map : ndarray
        Growth rates
    m_arr, n_arr : ndarray
        Mode numbers
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: γ vs m for different n
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_arr)))

    for j, n in enumerate(n_arr):
        ax1.plot(m_arr, gamma_map[:, j], 'o-', color=colors[j],
                linewidth=2, markersize=6, label=f'n={n}')

    ax1.set_xlabel('Poloidal mode number m', fontsize=12)
    ax1.set_ylabel('Growth rate γ (s⁻¹)', fontsize=12)
    ax1.set_title('Growth Rate vs m (different n)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([m_arr[0]-0.5, m_arr[-1]+0.5])

    # Plot 2: γ vs n for different m
    ax2 = axes[1]
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(m_arr)))

    for i, m in enumerate(m_arr):
        if m <= 5:  # Plot only first few for clarity
            ax2.plot(n_arr, gamma_map[i, :], 's-', color=colors2[i],
                    linewidth=2, markersize=6, label=f'm={m}')

    ax2.set_xlabel('Toroidal mode number n', fontsize=12)
    ax2.set_ylabel('Growth rate γ (s⁻¹)', fontsize=12)
    ax2.set_title('Growth Rate vs n (different m)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([n_arr[0]-0.5, n_arr[-1]+0.5])

    plt.tight_layout()
    plt.savefig('growth_rate_vs_mode.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_most_dangerous_modes(gamma_map, m_arr, n_arr, top_n=10):
    """
    Plot bar chart of most dangerous modes.

    Parameters
    ----------
    gamma_map : ndarray
        Growth rates
    m_arr, n_arr : ndarray
        Mode numbers
    top_n : int
        Number of top modes to show
    """
    # Flatten and sort
    gamma_flat = gamma_map.flatten()
    idx_flat = np.argsort(gamma_flat)[::-1][:top_n]

    # Get mode numbers for top modes
    modes = []
    gammas = []

    for idx in idx_flat:
        i, j = np.unravel_index(idx, gamma_map.shape)
        m = m_arr[i]
        n = n_arr[j]
        gamma = gamma_map[i, j]

        modes.append(f'({m},{n})')
        gammas.append(gamma)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))

    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(gammas)))
    bars = ax.barh(modes, gammas, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Growth rate γ (s⁻¹)', fontsize=13)
    ax.set_ylabel('Mode (m,n)', fontsize=13)
    ax.set_title(f'Top {top_n} Most Dangerous Modes',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, val in zip(bars, gammas):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f' {val:.2e}',
               ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('most_dangerous_modes.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    # Create radial grid
    r_max = 0.01  # 1 cm
    r = np.linspace(r_max/200, r_max, 200)

    print("Growth Rate Scan Over Mode Numbers")
    print("=" * 60)
    print(f"Configuration: Z-pinch")
    print(f"  Radius: {r_max*100:.1f} cm")
    print()

    # Create equilibrium
    print("Creating equilibrium...")
    eq = create_equilibrium(r)

    print(f"  B_z: {eq['B_z']:.2f} T")
    print(f"  Peak B_θ: {np.max(eq['B_theta'])*1e3:.2f} mT")
    print(f"  Peak pressure: {np.max(eq['p'])/1e3:.2f} kPa")
    print()

    # Scan mode numbers
    m_max = 8
    n_max = 8
    print(f"Scanning mode numbers (0 ≤ m ≤ {m_max}, 1 ≤ n ≤ {n_max})...")

    gamma_map, m_arr, n_arr = scan_mode_numbers(eq, r, m_max, n_max)

    # Find most dangerous mode
    max_idx = np.unravel_index(np.argmax(gamma_map), gamma_map.shape)
    m_danger = m_arr[max_idx[0]]
    n_danger = n_arr[max_idx[1]]
    gamma_max = gamma_map[max_idx]

    print(f"\nMost dangerous mode: (m={m_danger}, n={n_danger})")
    print(f"  Growth rate: {gamma_max:.3e} s⁻¹")
    if gamma_max > 0:
        print(f"  Growth time: {1/gamma_max:.3e} s")
        print(f"  e-folding time: {1/gamma_max:.3e} s")

    # Count unstable modes
    n_unstable = np.sum(gamma_map > 1e-6)
    n_total = gamma_map.size
    print(f"\nUnstable modes: {n_unstable}/{n_total} ({100*n_unstable/n_total:.1f}%)")

    # Plot results
    print("\nGenerating plots...")
    plot_growth_rate_heatmap(gamma_map, m_arr, n_arr)
    print("  Heatmap saved as 'growth_rate_heatmap.png'")

    plot_growth_vs_mode(gamma_map, m_arr, n_arr)
    print("  Mode dependence saved as 'growth_rate_vs_mode.png'")

    plot_most_dangerous_modes(gamma_map, m_arr, n_arr, top_n=10)
    print("  Top modes saved as 'most_dangerous_modes.png'")


if __name__ == '__main__':
    main()
