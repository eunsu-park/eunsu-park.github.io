#!/usr/bin/env python3
"""
External Kink Instability

Analyzes external kink mode in sharp-boundary cylindrical plasma.
Computes dispersion relation for surface modes and shows stabilization
by conducting wall.

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv, kn, ivp, kvp  # Modified Bessel functions


def kink_dispersion(q_a, m, k_z, a, b_wall=None):
    """
    Compute kink mode growth rate.

    For m=1 external kink, stability requires q(a) > m.

    Parameters
    ----------
    q_a : float
        Safety factor at edge
    m : int
        Poloidal mode number
    k_z : float
        Axial wavenumber
    a : float
        Plasma radius
    b_wall : float or None
        Wall radius (None for no wall)

    Returns
    -------
    gamma : float
        Growth rate (normalized)
    """
    # Normalized growth rate (simplified)
    # Unstable if q(a) < m
    if q_a < m:
        gamma_sq = (m - q_a) / (m + 1)  # Simplified formula
    else:
        gamma_sq = 0

    # Wall stabilization
    if b_wall is not None and b_wall > a:
        # Wall reduces growth rate
        wall_factor = (a / b_wall)**m
        gamma_sq *= (1 - wall_factor)

    gamma = np.sqrt(max(gamma_sq, 0))
    return gamma


def plot_growth_vs_q(m=1, k_z_values=[0.1, 0.5, 1.0]):
    """
    Plot growth rate vs q(a) for m=1 kink.

    Parameters
    ----------
    m : int
        Mode number
    k_z_values : list
        Axial wavenumbers
    """
    q_a = np.linspace(0.5, 3.0, 100)
    a = 0.5  # m

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.plasma(np.linspace(0, 1, len(k_z_values)))

    for i, k_z in enumerate(k_z_values):
        gamma = np.array([kink_dispersion(q, m, k_z, a) for q in q_a])

        ax.plot(q_a, gamma, color=colors[i], linewidth=2.5,
               label=f'k_z = {k_z:.1f} m⁻¹')

    ax.axvline(x=m, color='red', linestyle='--', linewidth=2,
              label=f'q = {m} (marginal stability)')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    ax.set_xlabel('Edge safety factor q(a)', fontsize=13)
    ax.set_ylabel('Normalized growth rate γ', fontsize=13)
    ax.set_title(f'External Kink Mode (m={m})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Shade unstable region
    ax.fill_betweenx([0, ax.get_ylim()[1]], 0, m, alpha=0.2, color='red',
                     label='Unstable')

    plt.tight_layout()
    plt.savefig('kink_growth_vs_q.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_wall_stabilization(m=1):
    """
    Show wall stabilization effect.

    Parameters
    ----------
    m : int
        Mode number
    """
    q_a_values = [0.7, 0.8, 0.9]  # < 1, unstable without wall
    b_a_ratio = np.linspace(1.0, 3.0, 100)  # b/a

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(q_a_values)))

    a = 0.5
    k_z = 0.5

    for i, q_a in enumerate(q_a_values):
        gamma_arr = []

        for ratio in b_a_ratio:
            b = ratio * a
            gamma = kink_dispersion(q_a, m, k_z, a, b_wall=b)
            gamma_arr.append(gamma)

        # No-wall case
        gamma_no_wall = kink_dispersion(q_a, m, k_z, a, b_wall=None)

        ax.plot(b_a_ratio, gamma_arr, color=colors[i], linewidth=2.5,
               label=f'q(a)={q_a:.1f} (no wall: γ={gamma_no_wall:.2f})')

    ax.set_xlabel('Wall radius ratio b/a', fontsize=13)
    ax.set_ylabel('Normalized growth rate γ', fontsize=13)
    ax.set_title('Conducting Wall Stabilization of Kink Mode',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 3])

    plt.tight_layout()
    plt.savefig('kink_wall_stabilization.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_mode_structure(m=1, n=1):
    """
    Visualize kink mode structure.

    Parameters
    ----------
    m, n : int
        Mode numbers
    """
    # Cylindrical grid
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, 2*np.pi/n, 100)
    Theta, Z = np.meshgrid(theta, z)

    # Mode structure: ξ ~ exp(i(mθ + nz))
    # Visualize real part
    xi_r = np.cos(m * Theta + n * Z)

    fig = plt.figure(figsize=(12, 5))

    # 2D plot
    ax1 = fig.add_subplot(121)
    im1 = ax1.contourf(Theta, Z, xi_r, levels=20, cmap='RdBu_r')
    ax1.set_xlabel('Poloidal angle θ (rad)', fontsize=12)
    ax1.set_ylabel('Axial position z (rad)', fontsize=12)
    ax1.set_title(f'Kink Mode Structure (m={m}, n={n})',
                 fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Displacement ξ_r')

    # 3D-like view (unrolled cylinder)
    ax2 = fig.add_subplot(122, projection='3d')
    R = 1.0
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    surf = ax2.plot_surface(X, Y, Z, facecolors=plt.cm.RdBu_r((xi_r+1)/2),
                           alpha=0.9, rstride=2, cstride=2)
    ax2.set_xlabel('X', fontsize=11)
    ax2.set_ylabel('Y', fontsize=11)
    ax2.set_zlabel('Z', fontsize=11)
    ax2.set_title('3D View', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('kink_mode_structure.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("External Kink Instability Analysis")
    print("=" * 60)

    # Growth rate vs q(a)
    print("\nPlotting growth rate vs q(a)...")
    plot_growth_vs_q(m=1, k_z_values=[0.1, 0.5, 1.0])
    print("  Saved as 'kink_growth_vs_q.png'")

    # Wall stabilization
    print("\nPlotting wall stabilization effect...")
    plot_wall_stabilization(m=1)
    print("  Saved as 'kink_wall_stabilization.png'")

    # Mode structure
    print("\nVisualizing mode structure...")
    plot_mode_structure(m=1, n=1)
    print("  Saved as 'kink_mode_structure.png'")

    print("\nKey results:")
    print("  - m=1 kink unstable when q(a) < 1")
    print("  - Conducting wall provides stabilization")
    print("  - Stabilization stronger when wall closer to plasma")


if __name__ == '__main__':
    main()
