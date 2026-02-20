#!/usr/bin/env python3
"""
Ballooning Mode Instability

Solves the ballooning equation in s-α coordinates to find stability boundaries.

s = r q'/q (magnetic shear)
α = -2μ₀ Rq² p'/B² (normalized pressure gradient)

Identifies first and second stability regions in the s-α diagram.

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def ballooning_equation(theta, y, s, alpha):
    """
    Ballooning equation in ballooning representation.

    d²X/dθ² + [s²θ² + α sin(θ)]X = 0

    Parameters
    ----------
    theta : float
        Poloidal angle
    y : array
        [X, dX/dtheta]
    s : float
        Magnetic shear
    alpha : float
        Pressure gradient parameter

    Returns
    -------
    dydt theta : array
        [dX/dθ, d²X/dθ²]
    """
    X, dX = y

    # Ballooning equation coefficient
    coeff = s**2 * theta**2 + alpha * np.sin(theta)

    d2X = -coeff * X

    return [dX, d2X]


def solve_ballooning(s, alpha, theta_max=20):
    """
    Solve ballooning equation and determine stability.

    Parameters
    ----------
    s : float
        Magnetic shear
    alpha : float
        Pressure gradient
    theta_max : float
        Integration range

    Returns
    -------
    stable : bool
        True if stable
    growth_indicator : float
        Positive if unstable
    """
    # Initial conditions: X(0) = 1, dX/dθ(0) = 0
    y0 = [1.0, 0.0]

    # Integrate
    theta_span = (0, theta_max)
    theta_eval = np.linspace(0, theta_max, 500)

    try:
        sol = solve_ivp(ballooning_equation, theta_span, y0,
                       args=(s, alpha), t_eval=theta_eval,
                       method='RK45', max_step=0.1)

        X = sol.y[0]

        # Check for exponential growth
        # If |X| grows exponentially, mode is unstable
        growth_indicator = np.log(np.abs(X[-1]) / np.abs(X[0]) + 1e-10)

        # Threshold for instability
        stable = growth_indicator < 1.0

        return stable, growth_indicator

    except:
        # Integration failed, likely unstable
        return False, 10.0


def create_s_alpha_diagram(s_range, alpha_range, resolution=50):
    """
    Create stability diagram in s-α space.

    Parameters
    ----------
    s_range : tuple
        (s_min, s_max)
    alpha_range : tuple
        (alpha_min, alpha_max)
    resolution : int
        Grid resolution

    Returns
    -------
    S, Alpha : ndarray
        Meshgrids
    stability_map : ndarray
        Stability map (1 = stable, 0 = unstable)
    """
    s_arr = np.linspace(s_range[0], s_range[1], resolution)
    alpha_arr = np.linspace(alpha_range[0], alpha_range[1], resolution)

    S, Alpha = np.meshgrid(s_arr, alpha_arr)
    stability_map = np.zeros_like(S)

    print("Computing s-α stability diagram...")
    for i in range(len(alpha_arr)):
        if i % 10 == 0:
            print(f"  Progress: {100*i/len(alpha_arr):.0f}%")

        for j in range(len(s_arr)):
            stable, _ = solve_ballooning(s_arr[j], alpha_arr[i])
            stability_map[i, j] = 1.0 if stable else 0.0

    print("  Complete!")
    return S, Alpha, stability_map


def plot_s_alpha_diagram(S, Alpha, stability_map):
    """
    Plot s-α stability diagram.

    Parameters
    ----------
    S, Alpha : ndarray
        Coordinate grids
    stability_map : ndarray
        Stability (1 = stable, 0 = unstable)
    """
    fig, ax = plt.subplots(figsize=(11, 8))

    # Plot stability regions
    im = ax.contourf(S, Alpha, stability_map, levels=[0, 0.5, 1.0],
                     colors=['red', 'lightgreen'], alpha=0.7)

    # Stability boundary
    cs = ax.contour(S, Alpha, stability_map, levels=[0.5],
                   colors='black', linewidths=3)

    ax.set_xlabel('Magnetic shear s = r q\'/q', fontsize=13)
    ax.set_ylabel('Pressure parameter α = -2μ₀Rq²p\'/B²', fontsize=13)
    ax.set_title('Ballooning Stability Diagram',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Labels for regions
    ax.text(0.5, 0.3, 'FIRST STABLE\nREGION',
           fontsize=14, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Check for second stability region at high s
    if np.any((S > 2) & (stability_map > 0.5)):
        ax.text(3.5, 2.0, 'SECOND STABLE\nREGION',
               fontsize=14, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.text(1.0, 1.5, 'UNSTABLE',
           fontsize=14, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('ballooning_s_alpha_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_eigenfunction_examples(s_values, alpha_test):
    """
    Plot eigenfunctions for different shear values.

    Parameters
    ----------
    s_values : list
        Shear values to plot
    alpha_test : float
        Pressure gradient parameter
    """
    theta_max = 15
    theta = np.linspace(0, theta_max, 500)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, s in enumerate(s_values[:4]):
        ax = axes[i]

        y0 = [1.0, 0.0]
        sol = solve_ivp(ballooning_equation, (0, theta_max), y0,
                       args=(s, alpha_test), t_eval=theta,
                       method='RK45', max_step=0.1)

        X = sol.y[0]
        stable, growth = solve_ballooning(s, alpha_test)

        ax.plot(theta, X, 'b-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Poloidal angle θ', fontsize=11)
        ax.set_ylabel('Eigenfunction X(θ)', fontsize=11)

        status = "STABLE" if stable else "UNSTABLE"
        color = "green" if stable else "red"

        ax.set_title(f's={s:.2f}, α={alpha_test:.2f}\n{status}',
                    fontsize=12, fontweight='bold', color=color)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Ballooning Mode Eigenfunctions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ballooning_eigenfunctions.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_stability_boundaries():
    """
    Plot stability boundaries for different alpha values.
    """
    s_arr = np.linspace(0, 5, 100)
    alpha_values = [0.5, 1.0, 1.5, 2.0, 2.5]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_values)))

    for i, alpha in enumerate(alpha_values):
        growth_arr = []

        for s in s_arr:
            stable, growth = solve_ballooning(s, alpha)
            growth_arr.append(growth)

        ax.plot(s_arr, growth_arr, color=colors[i], linewidth=2.5,
               label=f'α = {alpha:.1f}')

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
              label='Stability threshold')
    ax.set_xlabel('Magnetic shear s', fontsize=13)
    ax.set_ylabel('Growth indicator', fontsize=13)
    ax.set_title('Ballooning Stability vs Magnetic Shear',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, 5])

    # Shade stable region
    ax.fill_between(s_arr, -0.5, 1.0, alpha=0.2, color='green',
                    label='Stable region')

    plt.tight_layout()
    plt.savefig('ballooning_vs_shear.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("Ballooning Mode Stability Analysis")
    print("=" * 60)

    # Create s-α diagram
    print("\nCreating stability diagram in s-α space...")
    s_range = (0, 5)
    alpha_range = (0, 3)
    S, Alpha, stability_map = create_s_alpha_diagram(s_range, alpha_range,
                                                     resolution=40)

    # Find stability boundaries
    n_stable = np.sum(stability_map > 0.5)
    n_total = stability_map.size
    print(f"\nStable points: {n_stable}/{n_total} ({100*n_stable/n_total:.1f}%)")

    # Plot s-α diagram
    plot_s_alpha_diagram(S, Alpha, stability_map)
    print("s-α diagram saved as 'ballooning_s_alpha_diagram.png'")

    # Plot eigenfunctions
    print("\nComputing example eigenfunctions...")
    s_test = [0.5, 1.0, 2.0, 3.0]
    alpha_test = 1.0
    plot_eigenfunction_examples(s_test, alpha_test)
    print("Eigenfunctions saved as 'ballooning_eigenfunctions.png'")

    # Plot stability vs shear
    print("\nPlotting stability vs magnetic shear...")
    plot_stability_boundaries()
    print("Shear dependence saved as 'ballooning_vs_shear.png'")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
