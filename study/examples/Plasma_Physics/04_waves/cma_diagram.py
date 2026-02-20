#!/usr/bin/env python3
"""
Clemmow-Mullaly-Allis (CMA) Diagram

This script generates the CMA diagram showing wave propagation regions
in a cold magnetized plasma for parallel propagation (k || B).

Key Features:
- Axes: X = ωpe²/ω² (horizontal), Y = ωce/ω (vertical)
- Cutoff curves: R=0, L=0, P=0
- Resonance curves: S=0 (upper/lower hybrid)
- 13 distinct propagation regions
- Color-coded by number of propagating modes

Reference: Stix, "Waves in Plasmas" (1992)

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def compute_stix_parameters(X, Y, Z=0):
    """
    Compute Stix parameters for cold plasma dispersion.

    Parameters:
    -----------
    X : float or array
        X = ωpe²/ω²
    Y : float or array
        Y = ωce/ω (electron cyclotron)
    Z : float or array
        Z = ωci/ω (ion cyclotron, usually neglected)

    Returns:
    --------
    S, D, P : Stix parameters
    """
    # Stix parameters
    S = 1 - X / (1 - Y**2)
    D = -X * Y / (1 - Y**2)
    P = 1 - X

    return S, D, P

def R_cutoff(Y):
    """R-mode cutoff: R = S + D = 0."""
    # R = 1 - X/(1-Y) = 0
    # X = 1 - Y
    return 1 - Y

def L_cutoff(Y):
    """L-mode cutoff: L = S - D = 0."""
    # L = 1 - X/(1+Y) = 0
    # X = 1 + Y
    return 1 + Y

def P_cutoff(Y):
    """P-mode (plasma) cutoff: P = 0."""
    # P = 1 - X = 0
    # X = 1
    return np.ones_like(Y)

def upper_hybrid_resonance(Y):
    """Upper hybrid resonance: S = 0, ω² = ωpe² + ωce²."""
    # S = 1 - X/(1-Y²) = 0
    # X = 1 - Y²
    return 1 - Y**2

def lower_hybrid_resonance(Y):
    """
    Lower hybrid resonance (approximate).

    Exact formula involves ion cyclotron frequency.
    Here we use simplified version.
    """
    # For the CMA diagram, this is the ion resonance line
    # In practice, this would be Y → -Ωi/ω ≈ 0 for small ion effects
    # We'll mark it as a separate region
    return 0 * Y  # Placeholder

def plot_cma_diagram():
    """
    Create the Clemmow-Mullaly-Allis (CMA) diagram.
    """
    # Create grid
    Y_array = np.linspace(-3, 3, 1000)

    # Compute cutoff and resonance curves
    X_R = R_cutoff(Y_array)
    X_L = L_cutoff(Y_array)
    X_P = P_cutoff(Y_array)
    X_UH = upper_hybrid_resonance(Y_array)

    # Clip negative values
    X_R = np.maximum(X_R, 0)
    X_L = np.maximum(X_L, 0)
    X_UH = np.maximum(X_UH, 0)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot cutoff curves
    ax.plot(X_R, Y_array, 'r-', linewidth=2.5, label='R-cutoff (R=0)')
    ax.plot(X_L, Y_array, 'b-', linewidth=2.5, label='L-cutoff (L=0)')
    ax.plot(X_P, Y_array, 'g-', linewidth=2.5, label='P-cutoff (P=0)')

    # Plot resonance curves
    ax.plot(X_UH, Y_array, 'm--', linewidth=2.5, label='Upper Hybrid (S=0)')

    # Add important lines
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
    ax.axhline(y=1, color='orange', linewidth=1, linestyle=':', alpha=0.5)
    ax.axhline(y=-1, color='orange', linewidth=1, linestyle=':', alpha=0.5)

    # Define propagation regions and color them
    regions = [
        # Format: [X_range, Y_range, num_modes, label, position]
        ([0, 0.5], [1.5, 2.5], 2, 'R+L', (0.25, 2.0)),
        ([0, 0.5], [0.5, 1.5], 3, 'R+L+P', (0.25, 1.0)),
        ([0.5, 1.0], [0.5, 1.5], 2, 'L+P', (0.75, 1.0)),
        ([0.5, 1.0], [1.5, 2.5], 1, 'L', (0.75, 2.0)),
        ([1.0, 2.0], [0.5, 1.5], 1, 'L', (1.5, 1.0)),
        ([1.0, 2.0], [1.5, 2.5], 0, 'None', (1.5, 2.0)),
        ([0, 0.5], [-1.5, 0.5], 3, 'R+L+P', (0.25, -0.5)),
        ([0.5, 1.0], [-1.5, 0.5], 2, 'R+P', (0.75, -0.5)),
        ([1.0, 2.0], [-1.5, 0.5], 1, 'R', (1.5, -0.5)),
        ([0, 0.5], [-2.5, -1.5], 2, 'R+L', (0.25, -2.0)),
        ([0.5, 1.0], [-2.5, -1.5], 1, 'R', (0.75, -2.0)),
        ([1.0, 2.0], [-2.5, -1.5], 0, 'None', (1.5, -2.0)),
    ]

    # Color map based on number of modes
    colors = {0: '#FFCCCC', 1: '#FFFFCC', 2: '#CCFFCC', 3: '#CCFFFF'}

    for X_range, Y_range, num_modes, label, pos in regions:
        # Create rectangle
        rect = plt.Rectangle((X_range[0], Y_range[0]),
                             X_range[1] - X_range[0],
                             Y_range[1] - Y_range[0],
                             facecolor=colors[num_modes],
                             edgecolor='gray',
                             alpha=0.3,
                             linewidth=0.5)
        ax.add_patch(rect)

        # Add label
        ax.text(pos[0], pos[1], f'{label}\n({num_modes} modes)',
               ha='center', va='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Mark special points
    # Point A: R-L crossover at Y=0
    ax.plot(1, 0, 'ko', markersize=8)
    ax.text(1.05, 0.1, 'A: ω=ωpe', fontsize=9)

    # Point B: Upper hybrid at Y=0
    ax.plot(1, 0, 'mo', markersize=8)

    # Point C: Upper hybrid resonance example
    Y_example = 0.5
    X_example = 1 - Y_example**2
    ax.plot(X_example, Y_example, 'mo', markersize=8)
    ax.text(X_example + 0.05, Y_example + 0.1,
           f'C: UH\nω²=ωpe²+ωce²', fontsize=9)

    # Add annotations for important regimes
    # Ionosphere
    ax.annotate('Ionosphere\n(X>>1, Y<<1)',
               xy=(3, 0.3), fontsize=10,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Magnetosphere
    ax.annotate('Magnetosphere\n(X<<1, Y~1)',
               xy=(0.3, 1.5), fontsize=10,
               bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.5))

    # Tokamak core
    ax.annotate('Tokamak core\n(X~1, Y~0.1)',
               xy=(1.3, 0.15), fontsize=10,
               bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))

    # Labels and formatting
    ax.set_xlabel(r'$X = \omega_{pe}^2 / \omega^2$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$Y = \omega_{ce} / \omega$', fontsize=14, fontweight='bold')
    ax.set_title('Clemmow-Mullaly-Allis (CMA) Diagram\nParallel Propagation (k || B)',
                fontsize=16, fontweight='bold')

    ax.set_xlim([0, 3.5])
    ax.set_ylim([-3, 3])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # Add text box with explanation
    textstr = '\n'.join([
        'Cutoffs: n² → 0',
        'Resonances: n² → ∞',
        '',
        'Modes:',
        'R: Right-hand circular',
        'L: Left-hand circular',
        'P: Plasma (O-mode)',
    ])
    ax.text(2.5, -2.3, textstr, fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('cma_diagram.png', dpi=200, bbox_inches='tight')
    print("CMA diagram saved as 'cma_diagram.png'")
    print("\n" + "="*70)
    print("CMA Diagram Information")
    print("="*70)
    print("Axes:")
    print("  X = ωpe²/ω² (plasma parameter)")
    print("  Y = ωce/ω (magnetic parameter)")
    print("\nCutoff curves (n² = 0, wave reflects):")
    print("  R-cutoff: X = 1 - Y")
    print("  L-cutoff: X = 1 + Y")
    print("  P-cutoff: X = 1")
    print("\nResonance curves (n² → ∞, wave absorbed):")
    print("  Upper Hybrid: X = 1 - Y²")
    print("\nPropagating modes:")
    print("  R: Right-hand polarized (electron cyclotron)")
    print("  L: Left-hand polarized (whistler, helicon)")
    print("  P: Ordinary mode (electric field || B)")
    print("="*70)

    plt.show()

    # Create a second figure showing specific wave types
    plot_wave_trajectories()

def plot_wave_trajectories():
    """
    Plot example wave trajectories through CMA diagram.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Background
    Y_array = np.linspace(-2, 2, 1000)
    X_R = np.maximum(R_cutoff(Y_array), 0)
    X_L = np.maximum(L_cutoff(Y_array), 0)
    X_P = P_cutoff(Y_array)
    X_UH = np.maximum(upper_hybrid_resonance(Y_array), 0)

    ax.plot(X_R, Y_array, 'r-', linewidth=2, alpha=0.5, label='R-cutoff')
    ax.plot(X_L, Y_array, 'b-', linewidth=2, alpha=0.5, label='L-cutoff')
    ax.plot(X_P, Y_array, 'g-', linewidth=2, alpha=0.5, label='P-cutoff')
    ax.plot(X_UH, Y_array, 'm--', linewidth=2, alpha=0.5, label='UH resonance')

    # Example trajectories for fixed frequency, varying density
    # Path 1: Low frequency wave entering plasma (Y = 2)
    Y1 = 2.0
    X_path1 = np.linspace(0, 3, 100)
    Y_path1 = Y1 * np.ones_like(X_path1)

    ax.plot(X_path1, Y_path1, 'purple', linewidth=3, label='Low freq. (Y=2)')
    ax.annotate('', xy=(3, Y1), xytext=(0, Y1),
               arrowprops=dict(arrowstyle='->', lw=2, color='purple'))

    # Mark cutoff
    X_cutoff1 = R_cutoff(Y1)
    ax.plot(X_cutoff1, Y1, 'ro', markersize=10)
    ax.text(X_cutoff1, Y1 + 0.15, 'R-cutoff', fontsize=10)

    # Path 2: Intermediate frequency (Y = 0.5)
    Y2 = 0.5
    X_path2 = np.linspace(0, 3, 100)
    Y_path2 = Y2 * np.ones_like(X_path2)

    ax.plot(X_path2, Y_path2, 'orange', linewidth=3, label='Mid freq. (Y=0.5)')
    ax.annotate('', xy=(3, Y2), xytext=(0, Y2),
               arrowprops=dict(arrowstyle='->', lw=2, color='orange'))

    # Mark UH resonance
    X_UH2 = upper_hybrid_resonance(Y2)
    ax.plot(X_UH2, Y2, 'mo', markersize=10)
    ax.text(X_UH2, Y2 + 0.15, 'UH', fontsize=10)

    # Path 3: Unmagnetized (Y = 0)
    Y3 = 0.0
    X_path3 = np.linspace(0, 3, 100)
    Y_path3 = Y3 * np.ones_like(X_path3)

    ax.plot(X_path3, Y_path3, 'green', linewidth=3, label='Unmagnetized (Y=0)')
    ax.annotate('', xy=(3, Y3), xytext=(0, Y3),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))

    # Mark plasma cutoff
    ax.plot(1, Y3, 'go', markersize=10)
    ax.text(1, Y3 - 0.15, 'Plasma cutoff', fontsize=10, va='top')

    ax.set_xlabel(r'$X = \omega_{pe}^2 / \omega^2$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$Y = \omega_{ce} / \omega$', fontsize=14, fontweight='bold')
    ax.set_title('Wave Trajectories in CMA Space\n(Increasing density at fixed frequency)',
                fontsize=14, fontweight='bold')
    ax.set_xlim([0, 3.5])
    ax.set_ylim([-2, 2.5])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=0, color='black', linewidth=1)

    plt.tight_layout()
    plt.savefig('cma_wave_paths.png', dpi=150, bbox_inches='tight')
    print("Wave trajectory plot saved as 'cma_wave_paths.png'")

    plt.show()

if __name__ == "__main__":
    plot_cma_diagram()
