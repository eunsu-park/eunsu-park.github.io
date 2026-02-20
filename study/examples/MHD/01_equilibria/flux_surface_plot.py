#!/usr/bin/env python3
"""
Flux Surface Visualization

Visualizes nested flux surfaces from Grad-Shafranov solution, including:
- Nested flux surfaces (contours of ψ)
- Last Closed Flux Surface (LCFS)
- Magnetic axis (ψ maximum)
- X-point location (for divertor configurations)
- Safety factor q on each surface

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize_scalar


def solovev_flux_function(R, Z, R0=1.0, a=0.3, kappa=1.0, delta=0.0):
    """
    Compute Solovev analytical flux function.

    Parameters
    ----------
    R, Z : array_like
        Major radius and vertical position
    R0 : float
        Major radius of magnetic axis
    a : float
        Minor radius
    kappa : float
        Elongation
    delta : float
        Triangularity

    Returns
    -------
    psi : ndarray
        Poloidal flux
    """
    x = (R - R0) / a
    y = Z / (a * kappa)

    # Solovev solution parameters
    c1 = 1.0
    c2 = -1.0 / (2 * a**2)

    # Include triangularity effect
    c3 = delta / a**2

    psi = c1 * x**2 + c2 * y**2 + c3 * x * y**2

    return psi


def find_magnetic_axis(R, Z, psi):
    """
    Find magnetic axis (maximum of ψ).

    Parameters
    ----------
    R, Z : ndarray
        Grid coordinates
    psi : ndarray
        Flux function

    Returns
    -------
    R_axis, Z_axis : float
        Magnetic axis location
    psi_axis : float
        Flux at axis
    """
    idx = np.unravel_index(np.argmax(psi), psi.shape)
    R_axis = R[idx]
    Z_axis = Z[idx]
    psi_axis = psi[idx]

    return R_axis, Z_axis, psi_axis


def find_x_point(R, Z, psi):
    """
    Find X-point (saddle point of ψ).

    For a divertor configuration, the X-point is a saddle point
    where ∂ψ/∂R = ∂ψ/∂Z = 0 and the Hessian has opposite-sign eigenvalues.

    Parameters
    ----------
    R, Z : ndarray
        Grid coordinates
    psi : ndarray
        Flux function

    Returns
    -------
    R_x, Z_x : float or None
        X-point location (None if not found)
    psi_x : float or None
        Flux at X-point
    """
    # Compute gradients
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]

    dpsi_dR = np.gradient(psi, dR, axis=0)
    dpsi_dZ = np.gradient(psi, dZ, axis=1)

    # Find points where gradient is small
    grad_mag = np.sqrt(dpsi_dR**2 + dpsi_dZ**2)
    threshold = 0.1 * np.max(grad_mag)

    candidates = grad_mag < threshold

    # Look for saddle points (below magnetic axis)
    Z_mid = (Z.min() + Z.max()) / 2
    candidates = candidates & (Z < Z_mid)

    if np.any(candidates):
        # Take the point with lowest Z
        idx_candidates = np.where(candidates)
        min_z_idx = np.argmin(Z[idx_candidates])
        i, j = idx_candidates[0][min_z_idx], idx_candidates[1][min_z_idx]

        R_x = R[i, j]
        Z_x = Z[i, j]
        psi_x = psi[i, j]

        return R_x, Z_x, psi_x
    else:
        return None, None, None


def compute_safety_factor(R, Z, psi, R0, B0):
    """
    Compute safety factor q(ψ) for each flux surface.

    q = (1/2π) ∮ (B·∇φ)/(B·∇θ) dθ

    For circular surfaces: q ≈ r*B_φ/(R*B_θ)

    Parameters
    ----------
    R, Z : ndarray
        Grid coordinates
    psi : ndarray
        Flux function
    R0 : float
        Major radius
    B0 : float
        Toroidal field at R0

    Returns
    -------
    psi_values : ndarray
        Flux values
    q_values : ndarray
        Safety factor values
    """
    # Select flux surfaces
    psi_min, psi_max = psi.min(), psi.max()
    psi_levels = np.linspace(psi_min, psi_max * 0.95, 10)

    q_values = []

    for psi_val in psi_levels:
        # Find contour at this psi value
        # Approximate q using circular approximation
        # Find average radius of this flux surface
        mask = (psi >= psi_val * 0.99) & (psi <= psi_val * 1.01)
        if np.any(mask):
            R_surf = R[mask]
            r_minor = np.mean(np.abs(R_surf - R0))

            # Toroidal field: B_phi ~ B0 * R0 / R
            B_phi = B0 * R0 / R0

            # Poloidal field estimate from psi
            dpsi = psi_max - psi_min
            B_poloidal = dpsi / (2 * np.pi * r_minor * R0)  # Rough estimate

            q = r_minor * B_phi / (R0 * B_poloidal + 1e-10)
            q_values.append(q)
        else:
            q_values.append(np.nan)

    return psi_levels, np.array(q_values)


def plot_flux_surfaces(R, Z, psi, n_surfaces=15):
    """
    Plot nested flux surfaces with annotations.

    Parameters
    ----------
    R, Z : ndarray
        Grid coordinates
    psi : ndarray
        Flux function
    n_surfaces : int
        Number of flux surfaces to plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Find magnetic axis and X-point
    R_axis, Z_axis, psi_axis = find_magnetic_axis(R, Z, psi)
    R_x, Z_x, psi_x = find_x_point(R, Z, psi)

    # Plot 1: Flux surfaces with annotations
    ax = axes[0]

    # Flux surface levels
    psi_min, psi_max = psi.min(), psi.max()
    levels = np.linspace(psi_min, psi_max, n_surfaces)

    # Plot flux surfaces
    cs = ax.contour(R, Z, psi, levels=levels, colors='blue', linewidths=1.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.3f')

    # Mark LCFS (outermost closed surface)
    lcfs_level = psi_max * 0.95
    ax.contour(R, Z, psi, levels=[lcfs_level], colors='red',
               linewidths=3, linestyles='--')

    # Mark magnetic axis
    ax.plot(R_axis, Z_axis, 'r*', markersize=20, label='Magnetic Axis')
    ax.text(R_axis + 0.05, Z_axis + 0.05, f'Axis\n({R_axis:.2f}, {Z_axis:.2f})',
            fontsize=10, color='red', fontweight='bold')

    # Mark X-point if found
    if R_x is not None:
        ax.plot(R_x, Z_x, 'kx', markersize=15, markeredgewidth=3, label='X-point')
        ax.text(R_x + 0.05, Z_x - 0.05, f'X-point\n({R_x:.2f}, {Z_x:.2f})',
                fontsize=10, color='black', fontweight='bold')

    ax.set_xlabel('R (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_title('Flux Surfaces', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Safety factor q(ψ)
    ax2 = axes[1]

    R0 = R_axis
    B0 = 2.0  # Tesla
    psi_vals, q_vals = compute_safety_factor(R, Z, psi, R0, B0)

    # Normalize psi for plotting
    psi_norm = (psi_vals - psi_min) / (psi_max - psi_min)

    ax2.plot(psi_norm, q_vals, 'b-o', linewidth=2, markersize=6)
    ax2.axhline(y=1, color='r', linestyle='--', linewidth=1, alpha=0.5, label='q=1')
    ax2.axhline(y=2, color='g', linestyle='--', linewidth=1, alpha=0.5, label='q=2')
    ax2.axhline(y=3, color='m', linestyle='--', linewidth=1, alpha=0.5, label='q=3')

    ax2.set_xlabel('Normalized flux ψ_N', fontsize=12)
    ax2.set_ylabel('Safety factor q', fontsize=12)
    ax2.set_title('Safety Factor Profile', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    ax2.set_xlim([0, 1])

    plt.suptitle('Tokamak Flux Surface Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('flux_surfaces.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_surface_quantities(R, Z, psi):
    """
    Plot additional surface quantities.

    Parameters
    ----------
    R, Z : ndarray
        Grid coordinates
    psi : ndarray
        Flux function
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Compute gradients for field components
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]

    dpsi_dR = np.gradient(psi, dR, axis=0)
    dpsi_dZ = np.gradient(psi, dZ, axis=1)

    # Poloidal field components: B_R = -1/(R) * ∂ψ/∂Z, B_Z = 1/(R) * ∂ψ/∂R
    B_R = -dpsi_dZ / (R + 1e-10)
    B_Z = dpsi_dR / (R + 1e-10)
    B_poloidal = np.sqrt(B_R**2 + B_Z**2)

    # Plot B_R
    im1 = axes[0, 0].contourf(R, Z, B_R, levels=20, cmap='RdBu_r')
    axes[0, 0].contour(R, Z, psi, levels=10, colors='black',
                       linewidths=0.5, alpha=0.3)
    axes[0, 0].set_xlabel('R (m)', fontsize=11)
    axes[0, 0].set_ylabel('Z (m)', fontsize=11)
    axes[0, 0].set_title('B_R (radial field)', fontsize=12, fontweight='bold')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0], label='B_R (T)')

    # Plot B_Z
    im2 = axes[0, 1].contourf(R, Z, B_Z, levels=20, cmap='RdBu_r')
    axes[0, 1].contour(R, Z, psi, levels=10, colors='black',
                       linewidths=0.5, alpha=0.3)
    axes[0, 1].set_xlabel('R (m)', fontsize=11)
    axes[0, 1].set_ylabel('Z (m)', fontsize=11)
    axes[0, 1].set_title('B_Z (vertical field)', fontsize=12, fontweight='bold')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1], label='B_Z (T)')

    # Plot |B_poloidal|
    im3 = axes[1, 0].contourf(R, Z, B_poloidal, levels=20, cmap='viridis')
    axes[1, 0].contour(R, Z, psi, levels=10, colors='white',
                       linewidths=0.5, alpha=0.5)
    axes[1, 0].set_xlabel('R (m)', fontsize=11)
    axes[1, 0].set_ylabel('Z (m)', fontsize=11)
    axes[1, 0].set_title('|B_poloidal|', fontsize=12, fontweight='bold')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(im3, ax=axes[1, 0], label='|B_p| (T)')

    # Plot flux surface spacing (proxy for confinement quality)
    dpsi_norm = np.gradient(psi, axis=0)**2 + np.gradient(psi, axis=1)**2
    dpsi_norm = np.sqrt(dpsi_norm)

    im4 = axes[1, 1].contourf(R, Z, dpsi_norm, levels=20, cmap='hot')
    axes[1, 1].contour(R, Z, psi, levels=10, colors='blue',
                       linewidths=0.5, alpha=0.3)
    axes[1, 1].set_xlabel('R (m)', fontsize=11)
    axes[1, 1].set_ylabel('Z (m)', fontsize=11)
    axes[1, 1].set_title('|∇ψ| (flux surface spacing)', fontsize=12, fontweight='bold')
    axes[1, 1].set_aspect('equal')
    plt.colorbar(im4, ax=axes[1, 1], label='|∇ψ|')

    plt.suptitle('Poloidal Field Components',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('flux_surface_fields.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    # Create grid
    R = np.linspace(0.5, 1.5, 150)
    Z = np.linspace(-0.6, 0.6, 150)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing='ij')

    # Generate Solovev solution
    R0 = 1.0  # Major radius
    a = 0.3   # Minor radius
    kappa = 1.5  # Elongation
    delta = 0.3  # Triangularity

    print("Flux Surface Visualization")
    print("=" * 50)
    print(f"Configuration parameters:")
    print(f"  Major radius R0: {R0:.2f} m")
    print(f"  Minor radius a: {a:.2f} m")
    print(f"  Elongation κ: {kappa:.2f}")
    print(f"  Triangularity δ: {delta:.2f}")
    print()

    psi = solovev_flux_function(R_grid, Z_grid, R0, a, kappa, delta)

    # Find key locations
    R_axis, Z_axis, psi_axis = find_magnetic_axis(R_grid, Z_grid, psi)
    R_x, Z_x, psi_x = find_x_point(R_grid, Z_grid, psi)

    print(f"Magnetic axis: R = {R_axis:.3f} m, Z = {Z_axis:.3f} m")
    print(f"  ψ_axis = {psi_axis:.3f}")

    if R_x is not None:
        print(f"X-point: R = {R_x:.3f} m, Z = {Z_x:.3f} m")
        print(f"  ψ_x = {psi_x:.3f}")
    else:
        print("No X-point found (limiter configuration)")

    print()

    # Plot flux surfaces
    plot_flux_surfaces(R_grid, Z_grid, psi, n_surfaces=15)
    print("Flux surface plot saved as 'flux_surfaces.png'")

    # Plot field components
    plot_surface_quantities(R_grid, Z_grid, psi)
    print("Field components plot saved as 'flux_surface_fields.png'")


if __name__ == '__main__':
    main()
