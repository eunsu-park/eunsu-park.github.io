#!/usr/bin/env python3
"""
Sweet-Parker Reconnection Model

Analyzes steady-state Sweet-Parker current sheet reconnection.

Key results:
- Reconnection rate ~ S^(-1/2) where S is Lundquist number
- Too slow for solar flares (S ~ 10^12)
- Comparison with Petschek model

Author: Claude
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt


def sweet_parker_rate(S):
    """
    Sweet-Parker reconnection rate.

    Rate = 1 / sqrt(S)

    where S = L V_A / η is the Lundquist number.

    Parameters
    ----------
    S : float or array
        Lundquist number

    Returns
    -------
    rate : float or array
        Reconnection rate (normalized)
    """
    return 1.0 / np.sqrt(S)


def petschek_rate(S):
    """
    Petschek reconnection rate.

    Rate = π / (8 ln(S))

    Parameters
    ----------
    S : float or array
        Lundquist number

    Returns
    -------
    rate : float or array
        Reconnection rate
    """
    return np.pi / (8 * np.log(S + 1))  # +1 to avoid log(0)


def sheet_parameters(S, L, B0, n, T):
    """
    Compute Sweet-Parker current sheet parameters.

    Parameters
    ----------
    S : float
        Lundquist number
    L : float
        System size (m)
    B0 : float
        Magnetic field (T)
    n : float
        Density (m⁻³)
    T : float
        Temperature (eV)

    Returns
    -------
    dict
        Sheet parameters
    """
    mu_0 = 4 * np.pi * 1e-7
    m_i = 1.67e-27  # Proton mass

    # Alfven speed
    V_A = B0 / np.sqrt(mu_0 * n * m_i)

    # Reconnection rate
    rate = sweet_parker_rate(S)

    # Inflow velocity
    V_in = rate * V_A

    # Sheet length
    sheet_length = L

    # Sheet width
    delta = L / np.sqrt(S)

    # Outflow velocity
    V_out = V_A

    # Current density (approximate)
    J = B0 / (mu_0 * delta)

    # Resistive diffusion time
    tau_resist = L**2 / (V_A * L / S)  # = S * tau_A

    # Reconnection time
    tau_reconnect = L / V_in

    return {
        'V_A': V_A,
        'rate': rate,
        'V_in': V_in,
        'V_out': V_out,
        'delta': delta,
        'L': sheet_length,
        'J': J,
        'tau_resist': tau_resist,
        'tau_reconnect': tau_reconnect
    }


def plot_rate_vs_lundquist():
    """
    Plot reconnection rate vs Lundquist number.
    """
    S = np.logspace(2, 14, 100)  # 10² to 10¹⁴

    rate_SP = sweet_parker_rate(S)
    rate_Petschek = petschek_rate(S)

    # Hall MHD rate (approximately constant)
    rate_Hall = 0.1 * np.ones_like(S)

    # Plasmoid-mediated rate
    rate_plasmoid = 0.01 * np.ones_like(S)

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.loglog(S, rate_SP, 'b-', linewidth=3, label='Sweet-Parker ~ S^(-1/2)')
    ax.loglog(S, rate_Petschek, 'r-', linewidth=3,
             label='Petschek ~ π/(8 ln S)')
    ax.axhline(rate_Hall[0], color='g', linestyle='--', linewidth=2.5,
              label='Hall MHD ~ 0.1')
    ax.axhline(rate_plasmoid[0], color='m', linestyle='--', linewidth=2.5,
              label='Plasmoid-mediated ~ 0.01')

    # Mark solar flare regime
    ax.axvline(1e12, color='orange', linestyle=':', linewidth=2,
              alpha=0.7, label='Solar flares (S~10¹²)')

    # Mark observed rates
    ax.fill_between([1e2, 1e14], 0.01, 0.1, alpha=0.2, color='yellow',
                    label='Observed rates')

    ax.set_xlabel('Lundquist number S', fontsize=13)
    ax.set_ylabel('Reconnection rate (V_in / V_A)', fontsize=13)
    ax.set_title('Magnetic Reconnection Rate vs Lundquist Number',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e-7, 1])

    plt.tight_layout()
    plt.savefig('reconnection_rate_vs_S.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_sheet_geometry():
    """
    Visualize Sweet-Parker current sheet geometry.
    """
    # Sheet dimensions (normalized)
    L = 1.0
    S = 1e4
    delta = L / np.sqrt(S)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw sheet
    sheet_x = [-L/2, L/2, L/2, -L/2, -L/2]
    sheet_y = [-delta/2, -delta/2, delta/2, delta/2, -delta/2]
    ax.fill(sheet_x, sheet_y, color='orange', alpha=0.5, label='Current sheet')
    ax.plot(sheet_x, sheet_y, 'k-', linewidth=2)

    # Draw magnetic field lines
    y = np.linspace(-3*delta, 3*delta, 20)
    for yi in y:
        if np.abs(yi) > delta/2:
            # Field lines outside sheet
            x_left = np.linspace(-L, -L/2, 50)
            x_right = np.linspace(L/2, L, 50)

            # Approximate field lines
            ax.plot(x_left, yi * np.ones_like(x_left), 'b-', linewidth=1, alpha=0.7)
            ax.plot(x_right, yi * np.ones_like(x_right), 'b-', linewidth=1, alpha=0.7)

    # Arrows for flow
    # Inflow
    ax.arrow(0, 2*delta, 0, -0.5*delta, head_width=0.02, head_length=0.1*delta,
            fc='red', ec='red', linewidth=2)
    ax.text(0.05, 2*delta, 'Inflow\n$V_{in}$', fontsize=11, color='red',
           fontweight='bold')

    # Outflow
    ax.arrow(L/2, 0, 0.2*L, 0, head_width=0.1*delta, head_length=0.05*L,
            fc='green', ec='green', linewidth=2)
    ax.text(0.7*L, 0.3*delta, 'Outflow\n$V_{out} \\sim V_A$',
           fontsize=11, color='green', fontweight='bold')

    # Annotations
    ax.annotate('', xy=(L/2, -delta), xytext=(-L/2, -delta),
               arrowprops=dict(arrowstyle='<->', lw=2))
    ax.text(0, -1.5*delta, f'L = {L:.1f}', fontsize=12, ha='center')

    ax.annotate('', xy=(L/2+0.1, delta/2), xytext=(L/2+0.1, -delta/2),
               arrowprops=dict(arrowstyle='<->', lw=2))
    ax.text(L/2+0.2, 0, f'δ = L/√S\\n={delta:.3f}',
           fontsize=11, ha='left')

    ax.set_xlabel('x / L', fontsize=13)
    ax.set_ylabel('y / L', fontsize=13)
    ax.set_title(f'Sweet-Parker Current Sheet (S={S:.0e})',
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('sweet_parker_geometry.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_scaling_comparison():
    """
    Compare different reconnection timescales.
    """
    S_arr = np.logspace(2, 14, 100)

    # Alfven time (reference)
    tau_A = 1.0  # Normalized

    # Resistive time
    tau_resist = S_arr * tau_A

    # Sweet-Parker reconnection time
    tau_SP = np.sqrt(S_arr) * tau_A

    # Petschek reconnection time
    tau_Petschek = (8 * np.log(S_arr) / np.pi) * tau_A

    # Fast (observed) reconnection
    tau_fast = 10 * tau_A * np.ones_like(S_arr)

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.loglog(S_arr, tau_resist / tau_A, 'k--', linewidth=2.5,
             label='Resistive ~ S τ_A')
    ax.loglog(S_arr, tau_SP / tau_A, 'b-', linewidth=3,
             label='Sweet-Parker ~ √S τ_A')
    ax.loglog(S_arr, tau_Petschek / tau_A, 'r-', linewidth=3,
             label='Petschek ~ ln(S) τ_A')
    ax.axhline(10, color='g', linestyle='--', linewidth=2.5,
              label='Fast reconnection ~ 10 τ_A')

    # Solar flare constraint
    ax.axvline(1e12, color='orange', linestyle=':', linewidth=2,
              alpha=0.7)
    ax.fill_between([1e12, 1e14], 1, 100, alpha=0.2, color='yellow',
                    label='Solar flare requirement')

    ax.set_xlabel('Lundquist number S', fontsize=13)
    ax.set_ylabel('Reconnection time / τ_A', fontsize=13)
    ax.set_title('Reconnection Timescale Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('reconnection_timescales.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("Sweet-Parker Reconnection Model")
    print("=" * 60)

    # Example parameters (solar corona)
    S = 1e12  # Lundquist number
    L = 1e7  # 10,000 km
    B0 = 0.01  # 100 G
    n = 1e15  # m⁻³
    T = 100  # eV

    params = sheet_parameters(S, L, B0, n, T)

    print(f"\nSolar corona parameters:")
    print(f"  Lundquist number S: {S:.1e}")
    print(f"  System size L: {L/1e3:.0f} km")
    print(f"  Magnetic field B₀: {B0*1e4:.0f} G")
    print(f"  Density n: {n:.1e} m⁻³")
    print()

    print(f"Sweet-Parker sheet:")
    print(f"  Alfven speed: {params['V_A']/1e3:.0f} km/s")
    print(f"  Reconnection rate: {params['rate']:.2e}")
    print(f"  Inflow velocity: {params['V_in']/1e3:.1f} km/s")
    print(f"  Sheet width δ: {params['delta']/1e3:.1f} km")
    print(f"  Reconnection time: {params['tau_reconnect']:.1e} s ({params['tau_reconnect']/60:.1f} min)")
    print()

    print("Problem: Solar flares occur in ~100 s, but Sweet-Parker predicts ~weeks!")
    print()

    # Plots
    print("Generating plots...")
    plot_rate_vs_lundquist()
    print("  Saved 'reconnection_rate_vs_S.png'")

    plot_sheet_geometry()
    print("  Saved 'sweet_parker_geometry.png'")

    plot_scaling_comparison()
    print("  Saved 'reconnection_timescales.png'")


if __name__ == '__main__':
    main()
