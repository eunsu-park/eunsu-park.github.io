#!/usr/bin/env python3
"""
Generalized Ohm's Law: Term-by-Term Comparison

This script compares the relative magnitude of different terms in
generalized Ohm's law for various plasma regimes.

Generalized Ohm's law:
E + v×B = ηJ + (J×B)/(ne) - ∇pe/(ne) + (me/ne²)(dJ/dt)

Key Physics:
- Ideal MHD: E + v×B = 0 (only this term)
- Resistive MHD: + ηJ
- Hall MHD: + J×B/(ne)
- Electron pressure: - ∇pe/(ne)
- Electron inertia: + (me/ne²)dJ/dt

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Physical constants
QE = 1.602176634e-19    # C
ME = 9.10938356e-31     # kg
MP = 1.672621898e-27    # kg
EPS0 = 8.854187817e-12  # F/m
MU0 = 4 * np.pi * 1e-7  # H/m

def compute_plasma_parameters(ne, B0, Te, L, v):
    """
    Compute characteristic parameters for a plasma.

    Parameters:
    -----------
    ne : float
        Electron density [m^-3]
    B0 : float
        Magnetic field [T]
    Te : float
        Electron temperature [eV]
    L : float
        Characteristic length scale [m]
    v : float
        Characteristic velocity [m/s]

    Returns:
    --------
    dict : Plasma parameters
    """
    mi = MP

    # Characteristic current density (from force balance)
    J = B0 / (MU0 * L)

    # Ion skin depth
    di = np.sqrt(ME * MU0 / (QE * ne)) / np.sqrt(MP / ME)  # Corrected
    di = np.sqrt(MP / (MU0 * ne * QE**2))

    # Resistivity (Spitzer)
    Te_joule = Te * QE
    ln_Lambda = 15  # Coulomb logarithm
    eta = 5.2e-5 * ln_Lambda / (Te**1.5)  # Ω·m

    # Electron thermal velocity
    vth_e = np.sqrt(2 * Te_joule / ME)

    # Electron plasma frequency
    omega_pe = np.sqrt(ne * QE**2 / (ME * EPS0))

    # Characteristic time
    tau = L / v

    return {
        'ne': ne,
        'B0': B0,
        'Te': Te,
        'L': L,
        'v': v,
        'J': J,
        'di': di,
        'eta': eta,
        'vth_e': vth_e,
        'omega_pe': omega_pe,
        'tau': tau
    }

def ideal_mhd_term(params):
    """E + v×B term magnitude."""
    return params['v'] * params['B0']

def resistive_term(params):
    """η·J term magnitude."""
    return params['eta'] * params['J']

def hall_term(params):
    """(J×B)/(ne) term magnitude."""
    return params['J'] * params['B0'] / (params['ne'] * QE)

def pressure_term(params):
    """∇pe/(ne) term magnitude."""
    Te_joule = params['Te'] * QE
    # ∇p ~ p/L = nkT/L
    return Te_joule / params['L']

def inertia_term(params):
    """(me/ne²)(dJ/dt) term magnitude."""
    # dJ/dt ~ J/τ
    dJ_dt = params['J'] / params['tau']
    return (ME / (params['ne'] * QE)) * dJ_dt

def plot_ohms_law_comparison():
    """
    Compare Ohm's law terms for different plasma regimes.
    """
    print("=" * 70)
    print("Generalized Ohm's Law: Term Comparison")
    print("=" * 70)

    # Create figure
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Define plasma regimes
    regimes = {
        'Tokamak Core': {
            'ne': 1e20,      # m^-3
            'B0': 5.0,       # T
            'Te': 10000,     # eV (10 keV)
            'L': 1.0,        # m
            'v': 1e5,        # m/s
            'color': 'red'
        },
        'Solar Wind': {
            'ne': 1e7,       # m^-3
            'B0': 5e-9,      # T (5 nT)
            'Te': 10,        # eV
            'L': 1e6,        # m (1000 km)
            'v': 4e5,        # m/s (400 km/s)
            'color': 'orange'
        },
        'Magnetopause': {
            'ne': 1e7,       # m^-3
            'B0': 50e-9,     # T (50 nT)
            'Te': 100,       # eV
            'L': 1e5,        # m (100 km)
            'v': 1e5,        # m/s
            'color': 'blue'
        },
        'Ionosphere': {
            'ne': 1e12,      # m^-3
            'B0': 50e-6,     # T (50 μT)
            'Te': 0.1,       # eV
            'L': 1e4,        # m (10 km)
            'v': 1e3,        # m/s
            'color': 'green'
        },
        'Lab Plasma': {
            'ne': 1e18,      # m^-3
            'B0': 0.1,       # T
            'Te': 10,        # eV
            'L': 0.1,        # m
            'v': 1e4,        # m/s
            'color': 'purple'
        }
    }

    # Plot 1: Term comparison for each regime (bar chart)
    ax1 = fig.add_subplot(gs[0, :])

    regime_names = list(regimes.keys())
    x_pos = np.arange(len(regime_names))
    width = 0.15

    terms_data = {name: [] for name in regime_names}
    term_names = ['Ideal', 'Resistive', 'Hall', 'Pressure', 'Inertia']

    for regime_name, regime_params in regimes.items():
        params = compute_plasma_parameters(**regime_params)

        # Compute terms
        E_ideal = ideal_mhd_term(params)
        E_resist = resistive_term(params)
        E_hall = hall_term(params)
        E_press = pressure_term(params)
        E_inert = inertia_term(params)

        # Normalize by ideal term
        terms_data[regime_name] = [
            1.0,  # Ideal
            E_resist / E_ideal,
            E_hall / E_ideal,
            E_press / E_ideal,
            E_inert / E_ideal
        ]

    # Plot bars
    for i, term_name in enumerate(term_names):
        values = [terms_data[name][i] for name in regime_names]
        ax1.bar(x_pos + i * width, values, width, label=term_name)

    ax1.set_ylabel('Magnitude (normalized to Ideal MHD)', fontsize=11)
    ax1.set_title('Relative Magnitude of Ohm\'s Law Terms by Regime',
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos + width * 2)
    ax1.set_xticklabels(regime_names, rotation=15, ha='right')
    ax1.set_yscale('log')
    ax1.legend(fontsize=9, ncol=5, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=1, color='k', linestyle='--', linewidth=1)

    # Plot 2: Terms vs scale length (tokamak)
    ax2 = fig.add_subplot(gs[1, 0])

    L_array = np.logspace(-2, 2, 100)  # 1 cm to 100 m
    tokamak_base = regimes['Tokamak Core'].copy()

    terms_vs_L = {name: [] for name in term_names}

    for L in L_array:
        tokamak_base['L'] = L
        params = compute_plasma_parameters(**tokamak_base)

        E_ideal = ideal_mhd_term(params)
        terms_vs_L['Ideal'].append(E_ideal)
        terms_vs_L['Resistive'].append(resistive_term(params))
        terms_vs_L['Hall'].append(hall_term(params))
        terms_vs_L['Pressure'].append(pressure_term(params))
        terms_vs_L['Inertia'].append(inertia_term(params))

    # Normalize by ideal
    for term_name in term_names[1:]:  # Skip ideal
        normalized = np.array(terms_vs_L[term_name]) / np.array(terms_vs_L['Ideal'])
        ax2.loglog(L_array / params['di'], normalized, linewidth=2, label=term_name)

    ax2.axhline(y=1, color='k', linestyle='--', linewidth=1,
                label='Equal to Ideal')
    ax2.set_xlabel(r'$L / d_i$ (scale length / ion skin depth)', fontsize=11)
    ax2.set_ylabel('Term / Ideal MHD Term', fontsize=11)
    ax2.set_title('Tokamak: Terms vs Scale Length', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Terms vs scale length (magnetopause)
    ax3 = fig.add_subplot(gs[1, 1])

    L_array_mp = np.logspace(3, 7, 100)  # 1 km to 10,000 km
    mp_base = regimes['Magnetopause'].copy()

    terms_vs_L_mp = {name: [] for name in term_names}

    for L in L_array_mp:
        mp_base['L'] = L
        params = compute_plasma_parameters(**mp_base)

        E_ideal = ideal_mhd_term(params)
        terms_vs_L_mp['Ideal'].append(E_ideal)
        terms_vs_L_mp['Resistive'].append(resistive_term(params))
        terms_vs_L_mp['Hall'].append(hall_term(params))
        terms_vs_L_mp['Pressure'].append(pressure_term(params))
        terms_vs_L_mp['Inertia'].append(inertia_term(params))

    # Normalize by ideal
    for term_name in term_names[1:]:
        normalized = np.array(terms_vs_L_mp[term_name]) / np.array(terms_vs_L_mp['Ideal'])
        ax3.loglog(L_array_mp / params['di'], normalized, linewidth=2, label=term_name)

    ax3.axhline(y=1, color='k', linestyle='--', linewidth=1,
                label='Equal to Ideal')
    ax3.set_xlabel(r'$L / d_i$ (scale length / ion skin depth)', fontsize=11)
    ax3.set_ylabel('Term / Ideal MHD Term', fontsize=11)
    ax3.set_title('Magnetopause: Terms vs Scale Length',
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: Summary table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    # Create table data
    table_data = [
        ['Term', 'Formula', 'When Important', 'Example'],
        ['Ideal MHD', 'E + v×B = 0', 'L >> di, collisional', 'MHD turbulence'],
        ['Resistive', '+ η·J', 'Collisional plasma', 'Resistive instabilities'],
        ['Hall', '+ J×B/(ne)', 'L ~ di', 'Magnetic reconnection'],
        ['Pressure', '- ∇pe/(ne)', 'Strong gradients', 'Current sheets'],
        ['Inertia', '+ (me/ne²)·dJ/dt', 'Fast dynamics (ω ~ ωpe)', 'Beam instabilities'],
    ]

    table = ax4.table(cellText=table_data, cellLoc='left',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f5f5f5')

    ax4.set_title('Generalized Ohm\'s Law: When Each Term Matters',
                  fontsize=12, fontweight='bold', pad=10)

    plt.suptitle('Generalized Ohm\'s Law Term-by-Term Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('ohms_law_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'ohms_law_comparison.png'\n")

    # Print detailed results
    print("\nDetailed Results by Regime:")
    print("-" * 70)

    for regime_name, regime_params in regimes.items():
        params = compute_plasma_parameters(**regime_params)

        print(f"\n{regime_name}:")
        print(f"  ne = {params['ne']:.2e} m^-3")
        print(f"  B0 = {params['B0']:.2e} T")
        print(f"  Te = {params['Te']:.2e} eV")
        print(f"  L = {params['L']:.2e} m")
        print(f"  L/di = {params['L']/params['di']:.2f}")
        print(f"  η = {params['eta']:.2e} Ω·m")

        E_ideal = ideal_mhd_term(params)
        E_resist = resistive_term(params)
        E_hall = hall_term(params)
        E_press = pressure_term(params)
        E_inert = inertia_term(params)

        print(f"\n  Term magnitudes (V/m):")
        print(f"    Ideal MHD:  {E_ideal:.2e}")
        print(f"    Resistive:  {E_resist:.2e} ({E_resist/E_ideal:.2e} × Ideal)")
        print(f"    Hall:       {E_hall:.2e} ({E_hall/E_ideal:.2e} × Ideal)")
        print(f"    Pressure:   {E_press:.2e} ({E_press/E_ideal:.2e} × Ideal)")
        print(f"    Inertia:    {E_inert:.2e} ({E_inert/E_ideal:.2e} × Ideal)")

        # Determine dominant terms
        terms = {
            'Resistive': E_resist / E_ideal,
            'Hall': E_hall / E_ideal,
            'Pressure': E_press / E_ideal,
            'Inertia': E_inert / E_ideal
        }
        important_terms = [name for name, ratio in terms.items() if ratio > 0.1]

        if important_terms:
            print(f"\n  Important corrections: {', '.join(important_terms)}")
        else:
            print(f"\n  Regime: Pure Ideal MHD")

    print("\n" + "=" * 70)

    plt.show()

if __name__ == "__main__":
    plot_ohms_law_comparison()
