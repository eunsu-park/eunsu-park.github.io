#!/usr/bin/env python3
"""
Safety Factor (q) Profile Stability Analysis

This module analyzes the stability of tokamak q-profiles with respect to
various MHD instabilities.

Stability criteria checked:
1. q(0) > 1: Sawtooth stability (internal kink mode)
2. q(a) > 2: External kink mode stability
3. Suydam criterion: Local interchange stability
4. Rational surfaces q = m/n: Locations of resonant modes
5. Magnetic shear: s = (r/q)(dq/dr)

Author: MHD Course Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class QProfileStability:
    """
    Safety factor profile stability analyzer.

    Attributes:
        r (ndarray): Minor radius grid
        a (float): Minor radius
        q (ndarray): Safety factor profile
    """

    def __init__(self, r, q):
        """
        Initialize with q-profile.

        Parameters:
            r (ndarray): Minor radius array
            q (ndarray): Safety factor array
        """
        self.r = r
        self.a = r[-1]
        self.q = q

        # Create interpolator
        self.q_interp = interp1d(r, q, kind='cubic', fill_value='extrapolate')

    def check_sawtooth_stability(self):
        """
        Check sawtooth stability: q(0) > 1

        Returns:
            dict: Stability info
        """
        q0 = self.q[0]
        stable = q0 > 1.0

        return {
            'stable': stable,
            'q0': q0,
            'criterion': 'q(0) > 1',
            'message': f'q(0) = {q0:.3f}' + (' ✓' if stable else ' ✗ UNSTABLE')
        }

    def check_kink_stability(self):
        """
        Check external kink stability: q(a) > 2

        Returns:
            dict: Stability info
        """
        qa = self.q[-1]
        stable = qa > 2.0

        return {
            'stable': stable,
            'qa': qa,
            'criterion': 'q(a) > 2',
            'message': f'q(a) = {qa:.3f}' + (' ✓' if stable else ' ✗ UNSTABLE')
        }

    def compute_shear(self):
        """
        Compute magnetic shear s = (r/q)(dq/dr).

        Returns:
            ndarray: Shear profile
        """
        # Compute dq/dr using centered differences
        dq_dr = np.gradient(self.q, self.r)

        # Shear
        shear = (self.r / self.q) * dq_dr

        # Handle r=0
        shear[0] = shear[1]

        return shear

    def suydam_criterion(self, pressure_gradient=None):
        """
        Check Suydam criterion for local interchange stability.

        Criterion: (r q'²) / (4 q²) + 2μ₀ p'/B² > 0

        Simplified version without pressure: q' > 0 (positive shear)

        Parameters:
            pressure_gradient (ndarray): dp/dr if available

        Returns:
            dict: Stability info
        """
        dq_dr = np.gradient(self.q, self.r)

        # Simplified Suydam (no pressure)
        suydam_parameter = dq_dr

        if pressure_gradient is not None:
            # Full Suydam with pressure (simplified)
            # Assuming cylindrical approximation
            term1 = (self.r * dq_dr**2) / (4 * self.q**2)
            term2 = pressure_gradient  # Simplified pressure term
            suydam_parameter = term1 + term2

        # Check if positive everywhere
        stable = np.all(suydam_parameter > 0)

        # Find unstable regions
        unstable_regions = self.r[suydam_parameter < 0]

        return {
            'stable': stable,
            'criterion': "Suydam: r q'²/(4q²) + 2μ₀p'/B² > 0",
            'parameter': suydam_parameter,
            'unstable_r': unstable_regions,
            'message': 'Suydam criterion satisfied' if stable else
                      f'Suydam unstable at {len(unstable_regions)} points'
        }

    def find_rational_surfaces(self, m_max=5, n_max=3):
        """
        Find locations of rational surfaces q = m/n.

        Parameters:
            m_max (int): Maximum poloidal mode number
            n_max (int): Maximum toroidal mode number

        Returns:
            list: Rational surface locations
        """
        rational_surfaces = []

        for m in range(1, m_max + 1):
            for n in range(1, n_max + 1):
                q_rational = m / n

                # Check if this q value exists in the profile
                if self.q[0] <= q_rational <= self.q[-1]:
                    # Find radial location
                    try:
                        # Interpolate to find r where q = m/n
                        idx = np.where(np.diff(np.sign(self.q - q_rational)))[0]
                        if len(idx) > 0:
                            r_rational = self.r[idx[0]]
                            rational_surfaces.append({
                                'm': m, 'n': n,
                                'q': q_rational,
                                'r': r_rational,
                                'rho': r_rational / self.a
                            })
                    except:
                        pass

        return rational_surfaces

    def compute_tearing_mode_parameter(self):
        """
        Compute simplified tearing mode parameter Δ'.

        Δ' > 0 indicates tearing instability.
        This is a very simplified estimate.

        Returns:
            ndarray: Tearing parameter (approximate)
        """
        # Simplified: Δ' ≈ -1/q² × d²q/dr²
        dq_dr = np.gradient(self.q, self.r)
        d2q_dr2 = np.gradient(dq_dr, self.r)

        Delta_prime = -d2q_dr2 / self.q**2

        return Delta_prime

    def plot_stability_analysis(self):
        """
        Plot comprehensive stability analysis.
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2)

        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])

        rho = self.r / self.a
        shear = self.compute_shear()
        suydam = self.suydam_criterion()
        rational = self.find_rational_surfaces()

        # q-profile with rational surfaces
        ax1.plot(rho, self.q, 'b-', linewidth=2.5, label='q(r)')
        ax1.axhline(y=1, color='r', linestyle='--', alpha=0.6,
                   label='q=1 (sawtooth)')
        ax1.axhline(y=2, color='orange', linestyle='--', alpha=0.6,
                   label='q=2 (external kink)')

        # Mark rational surfaces
        for rs in rational:
            ax1.plot(rs['rho'], rs['q'], 'go', markersize=8)
            ax1.text(rs['rho'], rs['q'], f"  {rs['m']}/{rs['n']}",
                    fontsize=9, verticalalignment='center')

        ax1.set_xlabel(r'$\rho = r/a$', fontsize=11)
        ax1.set_ylabel('Safety Factor q', fontsize=11)
        ax1.set_title('Safety Factor Profile with Rational Surfaces', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Add stability status
        sawtooth = self.check_sawtooth_stability()
        kink = self.check_kink_stability()
        status_text = f"{sawtooth['message']}\n{kink['message']}"
        ax1.text(0.02, 0.98, status_text, transform=ax1.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Magnetic shear
        ax2.plot(rho, shear, 'purple', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.fill_between(rho, 0, shear, where=(shear>0),
                         alpha=0.3, color='green', label='Positive shear')
        ax2.fill_between(rho, shear, 0, where=(shear<0),
                         alpha=0.3, color='red', label='Negative shear')
        ax2.set_xlabel(r'$\rho = r/a$', fontsize=11)
        ax2.set_ylabel(r'Shear $s = (r/q)(dq/dr)$', fontsize=11)
        ax2.set_title('Magnetic Shear Profile', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        # Suydam parameter
        ax3.plot(rho, suydam['parameter'], 'brown', linewidth=2)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5,
                   label='Stability boundary')
        ax3.fill_between(rho, 0, np.maximum(suydam['parameter'], 0),
                         alpha=0.3, color='green', label='Stable')
        if not suydam['stable']:
            ax3.fill_between(rho, np.minimum(suydam['parameter'], 0), 0,
                            alpha=0.3, color='red', label='Unstable')
        ax3.set_xlabel(r'$\rho = r/a$', fontsize=11)
        ax3.set_ylabel('Suydam Parameter', fontsize=11)
        ax3.set_title('Suydam Stability Criterion', fontsize=13)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)

        # Tearing mode parameter
        Delta_prime = self.compute_tearing_mode_parameter()
        ax4.plot(rho, Delta_prime, 'teal', linewidth=2)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5,
                   label="Δ'=0 (marginal)")
        ax4.set_xlabel(r'$\rho = r/a$', fontsize=11)
        ax4.set_ylabel(r"Tearing Parameter $\Delta'$", fontsize=11)
        ax4.set_title('Tearing Mode Stability (simplified)', fontsize=13)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)

        # Stability diagram
        criteria = ['Sawtooth\n(q₀>1)', 'Kink\n(qₐ>2)', 'Suydam']
        status = [
            1 if sawtooth['stable'] else 0,
            1 if kink['stable'] else 0,
            1 if suydam['stable'] else 0
        ]
        colors_bar = ['green' if s else 'red' for s in status]

        bars = ax5.barh(criteria, status, color=colors_bar, alpha=0.7)
        ax5.set_xlim([0, 1])
        ax5.set_xlabel('Status', fontsize=11)
        ax5.set_title('Stability Summary', fontsize=13)
        ax5.set_xticks([0, 1])
        ax5.set_xticklabels(['UNSTABLE', 'STABLE'])
        ax5.grid(True, alpha=0.3, axis='x')

        # Add checkmarks/crosses
        for i, (bar, s) in enumerate(zip(bars, status)):
            symbol = '✓' if s else '✗'
            color = 'green' if s else 'red'
            ax5.text(0.5, i, symbol, fontsize=20, color=color,
                    ha='center', va='center', weight='bold')

        plt.tight_layout()
        return fig

    def generate_stability_report(self):
        """
        Generate text stability report.

        Returns:
            str: Stability report
        """
        report = []
        report.append("=" * 60)
        report.append("TOKAMAK Q-PROFILE STABILITY ANALYSIS REPORT")
        report.append("=" * 60)

        # Sawtooth
        sawtooth = self.check_sawtooth_stability()
        report.append(f"\n1. Sawtooth Stability ({sawtooth['criterion']})")
        report.append(f"   {sawtooth['message']}")

        # Kink
        kink = self.check_kink_stability()
        report.append(f"\n2. External Kink Stability ({kink['criterion']})")
        report.append(f"   {kink['message']}")

        # Suydam
        suydam = self.suydam_criterion()
        report.append(f"\n3. Suydam Criterion")
        report.append(f"   {suydam['message']}")

        # Rational surfaces
        rational = self.find_rational_surfaces()
        report.append(f"\n4. Rational Surfaces (q = m/n)")
        report.append(f"   Found {len(rational)} rational surfaces:")
        for rs in rational[:10]:  # Limit to first 10
            report.append(f"   - q = {rs['m']}/{rs['n']} = {rs['q']:.3f} at ρ = {rs['rho']:.3f}")

        # Overall assessment
        report.append(f"\n" + "=" * 60)
        all_stable = sawtooth['stable'] and kink['stable'] and suydam['stable']
        if all_stable:
            report.append("OVERALL: ALL STABILITY CRITERIA SATISFIED ✓")
        else:
            report.append("OVERALL: STABILITY ISSUES DETECTED ✗")
        report.append("=" * 60)

        return "\n".join(report)


def main():
    """
    Main function demonstrating q-profile stability analysis.
    """
    print("=" * 60)
    print("Q-Profile Stability Analysis")
    print("=" * 60)

    # Create sample q-profiles
    r = np.linspace(0, 1.0, 200)

    # Profile 1: Stable (monotonic, q0>1, qa>2)
    q_stable = 1.2 + 2.0 * r**2

    # Profile 2: Unstable (q0<1, sawtooth unstable)
    q_unstable_sawtooth = 0.8 + 2.5 * r**2

    # Profile 3: Unstable kink (qa<2)
    q_unstable_kink = 1.1 + 0.7 * r**2

    # Analyze stable profile
    print("\nAnalyzing STABLE q-profile:")
    analyzer = QProfileStability(r, q_stable)
    report = analyzer.generate_stability_report()
    print(report)

    fig1 = analyzer.plot_stability_analysis()
    fig1.suptitle('Stable Q-Profile Analysis', fontsize=14, weight='bold', y=0.995)

    # Analyze unstable profile
    print("\n\nAnalyzing UNSTABLE q-profile (sawtooth):")
    analyzer2 = QProfileStability(r, q_unstable_sawtooth)
    report2 = analyzer2.generate_stability_report()
    print(report2)

    fig2 = analyzer2.plot_stability_analysis()
    fig2.suptitle('Unstable Q-Profile Analysis (Sawtooth)', fontsize=14, weight='bold', y=0.995)

    plt.savefig('/tmp/q_profile_stability.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to /tmp/q_profile_stability.png")

    plt.show()


if __name__ == "__main__":
    main()
