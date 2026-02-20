#!/usr/bin/env python3
"""
Edge Localized Mode (ELM) Crash - Simplified Model

This module implements a simplified model of Edge Localized Modes (ELMs)
in tokamak H-mode plasmas.

Key physics:
- ELMs are periodic instabilities at the plasma edge (pedestal region)
- Driven by peeling-ballooning modes when pressure gradient or current
  exceeds stability boundary
- Lead to rapid loss of edge energy and particles
- ELM cycle: buildup → crash → recovery

Peeling-ballooning stability in (J_edge, α) space where:
    J_edge = edge current density
    α = pressure gradient parameter

Author: MHD Course Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class ELMModel:
    """
    Simplified ELM crash model with peeling-ballooning stability.

    Attributes:
        p_edge (float): Edge pedestal pressure
        j_edge (float): Edge current density
        alpha (float): Pressure gradient parameter
        tau_buildup (float): Pedestal buildup timescale
    """

    def __init__(self, tau_buildup=0.1, tau_crash=0.001,
                 p_source=1.0, alpha_crit=1.0, j_crit=1.0):
        """
        Initialize ELM model.

        Parameters:
            tau_buildup (float): Buildup time (s)
            tau_crash (float): Crash time (s)
            p_source (float): Heating power (normalized)
            alpha_crit (float): Critical pressure gradient
            j_crit (float): Critical current density
        """
        self.tau_buildup = tau_buildup
        self.tau_crash = tau_crash
        self.p_source = p_source
        self.alpha_crit = alpha_crit
        self.j_crit = j_crit

        # State
        self.p_ped = 0.1  # Pedestal height
        self.w_ped = 0.05  # Pedestal width

        # Derived quantities
        self.alpha = 0.0
        self.j_edge = 0.0

        # History
        self.time_history = []
        self.p_history = []
        self.alpha_history = []
        self.j_history = []
        self.elm_times = []
        self.elm_energy = []

    def compute_stability_parameters(self):
        """
        Compute peeling-ballooning parameters from pedestal.

        Returns:
            tuple: (alpha, j_edge)
        """
        # Pressure gradient parameter
        # α ~ ∇p in pedestal region
        alpha = self.p_ped / self.w_ped

        # Edge current (bootstrap + driven)
        # Simplified: j_edge ~ sqrt(p_ped)
        j_edge = np.sqrt(self.p_ped)

        return alpha, j_edge

    def peeling_ballooning_boundary(self, j_array):
        """
        Peeling-ballooning stability boundary in (j, α) space.

        Simplified boundary:
        - Low j: ballooning limit α_max ~ α_crit
        - High j: peeling limit α_max ~ α_crit - (j/j_crit)²

        Parameters:
            j_array (ndarray): Current density array

        Returns:
            ndarray: Critical alpha boundary
        """
        # Combined peeling-ballooning boundary
        j_norm = j_array / self.j_crit

        # Ballooning branch (low j)
        alpha_ballooning = self.alpha_crit * np.ones_like(j_array)

        # Peeling branch (high j)
        alpha_peeling = self.alpha_crit * (1 - j_norm**2)

        # Take minimum (most restrictive)
        alpha_boundary = np.minimum(alpha_ballooning, alpha_peeling)

        return np.maximum(alpha_boundary, 0)

    def is_elm_unstable(self):
        """
        Check if ELM is unstable (above stability boundary).

        Returns:
            bool: True if unstable
        """
        alpha, j_edge = self.compute_stability_parameters()

        # Check against boundary
        alpha_max = self.peeling_ballooning_boundary(np.array([j_edge]))[0]

        unstable = alpha > alpha_max

        return unstable

    def elm_crash(self):
        """
        Execute ELM crash: rapid loss of pedestal.

        Returns:
            float: Energy lost in ELM
        """
        # Energy loss proportional to pedestal height
        energy_loss = self.p_ped * 0.5  # Lose ~50% of pedestal

        # Reduce pedestal height
        self.p_ped *= 0.3  # Crash to 30% of pre-ELM value

        # Widen pedestal
        self.w_ped *= 1.5

        return energy_loss

    def evolve_pedestal(self, dt):
        """
        Evolve pedestal during inter-ELM period.

        dp_ped/dt = P_source - p_ped/τ_buildup

        Parameters:
            dt (float): Time step
        """
        # Buildup equation
        dp_dt = self.p_source - self.p_ped / self.tau_buildup

        # Narrow pedestal during buildup
        dw_dt = -0.1 * self.w_ped / self.tau_buildup

        # Euler step
        self.p_ped += dt * dp_dt
        self.w_ped += dt * dw_dt

        # Minimum width
        self.w_ped = max(self.w_ped, 0.02)

    def run_elm_cycle(self, t_end=2.0, dt=0.0001):
        """
        Run ELM cycle simulation.

        Parameters:
            t_end (float): End time (s)
            dt (float): Time step (s)
        """
        n_steps = int(t_end / dt)
        t = 0.0

        print("Running ELM cycle simulation...")
        print(f"Critical α = {self.alpha_crit:.2f}")
        print(f"Critical j = {self.j_crit:.2f}")

        self.time_history = []
        self.p_history = []
        self.alpha_history = []
        self.j_history = []
        self.elm_times = []
        self.elm_energy = []

        for step in range(n_steps):
            # Store state
            alpha, j_edge = self.compute_stability_parameters()
            self.time_history.append(t)
            self.p_history.append(self.p_ped)
            self.alpha_history.append(alpha)
            self.j_history.append(j_edge)

            # Check ELM stability
            if self.is_elm_unstable():
                # ELM crash
                energy = self.elm_crash()
                self.elm_times.append(t)
                self.elm_energy.append(energy)
                print(f"ELM at t={t:.4f} s, ΔW={energy:.3f}")
            else:
                # Inter-ELM buildup
                self.evolve_pedestal(dt)

            t += dt

        print(f"\nSimulation complete!")
        print(f"Total ELMs: {len(self.elm_times)}")
        if len(self.elm_times) > 1:
            elm_freq = 1.0 / np.mean(np.diff(self.elm_times))
            print(f"ELM frequency: {elm_freq:.1f} Hz")

    def plot_elm_cycle(self):
        """
        Plot ELM cycle evolution.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        t = np.array(self.time_history)

        # Pedestal height
        ax1.plot(t, self.p_history, 'b-', linewidth=1.5)
        for t_elm in self.elm_times:
            ax1.axvline(x=t_elm, color='r', alpha=0.3, linewidth=0.8)
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Pedestal Height', fontsize=11)
        ax1.set_title('Pedestal Evolution (ELM Crashes)', fontsize=13)
        ax1.grid(True, alpha=0.3)

        # Stability parameters
        ax2.plot(t, self.alpha_history, 'purple', linewidth=1.5, label='α')
        ax2.axhline(y=self.alpha_crit, color='r', linestyle='--',
                   linewidth=2, label='α_crit')
        for t_elm in self.elm_times:
            ax2.axvline(x=t_elm, color='r', alpha=0.3, linewidth=0.8)
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Pressure Gradient α', fontsize=11)
        ax2.set_title('Stability Parameter Evolution', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        # Peeling-ballooning diagram
        j_array = np.linspace(0, 1.5 * self.j_crit, 200)
        alpha_boundary = self.peeling_ballooning_boundary(j_array)

        ax3.plot(j_array, alpha_boundary, 'r-', linewidth=2.5,
                label='Stability boundary')
        ax3.fill_between(j_array, 0, alpha_boundary, alpha=0.3,
                        color='green', label='Stable')
        ax3.fill_between(j_array, alpha_boundary, 2*self.alpha_crit,
                        alpha=0.3, color='red', label='Unstable (ELM)')

        # Plot trajectory
        ax3.plot(self.j_history, self.alpha_history, 'b-',
                linewidth=1, alpha=0.5, label='Trajectory')

        # Mark ELM times
        elm_indices = [np.argmin(np.abs(t - t_elm)) for t_elm in self.elm_times]
        j_elm = [self.j_history[i] for i in elm_indices]
        alpha_elm = [self.alpha_history[i] for i in elm_indices]
        ax3.plot(j_elm, alpha_elm, 'ro', markersize=8, label='ELM crashes')

        ax3.set_xlabel('Edge Current j_edge', fontsize=11)
        ax3.set_ylabel('Pressure Gradient α', fontsize=11)
        ax3.set_title('Peeling-Ballooning Stability Diagram', fontsize=13)
        ax3.set_xlim([0, 1.5 * self.j_crit])
        ax3.set_ylim([0, 2 * self.alpha_crit])
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)

        # ELM energy vs pedestal height
        if len(self.elm_energy) > 0:
            # Find pedestal height just before each ELM
            p_before_elm = []
            for t_elm in self.elm_times:
                idx = np.argmin(np.abs(t - t_elm))
                if idx > 0:
                    p_before_elm.append(self.p_history[idx-1])

            ax4.plot(p_before_elm, self.elm_energy, 'go', markersize=8)

            # Fit line
            if len(p_before_elm) > 1:
                coeffs = np.polyfit(p_before_elm, self.elm_energy, 1)
                p_fit = np.linspace(min(p_before_elm), max(p_before_elm), 100)
                E_fit = np.polyval(coeffs, p_fit)
                ax4.plot(p_fit, E_fit, 'r--', linewidth=2,
                        label=f'Fit: ΔW = {coeffs[0]:.2f}p + {coeffs[1]:.3f}')

            ax4.set_xlabel('Pedestal Height (pre-ELM)', fontsize=11)
            ax4.set_ylabel('ELM Energy Loss ΔW', fontsize=11)
            ax4.set_title('ELM Energy vs Pedestal Height', fontsize=13)
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=10)
        else:
            ax4.text(0.5, 0.5, 'No ELMs occurred',
                    transform=ax4.transAxes, ha='center', va='center',
                    fontsize=14)

        plt.tight_layout()
        return fig


def main():
    """
    Main function demonstrating ELM crash model.
    """
    print("=" * 60)
    print("Edge Localized Mode (ELM) Simplified Model")
    print("=" * 60)

    # Create ELM model
    elm = ELMModel(
        tau_buildup=0.1,    # 100 ms buildup
        tau_crash=0.001,    # 1 ms crash
        p_source=1.5,       # Heating power
        alpha_crit=1.0,     # Critical gradient
        j_crit=0.8          # Critical current
    )

    # Run simulation
    elm.run_elm_cycle(t_end=2.0, dt=0.0001)

    # Plot results
    elm.plot_elm_cycle()

    plt.savefig('/tmp/elm_crash.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to /tmp/elm_crash.png")

    plt.show()


if __name__ == "__main__":
    main()
