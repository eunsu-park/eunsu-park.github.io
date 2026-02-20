#!/usr/bin/env python3
"""
Magnetic Field Polarity Reversal

This module demonstrates magnetic field polarity reversals similar to those
observed in Earth's paleomagnetic record and the Sun's 22-year magnetic cycle.

The model uses an α-Ω dynamo with stochastic noise to trigger reversals.
Key features:
- Bistable dipole states (normal/reversed polarity)
- Stochastic transitions between states
- Waiting time distribution analysis
- Comparison with paleomagnetic observations

Author: MHD Course Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class ReversalDynamo:
    """
    Stochastic dynamo model with field reversals.

    This implements a low-order model of the geodynamo/solar dynamo
    that exhibits polarity reversals due to stochastic fluctuations.

    Attributes:
        alpha (float): Alpha effect strength
        omega (float): Omega effect (differential rotation)
        eta (float): Magnetic diffusivity
        noise_strength (float): Amplitude of stochastic forcing
    """

    def __init__(self, alpha=1.0, omega=10.0, eta=0.1, noise_strength=0.5):
        """
        Initialize reversal dynamo.

        Parameters:
            alpha (float): Alpha effect strength
            omega (float): Omega effect
            eta (float): Diffusivity
            noise_strength (float): Noise amplitude
        """
        self.alpha = alpha
        self.omega = omega
        self.eta = eta
        self.noise_strength = noise_strength

        # Dynamo number
        self.D = alpha * omega / eta**2

        # State variables (dipole + quadrupole components)
        self.dipole = 0.1
        self.quadrupole = 0.1

        # History
        self.time_history = []
        self.dipole_history = []
        self.quadrupole_history = []
        self.reversal_times = []

    def rhs(self, state, t, noise):
        """
        Right-hand side of dynamo equations.

        Simplified model based on Rikitake dynamo / Lorenz-like system.

        Parameters:
            state (list): [dipole, quadrupole]
            t (float): Time
            noise (float): Noise realization

        Returns:
            list: Time derivatives
        """
        d, q = state

        # Coupled equations with cubic nonlinearity (saturation)
        # and stochastic forcing
        dd_dt = self.alpha * q - self.eta * d - d * (d**2 + q**2) + noise
        dq_dt = self.omega * d - self.eta * q - q * (d**2 + q**2)

        return [dd_dt, dq_dt]

    def detect_reversal(self, dipole_current, dipole_previous):
        """
        Detect if a reversal has occurred.

        Parameters:
            dipole_current (float): Current dipole value
            dipole_previous (float): Previous dipole value

        Returns:
            bool: True if reversal detected
        """
        # Reversal = sign change of dipole
        return dipole_current * dipole_previous < 0

    def run_simulation(self, t_end=1000.0, dt=0.01):
        """
        Run stochastic dynamo simulation.

        Parameters:
            t_end (float): End time
            dt (float): Time step
        """
        n_steps = int(t_end / dt)
        t_array = np.linspace(0, t_end, n_steps)

        print("Running magnetic reversal simulation...")
        print(f"Dynamo number D = {self.D:.2f}")
        print(f"Noise strength = {self.noise_strength:.2f}")

        # Initialize
        state = np.array([self.dipole, self.quadrupole])
        self.time_history = []
        self.dipole_history = []
        self.quadrupole_history = []
        self.reversal_times = []

        previous_dipole = state[0]

        for i, t in enumerate(t_array):
            # Generate noise
            noise = self.noise_strength * np.random.randn()

            # Integrate one step
            state_new = odeint(self.rhs, state, [t, t + dt], args=(noise,))[-1]
            state = state_new

            # Store
            self.time_history.append(t)
            self.dipole_history.append(state[0])
            self.quadrupole_history.append(state[1])

            # Detect reversal
            if self.detect_reversal(state[0], previous_dipole):
                self.reversal_times.append(t)
                print(f"Reversal at t = {t:.2f}")

            previous_dipole = state[0]

            if i % 10000 == 0:
                print(f"Step {i}/{n_steps}, t={t:.1f}")

        print(f"\nSimulation complete!")
        print(f"Total reversals: {len(self.reversal_times)}")

        if len(self.reversal_times) > 1:
            waiting_times = np.diff(self.reversal_times)
            print(f"Mean waiting time: {np.mean(waiting_times):.1f}")
            print(f"Std waiting time: {np.std(waiting_times):.1f}")

    def plot_field_evolution(self):
        """
        Plot magnetic field evolution and reversals.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        # Dipole evolution
        ax1.plot(self.time_history, self.dipole_history, 'b-', linewidth=1.5,
                label='Dipole component')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # Mark reversals
        for t_rev in self.reversal_times:
            ax1.axvline(x=t_rev, color='r', alpha=0.3, linewidth=0.5)

        ax1.set_ylabel('Dipole Field', fontsize=11)
        ax1.set_title('Magnetic Field Reversals (Dipole Component)', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Quadrupole evolution
        ax2.plot(self.time_history, self.quadrupole_history, 'g-', linewidth=1.5,
                label='Quadrupole component')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Quadrupole Field', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        # Phase space
        ax3.plot(self.dipole_history, self.quadrupole_history, 'b-',
                alpha=0.5, linewidth=0.5)
        ax3.plot(self.dipole_history[0], self.quadrupole_history[0], 'go',
                markersize=8, label='Start')
        ax3.plot(self.dipole_history[-1], self.quadrupole_history[-1], 'ro',
                markersize=8, label='End')
        ax3.set_xlabel('Dipole', fontsize=11)
        ax3.set_ylabel('Quadrupole', fontsize=11)
        ax3.set_title('Phase Space Trajectory', fontsize=13)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_reversal_statistics(self):
        """
        Plot reversal waiting time statistics.
        """
        if len(self.reversal_times) < 2:
            print("Not enough reversals for statistics")
            return None

        waiting_times = np.diff(self.reversal_times)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram of waiting times
        ax1.hist(waiting_times, bins=20, density=True, alpha=0.7,
                color='blue', edgecolor='black')

        # Theoretical exponential (Poisson process)
        mean_wait = np.mean(waiting_times)
        x_theory = np.linspace(0, np.max(waiting_times), 100)
        y_theory = (1 / mean_wait) * np.exp(-x_theory / mean_wait)
        ax1.plot(x_theory, y_theory, 'r-', linewidth=2,
                label=f'Exponential (λ={1/mean_wait:.3f})')

        ax1.set_xlabel('Waiting Time', fontsize=11)
        ax1.set_ylabel('Probability Density', fontsize=11)
        ax1.set_title('Distribution of Reversal Waiting Times', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Cumulative distribution
        sorted_waits = np.sort(waiting_times)
        cumulative = np.arange(1, len(sorted_waits) + 1) / len(sorted_waits)

        ax2.plot(sorted_waits, cumulative, 'b-', linewidth=2,
                label='Empirical CDF')

        # Theoretical CDF
        cdf_theory = 1 - np.exp(-x_theory / mean_wait)
        ax2.plot(x_theory, cdf_theory, 'r--', linewidth=2,
                label='Exponential CDF')

        ax2.set_xlabel('Waiting Time', fontsize=11)
        ax2.set_ylabel('Cumulative Probability', fontsize=11)
        ax2.set_title('Cumulative Distribution Function', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        return fig

    def plot_paleomagnetic_comparison(self):
        """
        Plot comparison with schematic paleomagnetic data.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Our simulation (zoom to show detail)
        t_max = min(200, self.time_history[-1])
        idx = np.searchsorted(self.time_history, t_max)

        dipole_sign = np.sign(self.dipole_history[:idx])
        ax1.fill_between(self.time_history[:idx], -1, 1,
                         where=(dipole_sign > 0),
                         alpha=0.7, color='black', label='Normal polarity')
        ax1.fill_between(self.time_history[:idx], -1, 1,
                         where=(dipole_sign < 0),
                         alpha=0.7, color='white', edgecolor='black',
                         label='Reversed polarity')
        ax1.set_ylabel('Polarity', fontsize=11)
        ax1.set_title('Simulated Magnetic Polarity History', fontsize=13)
        ax1.set_ylim([-1.5, 1.5])
        ax1.set_yticks([-1, 0, 1])
        ax1.set_yticklabels(['Reversed', 'Transition', 'Normal'])
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='x')

        # Schematic Earth paleomagnetic record (simplified)
        # Based on geomagnetic polarity time scale
        earth_times = np.array([0, 30, 60, 80, 100, 125, 150, 170, 200])
        earth_polarity = np.array([1, 1, -1, -1, 1, -1, 1, 1, -1])

        for i in range(len(earth_times) - 1):
            color = 'black' if earth_polarity[i] > 0 else 'white'
            ax2.fill_between([earth_times[i], earth_times[i+1]], -1, 1,
                            alpha=0.7, color=color, edgecolor='black')

        ax2.set_xlabel('Time (relative units)', fontsize=11)
        ax2.set_ylabel('Polarity', fontsize=11)
        ax2.set_title('Schematic Earth Paleomagnetic Record', fontsize=13)
        ax2.set_ylim([-1.5, 1.5])
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['Reversed', 'Transition', 'Normal'])
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig


def main():
    """
    Main function demonstrating magnetic field reversals.
    """
    print("=" * 60)
    print("Magnetic Field Polarity Reversal Simulation")
    print("=" * 60)

    # Create dynamo
    dynamo = ReversalDynamo(
        alpha=1.0,
        omega=10.0,
        eta=0.1,
        noise_strength=0.8  # Higher noise → more reversals
    )

    # Run simulation
    dynamo.run_simulation(t_end=1000.0, dt=0.01)

    # Plot results
    dynamo.plot_field_evolution()
    if len(dynamo.reversal_times) >= 2:
        dynamo.plot_reversal_statistics()
    dynamo.plot_paleomagnetic_comparison()

    plt.savefig('/tmp/field_reversal.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to /tmp/field_reversal.png")

    plt.show()


if __name__ == "__main__":
    main()
