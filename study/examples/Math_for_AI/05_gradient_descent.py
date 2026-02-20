"""
Gradient Descent Optimization Algorithms

This script implements various gradient descent optimizers from scratch:
- Gradient Descent (GD)
- Stochastic Gradient Descent (SGD)
- Momentum
- Adam

We optimize the Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
This is a classic non-convex test function with a narrow valley.

Author: Math for AI Examples
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List


class GradientDescent:
    """Basic Gradient Descent optimizer."""

    def __init__(self, learning_rate: float = 0.001):
        self.lr = learning_rate

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Update parameters using gradient descent."""
        return params - self.lr * grads


class Momentum:
    """Gradient Descent with Momentum."""

    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocity = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Update parameters using momentum."""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        # v_t = beta * v_{t-1} + grad
        self.velocity = self.beta * self.velocity + grads
        # theta = theta - lr * v_t
        return params - self.lr * self.velocity


class Adam:
    """Adam optimizer (Adaptive Moment Estimation)."""

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment (mean)
        self.v = None  # Second moment (variance)
        self.t = 0     # Timestep

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Update parameters using Adam."""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Update biased first and second moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update parameters
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    """
    Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
    Global minimum at (a, a^2) with f(a, a^2) = 0
    """
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2


def rosenbrock_gradient(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    """
    Gradient of Rosenbrock function.
    df/dx = -2(a-x) - 4bx(y-x^2)
    df/dy = 2b(y-x^2)
    """
    grad = np.zeros_like(x)
    grad[0] = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
    grad[1] = 2 * b * (x[1] - x[0]**2)
    return grad


def optimize(
    optimizer,
    initial_point: np.ndarray,
    n_iterations: int = 1000,
    a: float = 1.0,
    b: float = 100.0
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Run optimization and track trajectory.

    Returns:
        trajectory: List of parameter values at each step
        losses: List of loss values at each step
    """
    params = initial_point.copy()
    trajectory = [params.copy()]
    losses = [rosenbrock(params, a, b)]

    for i in range(n_iterations):
        grads = rosenbrock_gradient(params, a, b)
        params = optimizer.step(params, grads)

        trajectory.append(params.copy())
        losses.append(rosenbrock(params, a, b))

    return trajectory, losses


def learning_rate_schedule_demo():
    """Demonstrate learning rate scheduling strategies."""
    print("\n" + "="*60)
    print("Learning Rate Schedule Demonstration")
    print("="*60)

    n_iterations = 1000
    initial_lr = 0.1

    # Step decay: lr = lr_0 * decay_rate^(epoch / drop_every)
    step_decay = lambda t: initial_lr * (0.5 ** (t // 100))

    # Exponential decay: lr = lr_0 * exp(-decay_rate * t)
    exp_decay = lambda t: initial_lr * np.exp(-0.005 * t)

    # Cosine annealing: lr = lr_min + 0.5(lr_max - lr_min)(1 + cos(t*pi/T))
    cosine_decay = lambda t: 0.001 + 0.5 * (initial_lr - 0.001) * \
                             (1 + np.cos(np.pi * t / n_iterations))

    # Linear decay: lr = lr_0 * (1 - t/T)
    linear_decay = lambda t: initial_lr * (1 - t / n_iterations)

    steps = np.arange(n_iterations)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, [step_decay(t) for t in steps], label='Step Decay', linewidth=2)
    plt.plot(steps, [exp_decay(t) for t in steps], label='Exponential Decay', linewidth=2)
    plt.plot(steps, [cosine_decay(t) for t in steps], label='Cosine Annealing', linewidth=2)
    plt.plot(steps, [linear_decay(t) for t in steps], label='Linear Decay', linewidth=2)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedules', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_rate_schedules.png', dpi=150, bbox_inches='tight')
    print("Saved learning rate schedules plot to 'learning_rate_schedules.png'")


def compare_optimizers():
    """Compare different optimization algorithms on Rosenbrock function."""
    print("\n" + "="*60)
    print("Comparing Optimization Algorithms on Rosenbrock Function")
    print("="*60)

    # Starting point
    x0 = np.array([-1.0, 1.0])
    n_iterations = 500

    # Initialize optimizers
    optimizers = {
        'GD (lr=0.001)': GradientDescent(learning_rate=0.001),
        'Momentum (lr=0.001)': Momentum(learning_rate=0.001, beta=0.9),
        'Adam (lr=0.01)': Adam(learning_rate=0.01)
    }

    results = {}

    # Run optimization for each optimizer
    for name, optimizer in optimizers.items():
        trajectory, losses = optimize(optimizer, x0, n_iterations)
        results[name] = {
            'trajectory': trajectory,
            'losses': losses,
            'final_point': trajectory[-1],
            'final_loss': losses[-1]
        }

        print(f"\n{name}:")
        print(f"  Final point: ({trajectory[-1][0]:.6f}, {trajectory[-1][1]:.6f})")
        print(f"  Final loss: {losses[-1]:.6e}")
        print(f"  Target: (1.0, 1.0), Loss: 0.0")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Contour plot with trajectories
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rosenbrock(np.array([X[i, j], Y[i, j]]))

    # Use log scale for better visualization
    levels = np.logspace(-1, 3.5, 20)
    contour = ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)

    colors = ['red', 'blue', 'green']
    for (name, result), color in zip(results.items(), colors):
        traj = np.array(result['trajectory'])
        ax1.plot(traj[:, 0], traj[:, 1], 'o-', color=color,
                label=name, markersize=2, linewidth=1.5, alpha=0.7)
        ax1.plot(traj[0, 0], traj[0, 1], 'ko', markersize=8, label='Start' if color == 'red' else '')

    ax1.plot(1.0, 1.0, 'r*', markersize=15, label='Global Minimum')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Optimization Trajectories', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss curves
    for name, result in results.items():
        ax2.semilogy(result['losses'], label=name, linewidth=2)

    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Loss (log scale)', fontsize=12)
    ax2.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison plot to 'optimizer_comparison.png'")


if __name__ == "__main__":
    print("="*60)
    print("Gradient Descent Optimization Examples")
    print("="*60)

    # Compare different optimizers
    compare_optimizers()

    # Demonstrate learning rate schedules
    learning_rate_schedule_demo()

    print("\n" + "="*60)
    print("Key Takeaways:")
    print("="*60)
    print("1. Basic GD can be slow on narrow valleys (Rosenbrock)")
    print("2. Momentum helps accelerate convergence in relevant directions")
    print("3. Adam adapts learning rates per parameter, often converges fastest")
    print("4. Learning rate schedules can improve final convergence")
    print("5. Choice of optimizer depends on problem structure and constraints")
