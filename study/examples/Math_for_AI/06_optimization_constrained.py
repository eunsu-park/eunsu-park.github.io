"""
Constrained and Unconstrained Optimization

This script demonstrates:
1. Unconstrained optimization using scipy.optimize
2. Lagrange multipliers for equality constraints
3. KKT conditions for inequality constraints
4. Convex vs non-convex optimization comparison

Author: Math for AI Examples
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint
from typing import Callable, Tuple


def unconstrained_optimization_demo():
    """Demonstrate unconstrained optimization with scipy.optimize."""
    print("\n" + "="*60)
    print("1. Unconstrained Optimization")
    print("="*60)

    # Objective: Minimize f(x,y) = x^2 + y^2 + 2x + 4y + 5
    # Analytical solution: x = -1, y = -2, f_min = 0
    def objective(x):
        return x[0]**2 + x[1]**2 + 2*x[0] + 4*x[1] + 5

    def gradient(x):
        return np.array([2*x[0] + 2, 2*x[1] + 4])

    x0 = np.array([5.0, 5.0])

    # Using different optimization methods
    methods = ['BFGS', 'CG', 'Newton-CG', 'L-BFGS-B']
    results = {}

    for method in methods:
        if method == 'Newton-CG':
            result = minimize(objective, x0, method=method, jac=gradient)
        else:
            result = minimize(objective, x0, method=method)

        results[method] = result
        print(f"\n{method}:")
        print(f"  Optimal point: ({result.x[0]:.6f}, {result.x[1]:.6f})")
        print(f"  Minimum value: {result.fun:.6f}")
        print(f"  Iterations: {result.nit}")

    print("\nAnalytical solution: x = -1.0, y = -2.0, f_min = 0.0")


def lagrange_multipliers_demo():
    """
    Demonstrate Lagrange multipliers for equality constraints.

    Problem: Minimize f(x,y) = x^2 + y^2
             Subject to: g(x,y) = x + y - 1 = 0

    Using Lagrange multipliers:
    L(x,y,λ) = f(x,y) - λ*g(x,y) = x^2 + y^2 - λ(x + y - 1)

    KKT conditions:
    ∂L/∂x = 2x - λ = 0  =>  x = λ/2
    ∂L/∂y = 2y - λ = 0  =>  y = λ/2
    ∂L/∂λ = -(x + y - 1) = 0  =>  x + y = 1

    Solution: x = y = 0.5, f_min = 0.5
    """
    print("\n" + "="*60)
    print("2. Lagrange Multipliers (Equality Constraints)")
    print("="*60)
    print("\nProblem: Minimize f(x,y) = x^2 + y^2")
    print("         Subject to: x + y = 1")

    # Manual solution using Lagrange multipliers
    print("\n--- Analytical Solution (Lagrange Multipliers) ---")
    print("Setting up Lagrangian: L(x,y,λ) = x^2 + y^2 - λ(x + y - 1)")
    print("∂L/∂x = 2x - λ = 0")
    print("∂L/∂y = 2y - λ = 0")
    print("∂L/∂λ = -(x + y - 1) = 0")
    print("\nFrom first two equations: x = y = λ/2")
    print("From constraint: λ/2 + λ/2 = 1  =>  λ = 1")
    print("Solution: x = 0.5, y = 0.5, f_min = 0.5")

    # Numerical solution using scipy
    print("\n--- Numerical Solution (scipy.optimize) ---")

    def objective(x):
        return x[0]**2 + x[1]**2

    def constraint(x):
        return x[0] + x[1] - 1

    constraints = {'type': 'eq', 'fun': constraint}
    x0 = np.array([2.0, 2.0])

    result = minimize(objective, x0, method='SLSQP', constraints=constraints)

    print(f"Optimal point: ({result.x[0]:.6f}, {result.x[1]:.6f})")
    print(f"Minimum value: {result.fun:.6f}")
    print(f"Constraint satisfied: {constraint(result.x):.6e}")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot objective function contours
    x = np.linspace(-0.5, 2, 300)
    y = np.linspace(-0.5, 2, 300)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    contour = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)

    # Plot constraint line
    x_line = np.linspace(-0.5, 2, 100)
    y_line = 1 - x_line
    ax.plot(x_line, y_line, 'r-', linewidth=3, label='Constraint: x + y = 1')

    # Plot optimal point
    ax.plot(0.5, 0.5, 'r*', markersize=20, label='Optimal Point (0.5, 0.5)')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Lagrange Multipliers: Min x² + y² s.t. x + y = 1',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('lagrange_multipliers.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to 'lagrange_multipliers.png'")


def kkt_conditions_demo():
    """
    Demonstrate KKT conditions for inequality constraints.

    Problem: Minimize f(x,y) = (x-2)^2 + (y-1)^2
             Subject to: x + y <= 1  (g1)
                        x >= 0      (g2)
                        y >= 0      (g3)

    KKT conditions:
    1. Stationarity: ∇f(x*) + Σμ_i*∇g_i(x*) = 0
    2. Primal feasibility: g_i(x*) <= 0
    3. Dual feasibility: μ_i >= 0
    4. Complementary slackness: μ_i * g_i(x*) = 0
    """
    print("\n" + "="*60)
    print("3. KKT Conditions (Inequality Constraints)")
    print("="*60)
    print("\nProblem: Minimize f(x,y) = (x-2)^2 + (y-1)^2")
    print("         Subject to: x + y <= 1, x >= 0, y >= 0")

    def objective(x):
        return (x[0] - 2)**2 + (x[1] - 1)**2

    # Constraints: g(x) <= 0
    constraints = [
        {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]},  # x + y <= 1
        {'type': 'ineq', 'fun': lambda x: x[0]},              # x >= 0
        {'type': 'ineq', 'fun': lambda x: x[1]}               # y >= 0
    ]

    x0 = np.array([0.5, 0.5])
    result = minimize(objective, x0, method='SLSQP', constraints=constraints)

    print(f"\nOptimal point: ({result.x[0]:.6f}, {result.x[1]:.6f})")
    print(f"Minimum value: {result.fun:.6f}")

    # Check which constraints are active
    print("\nConstraint activity:")
    print(f"  x + y - 1 = {result.x[0] + result.x[1] - 1:.6f} (active if ≈ 0)")
    print(f"  x = {result.x[0]:.6f} (active if ≈ 0)")
    print(f"  y = {result.x[1]:.6f} (active if ≈ 0)")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 8))

    x = np.linspace(-0.5, 2.5, 300)
    y = np.linspace(-0.5, 2.5, 300)
    X, Y = np.meshgrid(x, y)
    Z = (X - 2)**2 + (Y - 1)**2

    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)

    # Plot feasible region
    x_constraint = np.linspace(0, 1, 100)
    y_constraint = 1 - x_constraint

    ax.fill([0, 1, 0, 0], [0, 0, 1, 0], color='lightblue',
            alpha=0.3, label='Feasible Region')
    ax.plot(x_constraint, y_constraint, 'r-', linewidth=2, label='x + y = 1')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1)

    # Plot unconstrained minimum
    ax.plot(2, 1, 'ko', markersize=10, label='Unconstrained Min (2, 1)')

    # Plot constrained minimum
    ax.plot(result.x[0], result.x[1], 'r*', markersize=20,
            label=f'Constrained Min ({result.x[0]:.2f}, {result.x[1]:.2f})')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('KKT Conditions: Inequality Constraints',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)

    plt.tight_layout()
    plt.savefig('kkt_conditions.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to 'kkt_conditions.png'")


def convex_vs_nonconvex():
    """Compare optimization on convex vs non-convex functions."""
    print("\n" + "="*60)
    print("4. Convex vs Non-Convex Optimization")
    print("="*60)

    # Convex function: f(x,y) = x^2 + y^2
    def convex_func(x):
        return x[0]**2 + x[1]**2

    # Non-convex function: Himmelblau's function
    # f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    # Has 4 local minima, all with same value
    def nonconvex_func(x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    # Test multiple starting points
    starting_points = [
        np.array([0.0, 0.0]),
        np.array([2.0, 2.0]),
        np.array([-2.0, 2.0]),
        np.array([2.0, -2.0])
    ]

    print("\n--- Convex Function: f(x,y) = x^2 + y^2 ---")
    print("Expected: All starting points converge to (0, 0)")

    for i, x0 in enumerate(starting_points):
        result = minimize(convex_func, x0, method='BFGS')
        print(f"Start {i+1} {x0}: -> ({result.x[0]:.4f}, {result.x[1]:.4f}), "
              f"f = {result.fun:.4f}")

    print("\n--- Non-Convex Function: Himmelblau's Function ---")
    print("Expected: Different starting points may converge to different minima")
    print("Known minima: (3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)")

    for i, x0 in enumerate(starting_points):
        result = minimize(nonconvex_func, x0, method='BFGS')
        print(f"Start {i+1} {x0}: -> ({result.x[0]:.4f}, {result.x[1]:.4f}), "
              f"f = {result.fun:.4f}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    x = np.linspace(-5, 5, 300)
    y = np.linspace(-5, 5, 300)
    X, Y = np.meshgrid(x, y)

    # Convex function
    Z1 = X**2 + Y**2
    contour1 = ax1.contour(X, Y, Z1, levels=20, cmap='viridis')
    ax1.clabel(contour1, inline=True, fontsize=8)
    ax1.plot(0, 0, 'r*', markersize=20, label='Global Minimum')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Convex Function: x² + y²', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Non-convex function
    Z2 = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2
    contour2 = ax2.contour(X, Y, Z2, levels=30, cmap='viridis')
    ax2.clabel(contour2, inline=True, fontsize=8)

    # Plot all four minima
    minima = [(3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)]
    for xm, ym in minima:
        ax2.plot(xm, ym, 'r*', markersize=15)

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title("Non-Convex: Himmelblau's Function", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convex_vs_nonconvex.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison plot to 'convex_vs_nonconvex.png'")


if __name__ == "__main__":
    print("="*60)
    print("Constrained and Unconstrained Optimization Examples")
    print("="*60)

    # Run demonstrations
    unconstrained_optimization_demo()
    lagrange_multipliers_demo()
    kkt_conditions_demo()
    convex_vs_nonconvex()

    print("\n" + "="*60)
    print("Key Takeaways:")
    print("="*60)
    print("1. Unconstrained: Use gradient-based methods (BFGS, CG, L-BFGS)")
    print("2. Equality constraints: Lagrange multipliers convert to unconstrained")
    print("3. Inequality constraints: KKT conditions generalize Lagrange multipliers")
    print("4. Convex problems: Guarantee global optimum, efficient algorithms")
    print("5. Non-convex problems: May have multiple local minima, need global search")
