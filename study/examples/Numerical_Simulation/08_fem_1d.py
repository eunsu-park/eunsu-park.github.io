#!/usr/bin/env python3
"""
1D Finite Element Method (FEM)
===============================

This module demonstrates the Finite Element Method for solving 1D boundary
value problems using piecewise linear (hat) basis functions.

Problem:
    -u''(x) = f(x),  x in [0, 1]
    u(0) = 0,  u(1) = 0  (Dirichlet boundary conditions)

Method:
    - Discretize domain into N elements
    - Use piecewise linear hat basis functions
    - Assemble global stiffness matrix and load vector
    - Solve linear system Au = b
    - Compare with analytical solution

Key Concepts:
    - Weak formulation and Galerkin method
    - Element-wise assembly
    - Hat basis functions
    - Numerical integration (quadrature)

Author: Educational example for Numerical Simulation
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


class FEM1D:
    """
    1D Finite Element Method solver for -u'' = f with Dirichlet BCs.

    Attributes:
        a (float): Left boundary
        b (float): Right boundary
        N (int): Number of elements
        nodes (ndarray): Node coordinates
        h (float): Element size
    """

    def __init__(self, a, b, N):
        """
        Initialize the FEM solver.

        Args:
            a (float): Left boundary
            b (float): Right boundary
            N (int): Number of elements
        """
        self.a = a
        self.b = b
        self.N = N

        # Generate uniform mesh
        self.nodes = np.linspace(a, b, N + 1)
        self.h = (b - a) / N

        # Number of nodes (unknowns)
        self.n_nodes = N + 1

    def hat_function(self, x, i):
        """
        Piecewise linear hat basis function φ_i(x).

        φ_i(x) = 1 at node i, 0 at all other nodes, linear in between.

        Args:
            x (float or ndarray): Evaluation point(s)
            i (int): Node index

        Returns:
            float or ndarray: Value of φ_i(x)
        """
        x = np.atleast_1d(x)
        phi = np.zeros_like(x)

        # Left support: [x_{i-1}, x_i]
        if i > 0:
            mask = (x >= self.nodes[i-1]) & (x <= self.nodes[i])
            phi[mask] = (x[mask] - self.nodes[i-1]) / self.h

        # Right support: [x_i, x_{i+1}]
        if i < self.N:
            mask = (x >= self.nodes[i]) & (x <= self.nodes[i+1])
            phi[mask] = (self.nodes[i+1] - x[mask]) / self.h

        return phi if len(phi) > 1 else phi[0]

    def hat_derivative(self, x, i):
        """
        Derivative of hat basis function dφ_i/dx.

        φ_i'(x) = 1/h on [x_{i-1}, x_i], -1/h on [x_i, x_{i+1}], 0 elsewhere.

        Args:
            x (float or ndarray): Evaluation point(s)
            i (int): Node index

        Returns:
            float or ndarray: Value of dφ_i/dx
        """
        x = np.atleast_1d(x)
        dphi = np.zeros_like(x)

        # Left support
        if i > 0:
            mask = (x >= self.nodes[i-1]) & (x < self.nodes[i])
            dphi[mask] = 1.0 / self.h

        # Right support
        if i < self.N:
            mask = (x >= self.nodes[i]) & (x < self.nodes[i+1])
            dphi[mask] = -1.0 / self.h

        return dphi if len(dphi) > 1 else dphi[0]

    def assemble_element_stiffness(self, e):
        """
        Assemble local stiffness matrix for element e.

        For -u'', the element stiffness matrix is:
            K_e[i,j] = ∫_{x_e}^{x_{e+1}} φ_i' φ_j' dx

        For linear elements, this integral can be computed exactly.

        Args:
            e (int): Element index (0 to N-1)

        Returns:
            ndarray: 2x2 local stiffness matrix
        """
        # For linear hat functions on uniform mesh:
        # K_local = (1/h) * [[1, -1], [-1, 1]]
        K_local = (1.0 / self.h) * np.array([
            [1.0, -1.0],
            [-1.0, 1.0]
        ])
        return K_local

    def assemble_element_load(self, e, f_func):
        """
        Assemble local load vector for element e.

        F_e[i] = ∫_{x_e}^{x_{e+1}} f(x) φ_i(x) dx

        Uses 2-point Gauss quadrature for integration.

        Args:
            e (int): Element index
            f_func (callable): Right-hand side function f(x)

        Returns:
            ndarray: 2x1 local load vector
        """
        # Element boundaries
        x_left = self.nodes[e]
        x_right = self.nodes[e + 1]

        # 2-point Gauss quadrature on reference element [-1, 1]
        # Gauss points and weights
        gauss_points = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        gauss_weights = np.array([1.0, 1.0])

        # Map to physical element [x_left, x_right]
        x_gauss = 0.5 * (x_right - x_left) * gauss_points + 0.5 * (x_right + x_left)
        jacobian = 0.5 * (x_right - x_left)  # dx/dξ

        # Local load vector
        F_local = np.zeros(2)

        # Integrate using quadrature
        for i in range(2):  # Two local nodes
            for q, (xq, wq) in enumerate(zip(x_gauss, gauss_weights)):
                # Hat function value at Gauss point
                # For element e: node 0 is at x_left, node 1 is at x_right
                if i == 0:
                    phi_val = (x_right - xq) / self.h
                else:
                    phi_val = (xq - x_left) / self.h

                F_local[i] += f_func(xq) * phi_val * wq * jacobian

        return F_local

    def assemble_global_system(self, f_func):
        """
        Assemble global stiffness matrix K and load vector F.

        Args:
            f_func (callable): Right-hand side function f(x)

        Returns:
            tuple: (K, F) global stiffness matrix and load vector
        """
        # Initialize global system (use sparse matrix)
        K = lil_matrix((self.n_nodes, self.n_nodes))
        F = np.zeros(self.n_nodes)

        # Loop over elements
        for e in range(self.N):
            # Local stiffness and load
            K_local = self.assemble_element_stiffness(e)
            F_local = self.assemble_element_load(e, f_func)

            # Global node indices for this element
            global_indices = [e, e + 1]

            # Add local contributions to global system
            for i in range(2):
                for j in range(2):
                    K[global_indices[i], global_indices[j]] += K_local[i, j]
                F[global_indices[i]] += F_local[i]

        # Convert to CSR format for efficient solving
        K = csr_matrix(K)

        return K, F

    def apply_boundary_conditions(self, K, F, u_left=0.0, u_right=0.0):
        """
        Apply Dirichlet boundary conditions u(a) = u_left, u(b) = u_right.

        Modify the system to enforce BC by setting diagonal to 1 and RHS to BC value.

        Args:
            K (sparse matrix): Global stiffness matrix
            F (ndarray): Global load vector
            u_left (float): BC at left boundary
            u_right (float): BC at right boundary

        Returns:
            tuple: (K_bc, F_bc) modified system
        """
        K_bc = K.tolil()  # Convert to lil for modification
        F_bc = F.copy()

        # Left boundary (node 0)
        K_bc[0, :] = 0
        K_bc[0, 0] = 1
        F_bc[0] = u_left

        # Right boundary (node N)
        K_bc[self.N, :] = 0
        K_bc[self.N, self.N] = 1
        F_bc[self.N] = u_right

        return K_bc.tocsr(), F_bc

    def solve(self, f_func, u_left=0.0, u_right=0.0):
        """
        Solve the BVP -u'' = f with Dirichlet BCs.

        Args:
            f_func (callable): Right-hand side function f(x)
            u_left (float): BC at left boundary
            u_right (float): BC at right boundary

        Returns:
            ndarray: Solution vector at nodes
        """
        # Assemble global system
        K, F = self.assemble_global_system(f_func)

        # Apply boundary conditions
        K_bc, F_bc = self.apply_boundary_conditions(K, F, u_left, u_right)

        # Solve linear system
        u = spsolve(K_bc, F_bc)

        return u

    def evaluate_solution(self, u_nodes, x_eval):
        """
        Evaluate FEM solution at arbitrary points using basis functions.

        Args:
            u_nodes (ndarray): Solution coefficients at nodes
            x_eval (ndarray): Evaluation points

        Returns:
            ndarray: Solution values at x_eval
        """
        u_eval = np.zeros_like(x_eval)

        # Sum over all basis functions
        for i in range(self.n_nodes):
            u_eval += u_nodes[i] * self.hat_function(x_eval, i)

        return u_eval


def example_1():
    """
    Example 1: -u'' = 2, u(0) = 0, u(1) = 0

    Analytical solution: u(x) = x(1 - x)
    """
    print("=" * 60)
    print("Example 1: -u'' = 2 with homogeneous Dirichlet BCs")
    print("=" * 60)

    # Right-hand side function
    def f(x):
        return 2.0

    # Analytical solution
    def u_exact(x):
        return x * (1 - x)

    # Solve with different mesh refinements
    N_values = [4, 8, 16, 32]
    errors = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Fine grid for plotting exact solution
    x_fine = np.linspace(0, 1, 200)
    ax1.plot(x_fine, u_exact(x_fine), 'k-', linewidth=2, label='Exact')

    for N in N_values:
        # Initialize and solve
        fem = FEM1D(a=0, b=1, N=N)
        u_fem = fem.solve(f)

        # Evaluate at fine grid for plotting
        u_fem_fine = fem.evaluate_solution(u_fem, x_fine)

        # Plot solution
        ax1.plot(x_fine, u_fem_fine, '--', label=f'FEM N={N}', alpha=0.7)
        ax1.plot(fem.nodes, u_fem, 'o', markersize=4)

        # Compute error at nodes
        u_exact_nodes = u_exact(fem.nodes)
        error = np.linalg.norm(u_fem - u_exact_nodes) / np.linalg.norm(u_exact_nodes)
        errors.append(error)
        print(f"N = {N:3d}: Relative L2 error = {error:.6e}")

    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.set_title('FEM Solution vs Exact Solution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Convergence plot
    ax2.loglog(N_values, errors, 'bo-', linewidth=2, markersize=8, label='FEM Error')
    ax2.loglog(N_values, [errors[0] * (N_values[0]/N)**2 for N in N_values],
               'r--', label='$O(h^2)$ reference')
    ax2.set_xlabel('Number of elements N')
    ax2.set_ylabel('Relative L2 Error')
    ax2.set_title('Convergence Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/fem_example1.png', dpi=150)
    print("Saved plot to /tmp/fem_example1.png")
    plt.show()


def example_2():
    """
    Example 2: -u'' = π^2 sin(πx), u(0) = 0, u(1) = 0

    Analytical solution: u(x) = sin(πx)
    """
    print("\n" + "=" * 60)
    print("Example 2: -u'' = π² sin(πx) with homogeneous Dirichlet BCs")
    print("=" * 60)

    # Right-hand side function
    def f(x):
        return np.pi**2 * np.sin(np.pi * x)

    # Analytical solution
    def u_exact(x):
        return np.sin(np.pi * x)

    # Solve
    N = 32
    fem = FEM1D(a=0, b=1, N=N)
    u_fem = fem.solve(f)

    # Evaluation
    x_fine = np.linspace(0, 1, 200)
    u_fem_fine = fem.evaluate_solution(u_fem, x_fine)
    u_exact_fine = u_exact(x_fine)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Solution comparison
    ax1.plot(x_fine, u_exact_fine, 'k-', linewidth=2, label='Exact')
    ax1.plot(x_fine, u_fem_fine, 'b--', linewidth=2, label=f'FEM (N={N})')
    ax1.plot(fem.nodes, u_fem, 'ro', markersize=6, label='FEM nodes')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.set_title('Solution: -u\'\' = π² sin(πx)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Error plot
    error_fine = np.abs(u_fem_fine - u_exact_fine)
    ax2.plot(x_fine, error_fine, 'r-', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('|u_FEM - u_exact|')
    ax2.set_title('Pointwise Absolute Error')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Compute error at nodes
    u_exact_nodes = u_exact(fem.nodes)
    error = np.linalg.norm(u_fem - u_exact_nodes) / np.linalg.norm(u_exact_nodes)
    print(f"N = {N}: Relative L2 error = {error:.6e}")

    plt.tight_layout()
    plt.savefig('/tmp/fem_example2.png', dpi=150)
    print("Saved plot to /tmp/fem_example2.png")
    plt.show()


def example_3_basis_functions():
    """
    Visualize hat basis functions.
    """
    print("\n" + "=" * 60)
    print("Example 3: Visualizing Hat Basis Functions")
    print("=" * 60)

    N = 5
    fem = FEM1D(a=0, b=1, N=N)

    x_plot = np.linspace(0, 1, 500)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each basis function
    for i in range(fem.n_nodes):
        phi_i = fem.hat_function(x_plot, i)
        ax.plot(x_plot, phi_i, linewidth=2, label=f'φ_{i}')
        ax.plot(fem.nodes[i], 1.0, 'ko', markersize=8)

    ax.set_xlabel('x')
    ax.set_ylabel('φ_i(x)')
    ax.set_title(f'Hat Basis Functions (N = {N} elements)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.2])

    plt.tight_layout()
    plt.savefig('/tmp/fem_basis_functions.png', dpi=150)
    print("Saved plot to /tmp/fem_basis_functions.png")
    plt.show()


if __name__ == "__main__":
    print("1D Finite Element Method")
    print("=" * 60)
    print("This script demonstrates FEM for solving -u'' = f with Dirichlet BCs.")
    print()

    # Run examples
    example_3_basis_functions()
    example_1()
    example_2()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - FEM uses piecewise polynomial basis functions (hat functions)")
    print("  - Assembly is done element-by-element (local to global)")
    print("  - Convergence rate is O(h²) for linear elements")
    print("  - Sparse matrices enable efficient solution of large systems")
    print("=" * 60)
