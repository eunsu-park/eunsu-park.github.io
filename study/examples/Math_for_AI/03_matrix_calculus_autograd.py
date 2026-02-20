"""
Matrix Calculus and Automatic Differentiation

Demonstrates:
- Manual gradient computation for simple functions
- Jacobian and Hessian computation with NumPy
- PyTorch autograd for comparison
- Numerical gradient checking
- ML application: computing gradients of MSE loss

Dependencies: numpy, torch, matplotlib
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


def manual_gradients():
    """Compute gradients manually for simple functions"""
    print("=" * 60)
    print("MANUAL GRADIENT COMPUTATION")
    print("=" * 60)

    # f(x) = x^2
    print("\n--- f(x) = x^2 ---")
    x = 3.0
    f_x = x**2
    df_dx = 2*x  # Derivative: f'(x) = 2x

    print(f"x = {x}")
    print(f"f(x) = {f_x}")
    print(f"f'(x) = {df_dx}")

    # Multivariate: f(x, y) = x^2 + 2xy + y^2
    print("\n--- f(x, y) = x^2 + 2xy + y^2 ---")
    x, y = 2.0, 3.0
    f_xy = x**2 + 2*x*y + y**2

    # Partial derivatives
    df_dx = 2*x + 2*y  # ∂f/∂x = 2x + 2y
    df_dy = 2*x + 2*y  # ∂f/∂y = 2x + 2y

    print(f"x = {x}, y = {y}")
    print(f"f(x, y) = {f_xy}")
    print(f"∂f/∂x = {df_dx}")
    print(f"∂f/∂y = {df_dy}")
    print(f"Gradient: ∇f = [{df_dx}, {df_dy}]")

    # Vector function: f(x) = ||x||^2 = x^T x
    print("\n--- f(x) = ||x||^2 = x^T x ---")
    x = np.array([1.0, 2.0, 3.0])
    f_x = np.dot(x, x)
    grad_f = 2 * x  # Gradient: ∇f = 2x

    print(f"x = {x}")
    print(f"f(x) = {f_x}")
    print(f"∇f = {grad_f}")


def compute_jacobian():
    """Compute Jacobian matrix for vector-valued functions"""
    print("\n" + "=" * 60)
    print("JACOBIAN MATRIX")
    print("=" * 60)

    print("\nJacobian: For f: R^n → R^m, J is m×n matrix of partial derivatives")
    print("J[i,j] = ∂f_i/∂x_j")

    # Example: f(x, y) = [x^2 + y, xy, y^2]
    # Input: R^2, Output: R^3
    print("\n--- Example: f(x, y) = [x^2 + y, xy, y^2] ---")

    def f(x, y):
        return np.array([
            x**2 + y,
            x * y,
            y**2
        ])

    # Analytical Jacobian
    def jacobian_analytical(x, y):
        return np.array([
            [2*x, 1],      # ∂f1/∂x, ∂f1/∂y
            [y, x],        # ∂f2/∂x, ∂f2/∂y
            [0, 2*y]       # ∂f3/∂x, ∂f3/∂y
        ])

    x, y = 2.0, 3.0
    f_val = f(x, y)
    J = jacobian_analytical(x, y)

    print(f"\nAt point (x, y) = ({x}, {y})")
    print(f"f(x, y) = {f_val}")
    print(f"\nJacobian (3×2):\n{J}")

    # Numerical Jacobian (finite differences)
    def numerical_jacobian(func, x, y, h=1e-7):
        f0 = func(x, y)
        m = len(f0)  # output dimension

        J_num = np.zeros((m, 2))

        # ∂f/∂x
        f_plus_x = func(x + h, y)
        J_num[:, 0] = (f_plus_x - f0) / h

        # ∂f/∂y
        f_plus_y = func(x, y + h)
        J_num[:, 1] = (f_plus_y - f0) / h

        return J_num

    J_num = numerical_jacobian(f, x, y)
    print(f"\nNumerical Jacobian:\n{J_num}")
    print(f"Max difference: {np.max(np.abs(J - J_num)):.10f}")


def compute_hessian():
    """Compute Hessian matrix (second derivatives)"""
    print("\n" + "=" * 60)
    print("HESSIAN MATRIX")
    print("=" * 60)

    print("\nHessian: For f: R^n → R, H is n×n matrix of second partial derivatives")
    print("H[i,j] = ∂²f/(∂x_i ∂x_j)")

    # Example: f(x, y) = x^2 + 2xy + 3y^2
    print("\n--- Example: f(x, y) = x^2 + 2xy + 3y^2 ---")

    def f(x, y):
        return x**2 + 2*x*y + 3*y**2

    # Analytical Hessian
    def hessian_analytical(x, y):
        return np.array([
            [2, 2],   # ∂²f/∂x², ∂²f/∂x∂y
            [2, 6]    # ∂²f/∂y∂x, ∂²f/∂y²
        ])

    x, y = 1.0, 2.0
    f_val = f(x, y)
    H = hessian_analytical(x, y)

    print(f"\nAt point (x, y) = ({x}, {y})")
    print(f"f(x, y) = {f_val}")
    print(f"\nHessian (2×2):\n{H}")

    # Check symmetry (Schwarz's theorem)
    print(f"Symmetric: {np.allclose(H, H.T)}")

    # Eigenvalues for convexity analysis
    eigenvalues = np.linalg.eigvals(H)
    print(f"\nEigenvalues: {eigenvalues}")

    if np.all(eigenvalues > 0):
        print("Hessian is positive definite → f is strictly convex")
    elif np.all(eigenvalues >= 0):
        print("Hessian is positive semidefinite → f is convex")
    else:
        print("Hessian has negative eigenvalues → f is not convex")


def pytorch_autograd():
    """Demonstrate PyTorch automatic differentiation"""
    print("\n" + "=" * 60)
    print("PYTORCH AUTOMATIC DIFFERENTIATION")
    print("=" * 60)

    # Scalar function
    print("\n--- Scalar Function: f(x) = x^2 + 2x + 1 ---")
    x = torch.tensor(3.0, requires_grad=True)
    f = x**2 + 2*x + 1

    print(f"x = {x.item()}")
    print(f"f(x) = {f.item()}")

    # Compute gradient
    f.backward()
    print(f"df/dx = {x.grad.item()}")
    print(f"Expected: 2*3 + 2 = {2*3 + 2}")

    # Vector function
    print("\n--- Vector Function ---")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    f = torch.sum(x**2)  # f(x) = ||x||^2

    print(f"x = {x.detach().numpy()}")
    print(f"f(x) = {f.item()}")

    f.backward()
    print(f"∇f = {x.grad.numpy()}")
    print(f"Expected: 2*x = {2*x.detach().numpy()}")

    # Matrix operations
    print("\n--- Matrix Operations ---")
    W = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x = torch.tensor([1.0, -1.0])

    # f(W) = ||Wx||^2
    y = W @ x
    f = torch.sum(y**2)

    print(f"W =\n{W.detach().numpy()}")
    print(f"x = {x.numpy()}")
    print(f"y = Wx = {y.detach().numpy()}")
    print(f"f = ||Wx||^2 = {f.item()}")

    f.backward()
    print(f"\n∂f/∂W =\n{W.grad.numpy()}")


def numerical_gradient_checking():
    """Verify gradients using numerical approximation"""
    print("\n" + "=" * 60)
    print("NUMERICAL GRADIENT CHECKING")
    print("=" * 60)

    print("\nFinite difference approximation:")
    print("f'(x) ≈ [f(x + h) - f(x - h)] / (2h)")

    def f(x):
        return np.sum(x**3 - 2*x**2 + x)

    def grad_analytical(x):
        return 3*x**2 - 4*x + 1

    def grad_numerical(x, h=1e-5):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h

            grad[i] = (f(x_plus) - f(x_minus)) / (2*h)
        return grad

    x = np.array([1.0, 2.0, 3.0])

    grad_ana = grad_analytical(x)
    grad_num = grad_numerical(x)

    print(f"\nAt x = {x}")
    print(f"f(x) = {f(x)}")
    print(f"Analytical gradient: {grad_ana}")
    print(f"Numerical gradient:  {grad_num}")
    print(f"Max difference: {np.max(np.abs(grad_ana - grad_num)):.10f}")

    if np.allclose(grad_ana, grad_num, rtol=1e-4):
        print("✓ Gradient check PASSED")
    else:
        print("✗ Gradient check FAILED")


def mse_loss_gradients():
    """Compute gradients of MSE loss for linear regression"""
    print("\n" + "=" * 60)
    print("ML APPLICATION: MSE LOSS GRADIENTS")
    print("=" * 60)

    # Linear regression: y_pred = Wx + b
    # Loss: L = (1/n) Σ (y_pred - y_true)^2

    print("\n--- Manual Computation ---")
    # Data
    X = np.array([[1.0, 2.0],
                  [2.0, 3.0],
                  [3.0, 4.0],
                  [4.0, 5.0]])
    y_true = np.array([3.0, 5.0, 7.0, 9.0])

    # Parameters
    W = np.array([1.0, 0.5])
    b = 0.5

    n = len(y_true)

    # Forward pass
    y_pred = X @ W + b
    loss = np.mean((y_pred - y_true)**2)

    print(f"X shape: {X.shape}")
    print(f"y_true: {y_true}")
    print(f"W: {W}, b: {b}")
    print(f"y_pred: {y_pred}")
    print(f"MSE Loss: {loss:.4f}")

    # Backward pass (gradients)
    # dL/dW = (2/n) X^T (y_pred - y_true)
    # dL/db = (2/n) Σ (y_pred - y_true)

    error = y_pred - y_true
    dL_dW = (2.0 / n) * X.T @ error
    dL_db = (2.0 / n) * np.sum(error)

    print(f"\nGradients (manual):")
    print(f"dL/dW = {dL_dW}")
    print(f"dL/db = {dL_db:.4f}")

    # PyTorch computation
    print("\n--- PyTorch Autograd ---")
    X_torch = torch.tensor(X, dtype=torch.float32)
    y_true_torch = torch.tensor(y_true, dtype=torch.float32)
    W_torch = torch.tensor(W, dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)

    y_pred_torch = X_torch @ W_torch + b_torch
    loss_torch = torch.mean((y_pred_torch - y_true_torch)**2)

    loss_torch.backward()

    print(f"Loss: {loss_torch.item():.4f}")
    print(f"dL/dW = {W_torch.grad.numpy()}")
    print(f"dL/db = {b_torch.grad.item():.4f}")

    print("\n--- Verification ---")
    print(f"W gradients match: {np.allclose(dL_dW, W_torch.grad.numpy())}")
    print(f"b gradients match: {np.isclose(dL_db, b_torch.grad.item())}")


def visualize_gradient_descent():
    """Visualize gradient descent on a simple function"""
    print("\n" + "=" * 60)
    print("GRADIENT DESCENT VISUALIZATION")
    print("=" * 60)

    # Function: f(x, y) = x^2 + 4y^2 (elliptic paraboloid)
    def f(x, y):
        return x**2 + 4*y**2

    def grad_f(x, y):
        return np.array([2*x, 8*y])

    # Starting point
    x0 = np.array([3.0, 2.0])
    learning_rate = 0.1
    n_iterations = 20

    # Gradient descent
    trajectory = [x0.copy()]
    x = x0.copy()

    for i in range(n_iterations):
        grad = grad_f(x[0], x[1])
        x = x - learning_rate * grad
        trajectory.append(x.copy())

    trajectory = np.array(trajectory)

    print(f"Starting point: {x0}")
    print(f"Final point: {trajectory[-1]}")
    print(f"Final loss: {f(trajectory[-1][0], trajectory[-1][1]):.6f}")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create meshgrid for contour plot
    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    # Contour plot
    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)

    # Gradient descent trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', linewidth=2,
            markersize=6, label='Gradient Descent', alpha=0.7)
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12,
            label='Start', zorder=5)
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15,
            label='End', zorder=5)

    # Arrows for gradients
    for i in range(0, len(trajectory)-1, 3):
        dx = trajectory[i+1, 0] - trajectory[i, 0]
        dy = trajectory[i+1, 1] - trajectory[i, 1]
        ax.arrow(trajectory[i, 0], trajectory[i, 1], dx, dy,
                head_width=0.15, head_length=0.1, fc='red', ec='red', alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Gradient Descent on f(x,y) = x² + 4y²')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Math_for_AI/gradient_descent.png', dpi=150)
    print("\nVisualization saved to gradient_descent.png")
    plt.close()


if __name__ == "__main__":
    # Run all demonstrations
    manual_gradients()
    compute_jacobian()
    compute_hessian()
    pytorch_autograd()
    numerical_gradient_checking()
    mse_loss_gradients()
    visualize_gradient_descent()

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
