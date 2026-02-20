"""
26. Optimizers Comparison

Demonstrates various optimization algorithms:
- SGD, SGD+Momentum, SGD+Nesterov
- Adam, AdamW
- Manual implementations
- Learning rate schedulers
- Practical optimization patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


# =============================================================================
# 1. SGD from Scratch
# =============================================================================

def rosenbrock(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x)**2 + 100 * (y - x**2)**2


def sgd_step(params: List[torch.Tensor], grads: List[torch.Tensor], lr: float):
    """Basic SGD step."""
    with torch.no_grad():
        for p, g in zip(params, grads):
            p -= lr * g


def sgd_momentum_step(params: List[torch.Tensor], grads: List[torch.Tensor],
                      velocities: List[torch.Tensor], lr: float, momentum: float):
    """SGD with momentum."""
    with torch.no_grad():
        for p, g, v in zip(params, grads, velocities):
            v.mul_(momentum).add_(g)
            p -= lr * v


def sgd_nesterov_step(params: List[torch.Tensor], grads: List[torch.Tensor],
                      velocities: List[torch.Tensor], lr: float, momentum: float):
    """SGD with Nesterov momentum."""
    with torch.no_grad():
        for p, g, v in zip(params, grads, velocities):
            v.mul_(momentum).add_(g)
            p -= lr * (g + momentum * v)


def optimize_rosenbrock():
    """Compare SGD variants on Rosenbrock function."""
    print_section("1. SGD from Scratch - Rosenbrock Optimization")

    # Starting point
    x0, y0 = -1.0, 1.0
    lr = 0.001
    n_steps = 1000

    # Basic SGD
    x, y = torch.tensor([x0], requires_grad=True), torch.tensor([y0], requires_grad=True)
    params = [x, y]

    for step in range(n_steps):
        loss = rosenbrock(x, y)
        loss.backward()

        with torch.no_grad():
            grads = [x.grad.clone(), y.grad.clone()]
            sgd_step(params, grads, lr)
            x.grad.zero_()
            y.grad.zero_()

        if step % 200 == 0:
            print(f"SGD Step {step:4d}: x={x.item():.4f}, y={y.item():.4f}, loss={loss.item():.6f}")

    print(f"SGD Final: x={x.item():.4f}, y={y.item():.4f} (optimal: x=1, y=1)")

    # SGD with momentum
    x, y = torch.tensor([x0], requires_grad=True), torch.tensor([y0], requires_grad=True)
    params = [x, y]
    velocities = [torch.zeros_like(x), torch.zeros_like(y)]
    momentum = 0.9

    for step in range(n_steps):
        loss = rosenbrock(x, y)
        loss.backward()

        with torch.no_grad():
            grads = [x.grad.clone(), y.grad.clone()]
            sgd_momentum_step(params, grads, velocities, lr, momentum)
            x.grad.zero_()
            y.grad.zero_()

    print(f"Momentum Final: x={x.item():.4f}, y={y.item():.4f}")

    # Nesterov
    x, y = torch.tensor([x0], requires_grad=True), torch.tensor([y0], requires_grad=True)
    params = [x, y]
    velocities = [torch.zeros_like(x), torch.zeros_like(y)]

    for step in range(n_steps):
        loss = rosenbrock(x, y)
        loss.backward()

        with torch.no_grad():
            grads = [x.grad.clone(), y.grad.clone()]
            sgd_nesterov_step(params, grads, velocities, lr, momentum)
            x.grad.zero_()
            y.grad.zero_()

    print(f"Nesterov Final: x={x.item():.4f}, y={y.item():.4f}")


# =============================================================================
# 2. Adam from Scratch
# =============================================================================

class ManualAdam:
    """Adam optimizer implemented manually."""

    def __init__(self, params: List[torch.Tensor], lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0

        # Initialize moments
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        """Perform one optimization step."""
        self.step_count += 1

        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                grad = p.grad

                # Update biased first moment
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

                # Update biased second moment
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

                # Bias correction
                m_hat = self.m[i] / (1 - self.beta1**self.step_count)
                v_hat = self.v[i] / (1 - self.beta2**self.step_count)

                # Update parameters
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Zero out gradients."""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


def compare_adam_implementations():
    """Compare manual Adam with PyTorch's Adam."""
    print_section("2. Adam from Scratch")

    # Simple quadratic: f(x) = x^2 + y^2
    x0, y0 = 5.0, 3.0
    n_steps = 100
    lr = 0.1

    # Manual Adam
    x1 = torch.tensor([x0], requires_grad=True)
    y1 = torch.tensor([y0], requires_grad=True)
    manual_adam = ManualAdam([x1, y1], lr=lr)

    # PyTorch Adam
    x2 = torch.tensor([x0], requires_grad=True)
    y2 = torch.tensor([y0], requires_grad=True)
    torch_adam = torch.optim.Adam([x2, y2], lr=lr)

    print(f"Initial: x={x0:.4f}, y={y0:.4f}")

    for step in range(n_steps):
        # Manual
        loss1 = x1**2 + y1**2
        loss1.backward()
        manual_adam.step()
        manual_adam.zero_grad()

        # PyTorch
        loss2 = x2**2 + y2**2
        loss2.backward()
        torch_adam.step()
        torch_adam.zero_grad()

        if step % 20 == 0:
            print(f"Step {step:3d} - Manual: ({x1.item():.4f}, {y1.item():.4f}), "
                  f"PyTorch: ({x2.item():.4f}, {y2.item():.4f})")

    print(f"\nFinal Manual: ({x1.item():.6f}, {y1.item():.6f})")
    print(f"Final PyTorch: ({x2.item():.6f}, {y2.item():.6f})")
    print(f"Difference: ({abs(x1.item()-x2.item()):.2e}, {abs(y1.item()-y2.item()):.2e})")


# =============================================================================
# 3. Optimizer Comparison on Toy Problem
# =============================================================================

class ToyMLP(nn.Module):
    """Small MLP for classification."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def generate_spiral_data(n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate two-class spiral dataset."""
    n_per_class = n_samples // 2

    theta = torch.linspace(0, 4 * np.pi, n_per_class)
    r = torch.linspace(0.5, 1.0, n_per_class)

    # Class 0
    x0 = r * torch.cos(theta)
    y0 = r * torch.sin(theta)
    class0 = torch.stack([x0, y0], dim=1)

    # Class 1 (rotated)
    x1 = r * torch.cos(theta + np.pi)
    y1 = r * torch.sin(theta + np.pi)
    class1 = torch.stack([x1, y1], dim=1)

    X = torch.cat([class0, class1], dim=0)
    y = torch.cat([torch.zeros(n_per_class, dtype=torch.long),
                   torch.ones(n_per_class, dtype=torch.long)])

    # Add noise
    X += 0.1 * torch.randn_like(X)

    # Shuffle
    perm = torch.randperm(n_samples)
    return X[perm], y[perm]


def train_with_optimizer(model: nn.Module, optimizer, X: torch.Tensor,
                        y: torch.Tensor, n_epochs: int = 50) -> List[float]:
    """Train model and return loss history."""
    losses = []
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def compare_optimizers():
    """Compare SGD, Adam, AdamW on toy problem."""
    print_section("3. Optimizer Comparison on Toy Problem")

    X, y = generate_spiral_data(1000)
    print(f"Generated spiral dataset: {X.shape}, {y.shape}")

    n_epochs = 100
    lr = 0.01

    # SGD
    model_sgd = ToyMLP()
    optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=lr)
    losses_sgd = train_with_optimizer(model_sgd, optimizer_sgd, X, y, n_epochs)

    # Adam
    model_adam = ToyMLP()
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=lr)
    losses_adam = train_with_optimizer(model_adam, optimizer_adam, X, y, n_epochs)

    # AdamW
    model_adamw = ToyMLP()
    optimizer_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=lr, weight_decay=0.01)
    losses_adamw = train_with_optimizer(model_adamw, optimizer_adamw, X, y, n_epochs)

    print(f"\nFinal losses after {n_epochs} epochs:")
    print(f"SGD:   {losses_sgd[-1]:.6f}")
    print(f"Adam:  {losses_adam[-1]:.6f}")
    print(f"AdamW: {losses_adamw[-1]:.6f}")

    # Accuracy
    with torch.no_grad():
        acc_sgd = (model_sgd(X).argmax(dim=1) == y).float().mean()
        acc_adam = (model_adam(X).argmax(dim=1) == y).float().mean()
        acc_adamw = (model_adamw(X).argmax(dim=1) == y).float().mean()

    print(f"\nAccuracies:")
    print(f"SGD:   {acc_sgd:.4f}")
    print(f"Adam:  {acc_adam:.4f}")
    print(f"AdamW: {acc_adamw:.4f}")


# =============================================================================
# 4. Weight Decay vs L2 Regularization
# =============================================================================

def demonstrate_weight_decay_vs_l2():
    """Show difference between weight decay and L2 regularization."""
    print_section("4. Weight Decay vs L2 Regularization")

    # Simple linear model
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = X @ torch.randn(10) + 0.1 * torch.randn(100)

    n_steps = 100
    lr = 0.01
    wd = 0.1

    # Adam with L2 regularization (add to loss)
    model_l2 = nn.Linear(10, 1)
    optimizer_l2 = torch.optim.Adam(model_l2.parameters(), lr=lr)

    for _ in range(n_steps):
        optimizer_l2.zero_grad()
        pred = model_l2(X).squeeze()
        loss = F.mse_loss(pred, y)

        # Add L2 penalty to loss
        l2_reg = sum(p.pow(2).sum() for p in model_l2.parameters())
        loss = loss + 0.5 * wd * l2_reg

        loss.backward()
        optimizer_l2.step()

    # AdamW with decoupled weight decay
    model_adamw = nn.Linear(10, 1)
    optimizer_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=lr, weight_decay=wd)

    for _ in range(n_steps):
        optimizer_adamw.zero_grad()
        pred = model_adamw(X).squeeze()
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer_adamw.step()

    # Compare weight norms
    l2_weight_norm = model_l2.weight.norm().item()
    adamw_weight_norm = model_adamw.weight.norm().item()

    print(f"Weight norm with Adam+L2: {l2_weight_norm:.4f}")
    print(f"Weight norm with AdamW:   {adamw_weight_norm:.4f}")
    print(f"\nAdamW typically produces smaller weights due to decoupled decay.")
    print("L2 regularization interacts with adaptive learning rates, weight decay doesn't.")


# =============================================================================
# 5. Learning Rate Schedulers
# =============================================================================

def demonstrate_schedulers():
    """Show different LR schedulers."""
    print_section("5. Learning Rate Schedulers")

    model = nn.Linear(10, 1)

    # StepLR
    print("\n--- StepLR (decay by 0.5 every 30 steps) ---")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    for step in range(100):
        if step % 20 == 0:
            print(f"Step {step:3d}: LR = {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()

    # CosineAnnealingLR
    print("\n--- CosineAnnealingLR (T_max=50) ---")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    for step in range(100):
        if step % 20 == 0:
            print(f"Step {step:3d}: LR = {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()

    # Linear warmup + cosine
    print("\n--- Linear Warmup + Cosine ---")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    warmup_steps = 10
    total_steps = 100

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for step in range(total_steps):
        if step % 20 == 0 or step < 15:
            print(f"Step {step:3d}: LR = {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()

    # OneCycleLR
    print("\n--- OneCycleLR (max_lr=0.1, total_steps=100) ---")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, total_steps=100
    )

    for step in range(100):
        if step % 20 == 0:
            print(f"Step {step:3d}: LR = {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()


# =============================================================================
# 6. Learning Rate Finder
# =============================================================================

def lr_finder(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
              start_lr: float = 1e-7, end_lr: float = 1.0, num_steps: int = 100):
    """Simple LR range test."""
    print_section("6. Learning Rate Finder")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)

    lr_mult = (end_lr / start_lr) ** (1 / num_steps)
    lrs = []
    losses = []

    best_loss = float('inf')

    for step in range(num_steps):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)

        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()

        # Stop if loss explodes
        if loss.item() > 4 * best_loss and step > 10:
            print(f"Stopping early at step {step} (loss exploded)")
            break

        loss.backward()
        optimizer.step()

        # Increase LR
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult

    # Find LR with steepest decrease
    smoothed_losses = []
    window = 5
    for i in range(len(losses)):
        start_idx = max(0, i - window)
        end_idx = min(len(losses), i + window + 1)
        smoothed_losses.append(sum(losses[start_idx:end_idx]) / (end_idx - start_idx))

    min_loss_idx = smoothed_losses.index(min(smoothed_losses))
    suggested_lr = lrs[min_loss_idx]

    print(f"\nTested LR range: {start_lr:.2e} to {lrs[-1]:.2e}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Suggested LR (at min loss): {suggested_lr:.2e}")
    print(f"Suggested LR (1/10 of that): {suggested_lr/10:.2e}")

    # Show some data points
    print("\nSample LR vs Loss:")
    for i in range(0, len(lrs), max(1, len(lrs)//10)):
        print(f"  LR={lrs[i]:.2e}, Loss={losses[i]:.4f}")


def run_lr_finder():
    """Run LR finder on toy problem."""
    X, y = generate_spiral_data(1000)
    model = ToyMLP()
    lr_finder(model, X, y, start_lr=1e-6, end_lr=1.0, num_steps=100)


# =============================================================================
# 7. Per-Parameter Group Learning Rates
# =============================================================================

def demonstrate_param_groups():
    """Show different LRs for different layers."""
    print_section("7. Per-Parameter Group Learning Rates")

    class BackboneHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 32)
            )
            self.head = nn.Linear(32, 2)

        def forward(self, x):
            return self.head(self.backbone(x))

    model = BackboneHead()

    # Different LRs for backbone and head
    optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-4},
        {'params': model.head.parameters(), 'lr': 1e-3}
    ])

    print("Parameter groups:")
    for i, group in enumerate(optimizer.param_groups):
        n_params = sum(p.numel() for p in group['params'])
        print(f"  Group {i}: LR={group['lr']:.2e}, {n_params} parameters")

    # Train a few steps
    X, y = generate_spiral_data(100)
    criterion = nn.CrossEntropyLoss()

    for step in range(5):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        print(f"Step {step}: Loss={loss.item():.4f}")

    print("\nThis pattern is common for transfer learning:")
    print("  - Backbone (pretrained): small LR (fine-tune)")
    print("  - Head (randomly initialized): large LR (learn from scratch)")


# =============================================================================
# 8. Gradient Clipping
# =============================================================================

class ExplodingGradientModel(nn.Module):
    """Model that can have exploding gradients."""

    def __init__(self):
        super().__init__()
        # Large weights can cause gradients to explode
        self.layers = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

        # Initialize with large weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -5, 5)

    def forward(self, x):
        return self.layers(x)


def demonstrate_gradient_clipping():
    """Show gradient clipping techniques."""
    print_section("8. Gradient Clipping")

    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)

    lr = 0.01
    n_steps = 20

    # Without clipping
    print("--- Without gradient clipping ---")
    model = ExplodingGradientModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, y)
        loss.backward()

        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()

        if step % 5 == 0:
            print(f"Step {step:2d}: Loss={loss.item():.4f}, Grad norm={total_norm:.4f}")

    # With clip_grad_norm_
    print("\n--- With clip_grad_norm_ (max_norm=1.0) ---")
    model = ExplodingGradientModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    max_norm = 1.0

    for step in range(n_steps):
        optimizer.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, y)
        loss.backward()

        # Clip gradients
        total_norm_before = sum(
            p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None
        )**0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        total_norm_after = sum(
            p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None
        )**0.5

        optimizer.step()

        if step % 5 == 0:
            print(f"Step {step:2d}: Loss={loss.item():.4f}, "
                  f"Grad before={total_norm_before:.4f}, after={total_norm_after:.4f}")

    # With clip_grad_value_
    print("\n--- With clip_grad_value_ (clip_value=0.5) ---")
    model = ExplodingGradientModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    clip_value = 0.5

    for step in range(n_steps):
        optimizer.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, y)
        loss.backward()

        # Check max gradient value
        max_grad = max(
            p.grad.abs().max().item() for p in model.parameters() if p.grad is not None
        )

        # Clip gradients by value
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)

        max_grad_after = max(
            p.grad.abs().max().item() for p in model.parameters() if p.grad is not None
        )

        optimizer.step()

        if step % 5 == 0:
            print(f"Step {step:2d}: Loss={loss.item():.4f}, "
                  f"Max grad before={max_grad:.4f}, after={max_grad_after:.4f}")

    print("\nGradient clipping is essential for:")
    print("  - RNNs/LSTMs (prevent exploding gradients)")
    print("  - Training with high learning rates")
    print("  - Reinforcement learning (PPO uses clip_grad_norm_)")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  PyTorch Optimizers Demonstration")
    print("="*70)

    optimize_rosenbrock()
    compare_adam_implementations()
    compare_optimizers()
    demonstrate_weight_decay_vs_l2()
    demonstrate_schedulers()
    run_lr_finder()
    demonstrate_param_groups()
    demonstrate_gradient_clipping()

    print("\n" + "="*70)
    print("  Summary")
    print("="*70)
    print("""
Key takeaways:
1. SGD variants: vanilla, momentum, Nesterov (each improves convergence)
2. Adam: adaptive learning rates per parameter (first & second moments)
3. AdamW: decoupled weight decay (better than L2 regularization)
4. Schedulers: StepLR, Cosine, OneCycle, Warmup (critical for training)
5. LR Finder: automated way to find good learning rate
6. Param groups: different LRs for different layers (transfer learning)
7. Gradient clipping: prevent exploding gradients (RNNs, RL)

Practical tips:
- Start with Adam/AdamW for most tasks
- Use SGD+momentum for CNNs if you have time to tune
- Always use a scheduler (cosine or OneCycle work well)
- Clip gradients for RNNs (max_norm=1.0 is common)
- Use param groups for transfer learning (small LR for pretrained layers)
    """)


if __name__ == '__main__':
    main()
