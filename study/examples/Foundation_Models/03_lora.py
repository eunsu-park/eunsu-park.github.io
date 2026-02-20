"""
Foundation Models - LoRA (Low-Rank Adaptation)

Implements LoRA from scratch using PyTorch.
Demonstrates parameter-efficient fine-tuning through low-rank decomposition.
Compares full fine-tuning vs LoRA in terms of trainable parameters and performance.

Requires: PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer.

    Instead of updating W (d_out × d_in), we add ΔW = BA where:
    - B: d_out × r
    - A: r × d_in
    - r << min(d_out, d_in)

    Forward: h = Wx + BAx = Wx + ΔWx
    """

    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        """
        Initialize LoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of decomposition (r)
            alpha: Scaling factor (typically 1.0 or rank)
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # Original pretrained weight (frozen)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # LoRA matrices (trainable)
        # A: Gaussian initialization
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / np.sqrt(rank))

        # B: Zero initialization (starts with identity)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Scaling factor
        self.scaling = alpha / rank

    def forward(self, x):
        """
        Forward pass: h = Wx + (BA)x

        Args:
            x: Input tensor (batch_size, in_features)

        Returns:
            Output tensor (batch_size, out_features)
        """
        # Original forward pass (frozen)
        h = F.linear(x, self.weight)

        # LoRA adaptation: x → A → B
        lora_out = F.linear(x, self.lora_A)  # (batch, rank)
        lora_out = F.linear(lora_out, self.lora_B)  # (batch, out_features)

        # Scale and add
        return h + lora_out * self.scaling

    def merge_weights(self):
        """Merge LoRA weights into original weight for inference."""
        with torch.no_grad():
            # W_merged = W + α/r * BA
            delta_w = self.lora_B @ self.lora_A * self.scaling
            self.weight.data += delta_w

            # Zero out LoRA to avoid double counting
            self.lora_A.zero_()
            self.lora_B.zero_()

    def get_num_params(self):
        """Get parameter counts."""
        total = self.weight.numel()
        lora_params = self.lora_A.numel() + self.lora_B.numel()
        trainable = lora_params

        return {
            'total': total,
            'trainable': trainable,
            'lora': lora_params,
            'frozen': total,
        }


class SimpleLinearModel(nn.Module):
    """Simple baseline model with standard Linear layers."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LoRAModel(nn.Module):
    """Model with LoRA layers for parameter-efficient fine-tuning."""

    def __init__(self, input_dim, hidden_dim, output_dim, rank=4):
        super().__init__()
        self.fc1 = LoRALayer(input_dim, hidden_dim, rank=rank)
        self.fc2 = LoRALayer(hidden_dim, hidden_dim, rank=rank)
        self.fc3 = LoRALayer(hidden_dim, output_dim, rank=rank)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_num_params(self):
        """Get total and trainable parameter counts."""
        stats = {
            'fc1': self.fc1.get_num_params(),
            'fc2': self.fc2.get_num_params(),
            'fc3': self.fc3.get_num_params(),
        }

        total_params = sum(s['total'] for s in stats.values())
        trainable_params = sum(s['trainable'] for s in stats.values())
        lora_params = sum(s['lora'] for s in stats.values())

        return {
            'total': total_params,
            'trainable': trainable_params,
            'lora': lora_params,
            'layers': stats,
        }


# ============================================================
# Demonstrations
# ============================================================

def demo_parameter_efficiency():
    """Compare parameter counts: full vs LoRA."""
    print("=" * 60)
    print("DEMO 1: Parameter Efficiency")
    print("=" * 60)

    input_dim = 512
    hidden_dim = 2048
    output_dim = 128

    # Standard model
    standard_model = SimpleLinearModel(input_dim, hidden_dim, output_dim)
    total_standard = sum(p.numel() for p in standard_model.parameters())

    print(f"\nModel architecture: {input_dim} → {hidden_dim} → {hidden_dim} → {output_dim}")
    print(f"\nStandard model (full fine-tuning):")
    print(f"  Total parameters: {total_standard:,}")
    print(f"  Trainable parameters: {total_standard:,}")

    # LoRA models with different ranks
    print(f"\nLoRA models:")
    print("-" * 60)

    for rank in [2, 4, 8, 16, 32]:
        lora_model = LoRAModel(input_dim, hidden_dim, output_dim, rank=rank)
        stats = lora_model.get_num_params()

        reduction = (1 - stats['trainable'] / total_standard) * 100

        print(f"\nRank {rank}:")
        print(f"  Total parameters: {stats['total']:,}")
        print(f"  Trainable parameters: {stats['trainable']:,}")
        print(f"  LoRA parameters: {stats['lora']:,}")
        print(f"  Parameter reduction: {reduction:.2f}%")
        print(f"  Compression ratio: {total_standard/stats['trainable']:.1f}x")


def demo_lora_layer():
    """Demonstrate single LoRA layer behavior."""
    print("\n" + "=" * 60)
    print("DEMO 2: LoRA Layer Mechanics")
    print("=" * 60)

    # Small example
    in_dim = 8
    out_dim = 4
    rank = 2

    layer = LoRALayer(in_dim, out_dim, rank=rank, alpha=1.0)

    print(f"\nLayer: {in_dim} → {out_dim}, rank={rank}")
    print(f"\nWeight matrix W: {layer.weight.shape}")
    print(f"LoRA matrix A: {layer.lora_A.shape}")
    print(f"LoRA matrix B: {layer.lora_B.shape}")

    # Check parameter counts
    stats = layer.get_num_params()
    print(f"\nParameter counts:")
    print(f"  Original weight W: {stats['frozen']}")
    print(f"  LoRA matrices (A + B): {stats['lora']}")
    print(f"  Reduction: {(1 - stats['lora']/stats['frozen']) * 100:.1f}%")

    # Forward pass
    batch_size = 3
    x = torch.randn(batch_size, in_dim)

    with torch.no_grad():
        output = layer(x)

    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")


def demo_training():
    """Demonstrate training with LoRA."""
    print("\n" + "=" * 60)
    print("DEMO 3: Training Comparison")
    print("=" * 60)

    # Toy regression task
    input_dim = 64
    hidden_dim = 256
    output_dim = 10
    num_samples = 1000

    # Generate synthetic data
    X = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, output_dim)

    # Standard model
    standard_model = SimpleLinearModel(input_dim, hidden_dim, output_dim)
    optimizer_std = torch.optim.Adam(standard_model.parameters(), lr=0.001)

    # LoRA model
    lora_model = LoRAModel(input_dim, hidden_dim, output_dim, rank=8)
    # Only optimize LoRA parameters
    lora_params = [p for p in lora_model.parameters() if p.requires_grad]
    optimizer_lora = torch.optim.Adam(lora_params, lr=0.001)

    print(f"\nTraining on {num_samples} samples...")

    # Training loop
    num_epochs = 50
    batch_size = 32

    std_losses = []
    lora_losses = []

    for epoch in range(num_epochs):
        # Standard model
        idx = torch.randperm(num_samples)[:batch_size]
        X_batch, y_batch = X[idx], y[idx]

        optimizer_std.zero_grad()
        pred = standard_model(X_batch)
        loss = F.mse_loss(pred, y_batch)
        loss.backward()
        optimizer_std.step()
        std_losses.append(loss.item())

        # LoRA model
        optimizer_lora.zero_grad()
        pred = lora_model(X_batch)
        loss = F.mse_loss(pred, y_batch)
        loss.backward()
        optimizer_lora.step()
        lora_losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: "
                  f"Standard loss = {std_losses[-1]:.4f}, "
                  f"LoRA loss = {lora_losses[-1]:.4f}")

    # Final evaluation
    with torch.no_grad():
        std_pred = standard_model(X)
        lora_pred = lora_model(X)

        std_final_loss = F.mse_loss(std_pred, y).item()
        lora_final_loss = F.mse_loss(lora_pred, y).item()

    print(f"\nFinal test loss:")
    print(f"  Standard model: {std_final_loss:.4f}")
    print(f"  LoRA model: {lora_final_loss:.4f}")


def demo_rank_impact():
    """Study impact of LoRA rank on capacity."""
    print("\n" + "=" * 60)
    print("DEMO 4: Impact of Rank")
    print("=" * 60)

    input_dim = 128
    hidden_dim = 512
    output_dim = 16

    # Training data
    num_samples = 500
    X = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, output_dim)

    print(f"\nTraining models with different ranks...")
    print("-" * 60)

    ranks = [1, 2, 4, 8, 16, 32]
    results = []

    for rank in ranks:
        model = LoRAModel(input_dim, hidden_dim, output_dim, rank=rank)
        lora_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(lora_params, lr=0.001)

        # Quick training
        for _ in range(100):
            idx = torch.randperm(num_samples)[:64]
            optimizer.zero_grad()
            pred = model(X[idx])
            loss = F.mse_loss(pred, y[idx])
            loss.backward()
            optimizer.step()

        # Evaluate
        with torch.no_grad():
            pred = model(X)
            final_loss = F.mse_loss(pred, y).item()

        stats = model.get_num_params()
        results.append((rank, stats['trainable'], final_loss))

    # Print results
    print(f"\n{'Rank':<8} {'Params':<12} {'Loss':<10}")
    print("-" * 60)
    for rank, params, loss in results:
        print(f"{rank:<8} {params:<12,} {loss:<10.4f}")


def demo_weight_merging():
    """Demonstrate merging LoRA weights for inference."""
    print("\n" + "=" * 60)
    print("DEMO 5: Weight Merging")
    print("=" * 60)

    # Create layer
    layer = LoRALayer(512, 256, rank=8)

    # Random input
    x = torch.randn(4, 512)

    # Output before merging
    with torch.no_grad():
        output_before = layer(x)

    print(f"\nBefore merging:")
    print(f"  Weight norm: {layer.weight.norm().item():.4f}")
    print(f"  LoRA A norm: {layer.lora_A.norm().item():.4f}")
    print(f"  LoRA B norm: {layer.lora_B.norm().item():.4f}")
    print(f"  Output sample: {output_before[0, :5]}")

    # Merge weights
    layer.merge_weights()

    # Output after merging
    with torch.no_grad():
        output_after = layer(x)

    print(f"\nAfter merging:")
    print(f"  Weight norm: {layer.weight.norm().item():.4f}")
    print(f"  LoRA A norm: {layer.lora_A.norm().item():.4f}")
    print(f"  LoRA B norm: {layer.lora_B.norm().item():.4f}")
    print(f"  Output sample: {output_after[0, :5]}")

    # Check equivalence
    diff = (output_before - output_after).abs().max().item()
    print(f"\nMax difference: {diff:.6e}")
    print(f"Outputs are equivalent: {diff < 1e-5}")


def demo_adapter_composition():
    """Demonstrate composing multiple LoRA adapters."""
    print("\n" + "=" * 60)
    print("DEMO 6: Adapter Composition")
    print("=" * 60)

    print("\nScenario: Fine-tune same base model for different tasks")

    input_dim = 256
    output_dim = 128

    # Shared base weight
    base_weight = torch.randn(output_dim, input_dim)

    # Create adapters for different tasks
    adapters = {}
    for task in ['task_A', 'task_B', 'task_C']:
        layer = LoRALayer(input_dim, output_dim, rank=4)
        layer.weight.data = base_weight.clone()  # Share base
        adapters[task] = layer

    # Test input
    x = torch.randn(1, input_dim)

    print(f"\nBase model:")
    with torch.no_grad():
        base_output = F.linear(x, base_weight)
        print(f"  Output norm: {base_output.norm().item():.4f}")

    print(f"\nWith task-specific adapters:")
    for task, adapter in adapters.items():
        with torch.no_grad():
            output = adapter(x)
            diff = (output - base_output).norm().item()
            print(f"  {task}: output norm = {output.norm().item():.4f}, "
                  f"delta = {diff:.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Foundation Models: LoRA (Low-Rank Adaptation)")
    print("=" * 60)

    demo_parameter_efficiency()
    demo_lora_layer()
    demo_training()
    demo_rank_impact()
    demo_weight_merging()
    demo_adapter_composition()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. LoRA decomposes weight updates as ΔW = BA (low-rank)")
    print("2. Reduces trainable params by 100-1000x vs full fine-tuning")
    print("3. Rank r controls capacity/efficiency tradeoff")
    print("4. Can merge adapters into base weights for inference")
    print("5. Enables multi-task learning with shared base model")
    print("=" * 60)
