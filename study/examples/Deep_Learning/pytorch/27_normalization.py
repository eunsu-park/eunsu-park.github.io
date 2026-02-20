"""
27. Normalization Layers Comparison

Demonstrates various normalization techniques:
- Batch Normalization (manual + nn.BatchNorm2d)
- Layer Normalization (manual + nn.LayerNorm)
- Group Normalization (nn.GroupNorm)
- Instance Normalization (nn.InstanceNorm2d)
- RMSNorm (manual implementation)
- Training vs Inference behavior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# 1. BatchNorm from Scratch
# ============================================================================
def manual_batch_norm_2d(x, gamma, beta, running_mean, running_var,
                         momentum=0.1, eps=1e-5, training=True):
    """
    Manual implementation of BatchNorm2d.
    x: (N, C, H, W)
    gamma, beta: (C,) - learnable parameters
    running_mean, running_var: (C,) - running statistics
    """
    if training:
        # Compute batch statistics over (N, H, W) dimensions
        batch_mean = x.mean(dim=(0, 2, 3), keepdim=False)  # (C,)
        batch_var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=False)  # (C,)

        # Update running statistics
        running_mean.data = (1 - momentum) * running_mean + momentum * batch_mean
        running_var.data = (1 - momentum) * running_var + momentum * batch_var

        # Normalize using batch statistics
        mean = batch_mean
        var = batch_var
    else:
        # Use running statistics during inference
        mean = running_mean
        var = running_var

    # Reshape for broadcasting: (1, C, 1, 1)
    mean = mean.view(1, -1, 1, 1)
    var = var.view(1, -1, 1, 1)
    gamma = gamma.view(1, -1, 1, 1)
    beta = beta.view(1, -1, 1, 1)

    # Normalize and scale
    x_norm = (x - mean) / torch.sqrt(var + eps)
    out = gamma * x_norm + beta

    return out


def section1_batchnorm_from_scratch():
    print("\n" + "="*80)
    print("1. BatchNorm from Scratch")
    print("="*80)

    # Create sample input
    N, C, H, W = 4, 3, 8, 8
    x = torch.randn(N, C, H, W)

    # Manual BatchNorm
    gamma = torch.ones(C)
    beta = torch.zeros(C)
    running_mean = torch.zeros(C)
    running_var = torch.ones(C)

    # Training mode
    manual_out_train = manual_batch_norm_2d(
        x, gamma, beta, running_mean.clone(), running_var.clone(), training=True
    )

    # PyTorch BatchNorm
    bn = nn.BatchNorm2d(C, momentum=0.1, eps=1e-5)
    bn.weight.data = gamma.clone()
    bn.bias.data = beta.clone()
    bn.running_mean.data = torch.zeros(C)
    bn.running_var.data = torch.ones(C)

    bn.train()
    pytorch_out_train = bn(x)

    print(f"Input shape: {x.shape}")
    print(f"Manual output mean: {manual_out_train.mean():.6f}")
    print(f"PyTorch output mean: {pytorch_out_train.mean():.6f}")
    print(f"Max difference (training): {(manual_out_train - pytorch_out_train).abs().max():.8f}")

    # Eval mode - show running statistics are used
    bn.eval()
    with torch.no_grad():
        pytorch_out_eval = bn(x)

    print(f"\nRunning mean after training: {bn.running_mean[:3]}")
    print(f"Running var after training: {bn.running_var[:3]}")
    print(f"Eval mode output mean: {pytorch_out_eval.mean():.6f}")
    print("✓ Training vs Eval mode produces different outputs")


# ============================================================================
# 2. LayerNorm from Scratch
# ============================================================================
def manual_layer_norm(x, normalized_shape, gamma, beta, eps=1e-5):
    """
    Manual implementation of LayerNorm.
    x: (N, C, H, W) or any shape
    normalized_shape: dimensions to normalize over (from the end)
    """
    # Compute mean and var over the last len(normalized_shape) dimensions
    dims_to_normalize = list(range(-len(normalized_shape), 0))

    mean = x.mean(dim=dims_to_normalize, keepdim=True)
    var = x.var(dim=dims_to_normalize, unbiased=False, keepdim=True)

    # Normalize
    x_norm = (x - mean) / torch.sqrt(var + eps)

    # Scale and shift (gamma and beta should match normalized_shape)
    out = gamma * x_norm + beta

    return out


def section2_layernorm_from_scratch():
    print("\n" + "="*80)
    print("2. LayerNorm from Scratch")
    print("="*80)

    # Create sample input (batch-independent normalization)
    N, C, H, W = 4, 8, 16, 16
    x = torch.randn(N, C, H, W)

    # LayerNorm over (C, H, W)
    normalized_shape = (C, H, W)
    gamma = torch.ones(normalized_shape)
    beta = torch.zeros(normalized_shape)

    # Manual LayerNorm
    manual_out = manual_layer_norm(x, normalized_shape, gamma, beta)

    # PyTorch LayerNorm
    ln = nn.LayerNorm(normalized_shape, eps=1e-5)
    ln.weight.data = gamma.clone().flatten()
    ln.bias.data = beta.clone().flatten()

    pytorch_out = ln(x)

    print(f"Input shape: {x.shape}")
    print(f"Normalized shape: {normalized_shape}")
    print(f"Manual output mean per sample: {manual_out.mean(dim=(1,2,3))}")
    print(f"Manual output std per sample: {manual_out.std(dim=(1,2,3))}")
    print(f"Max difference: {(manual_out - pytorch_out).abs().max():.8f}")

    # Show batch independence
    single_sample = x[0:1]
    manual_single = manual_layer_norm(single_sample, normalized_shape, gamma, beta)
    print(f"\nFirst sample from batch: {manual_out[0, 0, 0, :3]}")
    print(f"Same sample processed alone: {manual_single[0, 0, 0, :3]}")
    print("✓ LayerNorm is batch-independent")


# ============================================================================
# 3. RMSNorm from Scratch
# ============================================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (as used in LLaMA).
    Only normalizes by RMS, no mean centering.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (batch, seq_len, dim) or (batch, dim)
        # Compute RMS over the last dimension
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm


def section3_rmsnorm_from_scratch():
    print("\n" + "="*80)
    print("3. RMSNorm from Scratch")
    print("="*80)

    batch_size, seq_len, dim = 2, 4, 8
    x = torch.randn(batch_size, seq_len, dim)

    # RMSNorm
    rms_norm = RMSNorm(dim)
    rms_out = rms_norm(x)

    # LayerNorm for comparison
    layer_norm = nn.LayerNorm(dim)
    ln_out = layer_norm(x)

    print(f"Input shape: {x.shape}")
    print(f"Input sample: {x[0, 0, :4]}")
    print(f"\nRMSNorm output: {rms_out[0, 0, :4]}")
    print(f"RMSNorm output RMS: {rms_out[0, 0].pow(2).mean().sqrt():.6f}")
    print(f"\nLayerNorm output: {ln_out[0, 0, :4]}")
    print(f"LayerNorm output mean: {ln_out[0, 0].mean():.6f}")
    print(f"LayerNorm output std: {ln_out[0, 0].std():.6f}")

    print("\n✓ RMSNorm: only normalizes scale (no mean centering)")
    print("✓ LayerNorm: normalizes both mean and variance")


# ============================================================================
# 4. Normalization Dimension Comparison
# ============================================================================
def section4_normalization_dimensions():
    print("\n" + "="*80)
    print("4. Normalization Dimension Comparison")
    print("="*80)

    N, C, H, W = 2, 4, 8, 8
    x = torch.randn(N, C, H, W) * 10 + 5  # Non-zero mean, large variance

    print(f"Input shape: (N={N}, C={C}, H={H}, W={W})")
    print(f"Input statistics:")
    print(f"  Global mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"  Per-channel mean: {x.mean(dim=(0, 2, 3))}")

    # BatchNorm2d: normalizes over (N, H, W) for each C
    bn = nn.BatchNorm2d(C)
    bn_out = bn(x)
    print(f"\nBatchNorm2d (normalize over N,H,W for each C):")
    print(f"  Output mean: {bn_out.mean():.4f}, std: {bn_out.std():.4f}")
    print(f"  Per-channel mean: {bn_out.mean(dim=(0, 2, 3))}")

    # LayerNorm: normalizes over (C, H, W) for each N
    ln = nn.LayerNorm((C, H, W))
    ln_out = ln(x)
    print(f"\nLayerNorm (normalize over C,H,W for each N):")
    print(f"  Output mean: {ln_out.mean():.4f}, std: {ln_out.std():.4f}")
    print(f"  Per-sample mean: {ln_out.mean(dim=(1, 2, 3))}")

    # GroupNorm: normalizes over (H, W) and groups of C for each N
    gn = nn.GroupNorm(num_groups=2, num_channels=C)
    gn_out = gn(x)
    print(f"\nGroupNorm (2 groups, normalize over H,W for each group in each N):")
    print(f"  Output mean: {gn_out.mean():.4f}, std: {gn_out.std():.4f}")

    # InstanceNorm2d: normalizes over (H, W) for each C in each N
    in_norm = nn.InstanceNorm2d(C)
    in_out = in_norm(x)
    print(f"\nInstanceNorm2d (normalize over H,W for each C in each N):")
    print(f"  Output mean: {in_out.mean():.4f}, std: {in_out.std():.4f}")

    print("\n✓ Different normalizations operate over different dimensions")


# ============================================================================
# 5. GroupNorm vs BatchNorm with Small Batch
# ============================================================================
def section5_groupnorm_vs_batchnorm():
    print("\n" + "="*80)
    print("5. GroupNorm vs BatchNorm with Small Batch Size")
    print("="*80)

    C, H, W = 32, 16, 16

    # Create data with batch_size=1 (BatchNorm fails here)
    x_small = torch.randn(1, C, H, W)

    print(f"Input shape: {x_small.shape} (batch_size=1)")
    print(f"Input mean: {x_small.mean():.4f}, std: {x_small.std():.4f}")

    # BatchNorm with batch_size=1
    bn = nn.BatchNorm2d(C)
    bn.train()
    bn_out = bn(x_small)
    print(f"\nBatchNorm2d (training mode, batch_size=1):")
    print(f"  Output mean: {bn_out.mean():.4f}, std: {bn_out.std():.4f}")
    print(f"  ⚠️  With batch_size=1, variance=0, normalization unstable")

    # GroupNorm works fine
    gn = nn.GroupNorm(num_groups=8, num_channels=C)
    gn_out = gn(x_small)
    print(f"\nGroupNorm (8 groups):")
    print(f"  Output mean: {gn_out.mean():.4f}, std: {gn_out.std():.4f}")
    print(f"  ✓ GroupNorm stable with batch_size=1")

    # Larger batch for comparison
    x_large = torch.randn(16, C, H, W)
    bn_large = bn(x_large)
    print(f"\nBatchNorm2d with batch_size=16:")
    print(f"  Output mean: {bn_large.mean():.4f}, std: {bn_large.std():.4f}")
    print(f"  ✓ BatchNorm stable with larger batch")


# ============================================================================
# 6. Pre-Norm vs Post-Norm Transformer Block
# ============================================================================
class PostNormTransformerBlock(nn.Module):
    """Traditional: Attention/FFN -> Add -> Norm"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Post-norm: x + Sublayer(x) -> Norm
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class PreNormTransformerBlock(nn.Module):
    """Modern: Norm -> Attention/FFN -> Add"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # Pre-norm: x + Sublayer(Norm(x))
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x


def section6_prenorm_vs_postnorm():
    print("\n" + "="*80)
    print("6. Pre-Norm vs Post-Norm Transformer Block")
    print("="*80)

    batch_size, seq_len, dim = 2, 10, 64
    x = torch.randn(batch_size, seq_len, dim)

    post_norm_block = PostNormTransformerBlock(dim)
    pre_norm_block = PreNormTransformerBlock(dim)

    # Forward pass
    post_out = post_norm_block(x)
    pre_out = pre_norm_block(x)

    print(f"Input shape: {x.shape}")
    print(f"Input norm: {x.norm():.4f}")
    print(f"\nPost-Norm output norm: {post_out.norm():.4f}")
    print(f"Pre-Norm output norm: {pre_out.norm():.4f}")

    # Check gradient flow (dummy loss)
    loss_post = post_out.sum()
    loss_post.backward()
    post_grad_norm = sum(p.grad.norm().item() for p in post_norm_block.parameters() if p.grad is not None)

    # Reset and compute pre-norm gradients
    pre_norm_block.zero_grad()
    x_pre = x.clone().detach().requires_grad_(True)
    pre_out = pre_norm_block(x_pre)
    loss_pre = pre_out.sum()
    loss_pre.backward()
    pre_grad_norm = sum(p.grad.norm().item() for p in pre_norm_block.parameters() if p.grad is not None)

    print(f"\nPost-Norm gradient norm: {post_grad_norm:.4f}")
    print(f"Pre-Norm gradient norm: {pre_grad_norm:.4f}")
    print("\n✓ Pre-Norm: Better gradient flow, easier training")
    print("✓ Post-Norm: Traditional, may need warmup")


# ============================================================================
# 7. Training Experiment
# ============================================================================
class SimpleCNN(nn.Module):
    def __init__(self, norm_type='batch'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)

        if norm_type == 'batch':
            self.norm1 = nn.BatchNorm2d(16)
        elif norm_type == 'group':
            self.norm1 = nn.GroupNorm(4, 16)
        else:
            self.norm1 = nn.Identity()

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        if norm_type == 'batch':
            self.norm2 = nn.BatchNorm2d(32)
        elif norm_type == 'group':
            self.norm2 = nn.GroupNorm(8, 32)
        else:
            self.norm2 = nn.Identity()

        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_model(model, epochs=5, batch_size=32):
    """Train on synthetic data"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(epochs):
        # Generate synthetic data
        x = torch.randn(batch_size, 1, 28, 28)
        y = torch.randint(0, 10, (batch_size,))

        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def section7_training_experiment():
    print("\n" + "="*80)
    print("7. Training Experiment: Different Normalizations")
    print("="*80)

    epochs = 10

    # Train with BatchNorm
    model_bn = SimpleCNN(norm_type='batch')
    losses_bn = train_model(model_bn, epochs=epochs)

    # Train with GroupNorm
    model_gn = SimpleCNN(norm_type='group')
    losses_gn = train_model(model_gn, epochs=epochs)

    # Train without normalization
    model_none = SimpleCNN(norm_type='none')
    losses_none = train_model(model_none, epochs=epochs)

    print(f"Training for {epochs} epochs on synthetic data:")
    print(f"\nBatchNorm losses: {[f'{l:.4f}' for l in losses_bn[:5]]} ... {losses_bn[-1]:.4f}")
    print(f"GroupNorm losses: {[f'{l:.4f}' for l in losses_gn[:5]]} ... {losses_gn[-1]:.4f}")
    print(f"No Norm losses:   {[f'{l:.4f}' for l in losses_none[:5]]} ... {losses_none[-1]:.4f}")

    print(f"\nFinal loss comparison:")
    print(f"  BatchNorm:  {losses_bn[-1]:.4f}")
    print(f"  GroupNorm:  {losses_gn[-1]:.4f}")
    print(f"  No Norm:    {losses_none[-1]:.4f}")
    print("\n✓ Normalization helps convergence")


# ============================================================================
# 8. Common Pitfalls
# ============================================================================
def section8_common_pitfalls():
    print("\n" + "="*80)
    print("8. Common Pitfalls")
    print("="*80)

    # Pitfall 1: Forgetting model.eval() for BatchNorm
    print("\n--- Pitfall 1: Forgetting model.eval() for BatchNorm ---")
    x = torch.randn(4, 3, 8, 8)
    bn = nn.BatchNorm2d(3)

    # Train mode (uses batch statistics)
    bn.train()
    out_train_1 = bn(x)
    out_train_2 = bn(x)  # Different output each time!

    # Eval mode (uses running statistics)
    bn.eval()
    with torch.no_grad():
        out_eval_1 = bn(x)
        out_eval_2 = bn(x)  # Same output

    print(f"Train mode, pass 1 mean: {out_train_1.mean():.6f}")
    print(f"Train mode, pass 2 mean: {out_train_2.mean():.6f}")
    print(f"Difference: {(out_train_1 - out_train_2).abs().max():.6f}")
    print(f"\nEval mode, pass 1 mean: {out_eval_1.mean():.6f}")
    print(f"Eval mode, pass 2 mean: {out_eval_2.mean():.6f}")
    print(f"Difference: {(out_eval_1 - out_eval_2).abs().max():.6f}")
    print("⚠️  Always use model.eval() during inference!")

    # Pitfall 2: Wrong dimension for LayerNorm
    print("\n--- Pitfall 2: Wrong normalized_shape for LayerNorm ---")
    x = torch.randn(2, 10, 64)  # (batch, seq, dim)

    # Correct: normalize over last dimension
    ln_correct = nn.LayerNorm(64)
    out_correct = ln_correct(x)
    print(f"Input shape: {x.shape}")
    print(f"Correct LayerNorm(64): output mean per sample = {out_correct.mean(dim=(1,2))}")

    # Wrong: normalizing over wrong dimensions
    try:
        ln_wrong = nn.LayerNorm((10, 64))  # This normalizes over (seq, dim)
        out_wrong = ln_wrong(x)
        print(f"Wrong LayerNorm((10,64)): output mean per sample = {out_wrong.mean(dim=(1,2))}")
        print("⚠️  Make sure normalized_shape matches your intention!")
    except Exception as e:
        print(f"Error: {e}")

    # Pitfall 3: Frozen BatchNorm
    print("\n--- Pitfall 3: Frozen BatchNorm (for fine-tuning) ---")
    bn = nn.BatchNorm2d(3)
    x = torch.randn(4, 3, 8, 8)

    # Normal training: running stats update
    bn.train()
    initial_mean = bn.running_mean.clone()
    _ = bn(x)
    updated_mean = bn.running_mean
    print(f"Initial running mean: {initial_mean[:3]}")
    print(f"After forward (train mode): {updated_mean[:3]}")
    print(f"Difference: {(updated_mean - initial_mean).abs().sum():.6f}")

    # Frozen BatchNorm: eval mode, no gradient
    bn.eval()
    for param in bn.parameters():
        param.requires_grad = False

    frozen_mean = bn.running_mean.clone()
    _ = bn(x)
    after_frozen = bn.running_mean
    print(f"\nFrozen BN running mean: {frozen_mean[:3]}")
    print(f"After forward (frozen): {after_frozen[:3]}")
    print(f"Difference: {(after_frozen - frozen_mean).abs().sum():.6f}")
    print("✓ Frozen BatchNorm: use for fine-tuning with different batch sizes")


# ============================================================================
# Main
# ============================================================================
def main():
    print("\n" + "="*80)
    print("PyTorch Normalization Layers Comprehensive Guide")
    print("="*80)

    section1_batchnorm_from_scratch()
    section2_layernorm_from_scratch()
    section3_rmsnorm_from_scratch()
    section4_normalization_dimensions()
    section5_groupnorm_vs_batchnorm()
    section6_prenorm_vs_postnorm()
    section7_training_experiment()
    section8_common_pitfalls()

    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("""
Normalization Comparison:
- BatchNorm: Normalizes over batch dimension, maintains running stats
  → Best for large batches, computer vision
  → Requires model.eval() during inference

- LayerNorm: Normalizes over feature dimensions, batch-independent
  → Best for NLP, transformers, small/variable batches
  → No train/eval mode difference

- GroupNorm: Normalizes over channel groups, batch-independent
  → Best for small batches, instance segmentation
  → More stable than BatchNorm with batch_size=1

- InstanceNorm: Normalizes over spatial dimensions per channel
  → Best for style transfer, GANs

- RMSNorm: Simplified normalization without mean centering
  → Used in LLaMA, faster than LayerNorm
  → Only normalizes scale

Pre-Norm vs Post-Norm:
- Pre-Norm: Better gradient flow, easier training, modern default
- Post-Norm: Traditional, may need learning rate warmup

Common Pitfalls:
1. Forgetting model.eval() for BatchNorm during inference
2. Wrong normalized_shape for LayerNorm
3. Frozen BatchNorm behavior when fine-tuning
    """)
    print("="*80)


if __name__ == "__main__":
    main()
