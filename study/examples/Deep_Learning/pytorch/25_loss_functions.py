"""
25. Loss Functions Comparison

Demonstrates various loss functions used in deep learning:
- Regression: MSE, MAE, Huber, Log-Cosh
- Classification: BCE, Cross-Entropy, Focal Loss, Label Smoothing
- Metric Learning: Contrastive, Triplet, InfoNCE
- Segmentation: Dice Loss
- Custom loss implementation patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================================
# 1. Regression Losses: MSE vs MAE vs Huber
# ============================================================================
print_section("1. Regression Losses: Handling Outliers")

# Create synthetic data with outliers
torch.manual_seed(42)
predictions = torch.randn(100)
targets = predictions + torch.randn(100) * 0.1

# Add outliers
outlier_indices = torch.tensor([10, 25, 50, 75])
targets[outlier_indices] += torch.randn(4) * 5.0

print(f"Data shape: {predictions.shape}")
print(f"Outlier indices: {outlier_indices.tolist()}")
print(f"Outlier values: {targets[outlier_indices].tolist()}")

# MSE Loss (L2)
mse_loss = F.mse_loss(predictions, targets)
print(f"\nMSE Loss: {mse_loss.item():.4f}")
print("  → Squares errors, sensitive to outliers")

# MAE Loss (L1)
mae_loss = F.l1_loss(predictions, targets)
print(f"\nMAE Loss: {mae_loss.item():.4f}")
print("  → Absolute errors, robust to outliers")

# Huber Loss (smooth L1)
huber_loss = F.huber_loss(predictions, targets, delta=1.0)
print(f"\nHuber Loss (delta=1.0): {huber_loss.item():.4f}")
print("  → L2 for small errors, L1 for large errors")

# Manual Huber implementation
def huber_loss_manual(pred, target, delta=1.0):
    """Manual Huber loss implementation."""
    error = torch.abs(pred - target)
    quadratic = torch.min(error, torch.tensor(delta))
    linear = error - quadratic
    return torch.mean(0.5 * quadratic**2 + delta * linear)

huber_manual = huber_loss_manual(predictions, targets, delta=1.0)
print(f"Huber Loss (manual): {huber_manual.item():.4f}")

# Log-Cosh Loss
def log_cosh_loss(pred, target):
    """Log-Cosh loss: log(cosh(x)) ≈ smooth L1."""
    error = pred - target
    return torch.mean(torch.log(torch.cosh(error)))

log_cosh = log_cosh_loss(predictions, targets)
print(f"\nLog-Cosh Loss: {log_cosh.item():.4f}")
print("  → Smooth approximation to MAE, less sensitive to outliers")


# ============================================================================
# 2. Cross-Entropy Loss: Numerical Stability
# ============================================================================
print_section("2. Cross-Entropy Loss with Numerical Stability")

torch.manual_seed(42)
batch_size, num_classes = 8, 10
logits = torch.randn(batch_size, num_classes)
targets = torch.randint(0, num_classes, (batch_size,))

print(f"Logits shape: {logits.shape}")
print(f"Targets: {targets.tolist()}")

# Built-in CrossEntropyLoss
ce_loss = F.cross_entropy(logits, targets)
print(f"\nBuilt-in CE Loss: {ce_loss.item():.4f}")

# Manual implementation (naive - unstable)
def cross_entropy_naive(logits, targets):
    """Naive CE implementation (can cause numerical overflow)."""
    probs = torch.exp(logits) / torch.exp(logits).sum(dim=1, keepdim=True)
    log_probs = torch.log(probs)
    nll = -log_probs[range(len(targets)), targets]
    return nll.mean()

ce_naive = cross_entropy_naive(logits, targets)
print(f"Naive CE Loss: {ce_naive.item():.4f}")

# Manual implementation (stable with log-sum-exp trick)
def cross_entropy_stable(logits, targets):
    """Stable CE implementation using log-sum-exp trick."""
    # Subtract max for numerical stability
    logits_max = logits.max(dim=1, keepdim=True)[0]
    log_sum_exp = torch.log(torch.exp(logits - logits_max).sum(dim=1, keepdim=True))
    log_probs = logits - logits_max - log_sum_exp
    nll = -log_probs[range(len(targets)), targets]
    return nll.mean()

ce_stable = cross_entropy_stable(logits, targets)
print(f"Stable CE Loss: {ce_stable.item():.4f}")

# With extreme logits (show stability)
extreme_logits = logits * 100  # Scale up to cause overflow in naive version
ce_builtin_extreme = F.cross_entropy(extreme_logits, targets)
ce_stable_extreme = cross_entropy_stable(extreme_logits, targets)
print(f"\nExtreme logits (×100):")
print(f"  Built-in CE: {ce_builtin_extreme.item():.4f}")
print(f"  Stable CE:   {ce_stable_extreme.item():.4f}")


# ============================================================================
# 3. Focal Loss: Handling Class Imbalance
# ============================================================================
print_section("3. Focal Loss for Hard Examples")

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -α(1 - p_t)^γ log(p_t)

    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher → more focus on hard examples)
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    probs = F.softmax(logits, dim=1)
    p_t = probs[range(len(targets)), targets]

    # Focal term: (1 - p_t)^gamma
    focal_term = (1 - p_t) ** gamma

    # Apply alpha weighting
    loss = alpha * focal_term * ce_loss
    return loss.mean()

torch.manual_seed(42)
# Create predictions with varying confidence
easy_logits = torch.randn(8, 10) * 0.5  # Low variance → confident
easy_logits[range(8), targets] += 5.0   # Correct class has high logit

hard_logits = torch.randn(8, 10)        # Uncertain predictions

print("Easy examples (high confidence):")
ce_easy = F.cross_entropy(easy_logits, targets)
focal_easy = focal_loss(easy_logits, targets, gamma=2.0)
print(f"  CE Loss:    {ce_easy.item():.4f}")
print(f"  Focal Loss: {focal_easy.item():.4f}")
print(f"  Reduction:  {(ce_easy - focal_easy) / ce_easy * 100:.1f}%")

print("\nHard examples (uncertain):")
ce_hard = F.cross_entropy(hard_logits, targets)
focal_hard = focal_loss(hard_logits, targets, gamma=2.0)
print(f"  CE Loss:    {ce_hard.item():.4f}")
print(f"  Focal Loss: {focal_hard.item():.4f}")
print(f"  Reduction:  {(ce_hard - focal_hard) / ce_hard * 100:.1f}%")

print("\nEffect of gamma parameter:")
for gamma in [0.0, 1.0, 2.0, 5.0]:
    fl = focal_loss(easy_logits, targets, gamma=gamma)
    print(f"  γ={gamma:.1f}: {fl.item():.4f}")


# ============================================================================
# 4. Label Smoothing: Confidence Calibration
# ============================================================================
print_section("4. Label Smoothing for Calibration")

def cross_entropy_with_label_smoothing(logits, targets, smoothing=0.1):
    """
    Cross-entropy with label smoothing.
    Instead of [0, 0, 1, 0], use [ε/K, ε/K, 1-ε+ε/K, ε/K]
    where ε is smoothing factor, K is num_classes.
    """
    num_classes = logits.size(-1)
    confidence = 1.0 - smoothing

    # Create smoothed labels
    smooth_labels = torch.full_like(logits, smoothing / num_classes)
    smooth_labels.scatter_(1, targets.unsqueeze(1), confidence)

    # Compute loss
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(smooth_labels * log_probs).sum(dim=1)
    return loss.mean()

torch.manual_seed(42)
logits = torch.randn(8, 10)
targets = torch.randint(0, 10, (8,))

print("Without label smoothing:")
ce_standard = F.cross_entropy(logits, targets)
probs_standard = F.softmax(logits, dim=1)
confidence_standard = probs_standard[range(8), targets].mean()
print(f"  CE Loss: {ce_standard.item():.4f}")
print(f"  Avg confidence on true class: {confidence_standard.item():.4f}")

print("\nWith label smoothing (ε=0.1):")
ce_smooth = cross_entropy_with_label_smoothing(logits, targets, smoothing=0.1)
print(f"  CE Loss: {ce_smooth.item():.4f}")
print(f"  → Encourages model to be less confident")

print("\nSmoothing factor comparison:")
for eps in [0.0, 0.05, 0.1, 0.2, 0.3]:
    loss = cross_entropy_with_label_smoothing(logits, targets, smoothing=eps)
    print(f"  ε={eps:.2f}: {loss.item():.4f}")


# ============================================================================
# 5. Contrastive Loss: Metric Learning for Pairs
# ============================================================================
print_section("5. Contrastive Loss for Similarity Learning")

def contrastive_loss(embedding1, embedding2, label, margin=1.0):
    """
    Contrastive loss for pairs.
    L = (1-y) * 0.5 * D^2 + y * 0.5 * max(0, margin - D)^2
    where y=0 for similar pairs, y=1 for dissimilar pairs, D is distance.
    """
    distance = F.pairwise_distance(embedding1, embedding2)

    # Similar pairs: minimize distance
    loss_similar = (1 - label) * 0.5 * distance ** 2

    # Dissimilar pairs: push apart beyond margin
    loss_dissimilar = label * 0.5 * torch.clamp(margin - distance, min=0) ** 2

    return (loss_similar + loss_dissimilar).mean(), distance.mean()

torch.manual_seed(42)
embedding_dim = 128

# Positive pairs (similar)
emb1_pos = torch.randn(16, embedding_dim)
emb2_pos = emb1_pos + torch.randn(16, embedding_dim) * 0.1  # Small noise
labels_pos = torch.zeros(16)  # Label 0 = similar

# Negative pairs (dissimilar)
emb1_neg = torch.randn(16, embedding_dim)
emb2_neg = torch.randn(16, embedding_dim)  # Independent
labels_neg = torch.ones(16)  # Label 1 = dissimilar

# Combine
emb1 = torch.cat([emb1_pos, emb1_neg])
emb2 = torch.cat([emb2_pos, emb2_neg])
labels = torch.cat([labels_pos, labels_neg])

loss, avg_dist = contrastive_loss(emb1, emb2, labels, margin=1.0)

print(f"Total pairs: {len(labels)}")
print(f"  Positive pairs: {(labels == 0).sum().item()}")
print(f"  Negative pairs: {(labels == 1).sum().item()}")
print(f"\nContrastive Loss: {loss.item():.4f}")
print(f"Average pairwise distance: {avg_dist.item():.4f}")

# Check distances separately
pos_dist = F.pairwise_distance(emb1_pos, emb2_pos).mean()
neg_dist = F.pairwise_distance(emb1_neg, emb2_neg).mean()
print(f"\nPositive pair distance: {pos_dist.item():.4f} (should be small)")
print(f"Negative pair distance: {neg_dist.item():.4f} (should be > margin)")


# ============================================================================
# 6. Triplet Loss: Anchor/Positive/Negative
# ============================================================================
print_section("6. Triplet Loss for Ranking")

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Triplet loss: L = max(0, D(a,p) - D(a,n) + margin)
    Push positive closer than negative by at least margin.
    """
    distance_pos = F.pairwise_distance(anchor, positive)
    distance_neg = F.pairwise_distance(anchor, negative)

    loss = torch.clamp(distance_pos - distance_neg + margin, min=0)
    return loss.mean(), distance_pos.mean(), distance_neg.mean()

torch.manual_seed(42)
num_triplets = 16
embedding_dim = 128

# Anchors
anchors = torch.randn(num_triplets, embedding_dim)

# Positives: same class, small perturbation
positives = anchors + torch.randn(num_triplets, embedding_dim) * 0.2

# Negatives: different class
negatives = torch.randn(num_triplets, embedding_dim)

loss, d_pos, d_neg = triplet_loss(anchors, positives, negatives, margin=1.0)

print(f"Triplets: {num_triplets}")
print(f"Embedding dim: {embedding_dim}")
print(f"\nTriplet Loss: {loss.item():.4f}")
print(f"Avg D(anchor, positive): {d_pos.item():.4f}")
print(f"Avg D(anchor, negative): {d_neg.item():.4f}")
print(f"Margin satisfaction: {(d_neg - d_pos).item():.4f} (should be > margin=1.0)")

# Built-in triplet margin loss
loss_builtin = F.triplet_margin_loss(anchors, positives, negatives, margin=1.0)
print(f"\nBuilt-in TripletMarginLoss: {loss_builtin.item():.4f}")

print("\nEffect of margin:")
for m in [0.5, 1.0, 2.0, 5.0]:
    loss_m, _, _ = triplet_loss(anchors, positives, negatives, margin=m)
    print(f"  margin={m:.1f}: {loss_m.item():.4f}")


# ============================================================================
# 7. Dice Loss: Segmentation
# ============================================================================
print_section("7. Dice Loss for Binary Segmentation")

def dice_loss(pred, target, smooth=1e-6):
    """
    Dice loss for segmentation.
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice
    """
    pred = torch.sigmoid(pred)  # Convert logits to probabilities

    # Flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice

torch.manual_seed(42)
batch_size, height, width = 4, 64, 64

# Create synthetic segmentation masks
target_masks = torch.zeros(batch_size, height, width)
# Add some positive regions
for i in range(batch_size):
    x, y = torch.randint(10, 50, (2,))
    w, h = torch.randint(10, 20, (2,))
    target_masks[i, x:x+w, y:y+h] = 1.0

# Predicted logits (noisy version of target)
pred_logits = target_masks * 5.0 + torch.randn(batch_size, height, width) * 2.0

print(f"Mask shape: {target_masks.shape}")
print(f"Positive pixels: {target_masks.sum().item()} / {target_masks.numel()}")

# Compute losses
dice = dice_loss(pred_logits, target_masks)
bce = F.binary_cross_entropy_with_logits(pred_logits, target_masks)

print(f"\nDice Loss: {dice.item():.4f}")
print(f"BCE Loss:  {bce.item():.4f}")
print("  → Dice is better for imbalanced segmentation")

# Dice coefficient (metric, not loss)
pred_probs = torch.sigmoid(pred_logits)
pred_binary = (pred_probs > 0.5).float()
dice_coef = 1 - dice_loss(pred_logits, target_masks)
print(f"\nDice Coefficient: {dice_coef.item():.4f} (higher is better)")

# IoU for comparison
intersection = (pred_binary * target_masks).sum()
union = pred_binary.sum() + target_masks.sum() - intersection
iou = intersection / (union + 1e-6)
print(f"IoU: {iou.item():.4f}")


# ============================================================================
# 8. Custom Multi-Task Loss: Uncertainty Weighting
# ============================================================================
print_section("8. Multi-Task Loss with Learned Uncertainty")

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learned uncertainty weighting.
    From "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al.)

    L = (1/2σ₁²)L₁ + (1/2σ₂²)L₂ + log(σ₁σ₂)
    where σ₁, σ₂ are learned task uncertainties.
    """
    def __init__(self, num_tasks=2):
        super().__init__()
        # Log variance parameters (learnable)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        Args:
            losses: List of task losses [L1, L2, ...]
        """
        weighted_losses = []
        for i, loss in enumerate(losses):
            # Precision weighting: exp(-log_var) = 1/σ²
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)

        return torch.stack(weighted_losses).sum()

# Simulate two tasks
torch.manual_seed(42)

# Task 1: Regression (MSE)
pred1 = torch.randn(32, 10)
target1 = torch.randn(32, 10)
loss1 = F.mse_loss(pred1, target1)

# Task 2: Classification (CE)
pred2 = torch.randn(32, 10)
target2 = torch.randint(0, 10, (32,))
loss2 = F.cross_entropy(pred2, target2)

print("Task losses (unweighted):")
print(f"  Task 1 (regression): {loss1.item():.4f}")
print(f"  Task 2 (classification): {loss2.item():.4f}")
print(f"  Simple sum: {(loss1 + loss2).item():.4f}")

# Multi-task loss with learned weights
multi_task_loss = MultiTaskLoss(num_tasks=2)
weighted_loss = multi_task_loss([loss1, loss2])

print(f"\nWeighted multi-task loss: {weighted_loss.item():.4f}")
print(f"Task uncertainties (σ):")
print(f"  Task 1: {torch.exp(0.5 * multi_task_loss.log_vars[0]).item():.4f}")
print(f"  Task 2: {torch.exp(0.5 * multi_task_loss.log_vars[1]).item():.4f}")

# Simulate training: task 2 is harder (higher loss)
print("\nAfter simulated training (task 2 is harder):")
multi_task_loss.log_vars.data = torch.tensor([0.0, 1.0])  # Higher uncertainty for task 2
weighted_loss = multi_task_loss([loss1, loss2])
print(f"Weighted loss: {weighted_loss.item():.4f}")
print(f"  Task 1 weight (1/σ²): {torch.exp(-multi_task_loss.log_vars[0]).item():.4f}")
print(f"  Task 2 weight (1/σ²): {torch.exp(-multi_task_loss.log_vars[1]).item():.4f}")
print("  → Task 2 gets lower weight due to higher uncertainty")


# ============================================================================
# 9. Loss Landscape Visualization
# ============================================================================
print_section("9. Loss Landscape Comparison")

def quadratic_loss(w1, w2):
    """Simple quadratic loss: L = w1² + w2²"""
    return w1**2 + w2**2

def rosenbrock_loss(w1, w2, a=1, b=100):
    """Rosenbrock function: non-convex, valley-shaped"""
    return (a - w1)**2 + b * (w2 - w1**2)**2

def loss_landscape_stats(loss_fn, grid_size=50, range_val=2.0):
    """Compute statistics for a loss landscape."""
    w1 = np.linspace(-range_val, range_val, grid_size)
    w2 = np.linspace(-range_val, range_val, grid_size)

    losses = np.zeros((grid_size, grid_size))
    for i, w1_val in enumerate(w1):
        for j, w2_val in enumerate(w2):
            losses[i, j] = loss_fn(w1_val, w2_val)

    return {
        'min': losses.min(),
        'max': losses.max(),
        'mean': losses.mean(),
        'std': losses.std()
    }

print("Quadratic loss landscape:")
quad_stats = loss_landscape_stats(quadratic_loss)
print(f"  Min: {quad_stats['min']:.4f}")
print(f"  Max: {quad_stats['max']:.4f}")
print(f"  Mean: {quad_stats['mean']:.4f}")
print(f"  Std: {quad_stats['std']:.4f}")
print("  → Convex, smooth, single minimum")

print("\nRosenbrock loss landscape:")
rosen_stats = loss_landscape_stats(rosenbrock_loss, range_val=2.0)
print(f"  Min: {rosen_stats['min']:.4f}")
print(f"  Max: {rosen_stats['max']:.4f}")
print(f"  Mean: {rosen_stats['mean']:.4f}")
print(f"  Std: {rosen_stats['std']:.4f}")
print("  → Non-convex, narrow valley, harder optimization")

# Simple gradient descent comparison
print("\nGradient descent comparison (10 steps, lr=0.01):")
w_quad = torch.tensor([1.5, 1.5], requires_grad=True)
w_rosen = torch.tensor([1.5, 1.5], requires_grad=True)

lr = 0.01
for step in range(10):
    # Quadratic
    loss_q = w_quad[0]**2 + w_quad[1]**2
    loss_q.backward()
    with torch.no_grad():
        w_quad -= lr * w_quad.grad
        w_quad.grad.zero_()

    # Rosenbrock
    loss_r = (1 - w_rosen[0])**2 + 100 * (w_rosen[1] - w_rosen[0]**2)**2
    loss_r.backward()
    with torch.no_grad():
        w_rosen -= lr * w_rosen.grad
        w_rosen.grad.zero_()

print(f"Quadratic final: w = [{w_quad[0].item():.4f}, {w_quad[1].item():.4f}]")
print(f"  Distance to optimum (0,0): {torch.norm(w_quad).item():.4f}")
print(f"Rosenbrock final: w = [{w_rosen[0].item():.4f}, {w_rosen[1].item():.4f}]")
print(f"  Distance to optimum (1,1): {torch.norm(w_rosen - torch.tensor([1.0, 1.0])).item():.4f}")


# ============================================================================
# Summary
# ============================================================================
print_section("Summary: Loss Function Selection Guide")

print("""
Regression:
  - MSE: Standard, sensitive to outliers
  - MAE: Robust to outliers, but non-smooth at zero
  - Huber: Best of both worlds, smooth and robust
  - Log-Cosh: Smooth MAE alternative

Classification:
  - Cross-Entropy: Standard, use with softmax
  - Focal Loss: Class imbalance, focus on hard examples
  - Label Smoothing: Improve calibration, prevent overconfidence

Metric Learning:
  - Contrastive: Pairwise similarity
  - Triplet: Ranking, requires triplets
  - InfoNCE: Contrastive with multiple negatives

Segmentation:
  - Dice Loss: Handles class imbalance better than BCE
  - Focal Loss: Can be adapted for segmentation
  - Combo: Dice + BCE often works best

Multi-Task:
  - Uncertainty weighting: Let model learn task importance
  - Manual weighting: Domain knowledge required
  - Gradient balancing: GradNorm, PCGrad

Implementation Tips:
  1. Always use numerical stability tricks (log-sum-exp)
  2. Check for NaN/inf during training
  3. Normalize losses if combining multiple terms
  4. Visualize loss landscapes to understand optimization
  5. Consider task-specific requirements (imbalance, outliers, etc.)
""")

print("\n" + "=" * 60)
print("Loss functions demonstration complete!")
print("=" * 60)
