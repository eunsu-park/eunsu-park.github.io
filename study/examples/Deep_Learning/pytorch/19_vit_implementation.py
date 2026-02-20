"""
19. Vision Transformer (ViT) Implementation

A minimal but complete implementation of Vision Transformer:
- Patch Embedding
- Position Embedding
- Multi-Head Self-Attention
- Transformer Encoder
- Classification Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import math

print("=" * 60)
print("Vision Transformer (ViT) Implementation")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================
# 1. Patch Embedding
# ============================================
print("\n[1] Patch Embedding")
print("-" * 40)


class PatchEmbedding(nn.Module):
    """Convert image to patches and embed them"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Use Conv2d for efficient patch extraction and embedding
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)        # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, embed_dim)
        return x


# Test patch embedding
patch_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=768)
test_img = torch.randn(2, 3, 224, 224)
patches = patch_embed(test_img)
print(f"Input image: {test_img.shape}")
print(f"Patch embeddings: {patches.shape}")
print(f"Number of patches: {patch_embed.num_patches}")


# ============================================
# 2. Multi-Head Self-Attention
# ============================================
print("\n[2] Multi-Head Self-Attention")
print("-" * 40)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection in one matrix for efficiency
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # QKV: (B, N, 3*embed_dim) -> (B, N, 3, heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention: (B, heads, N, head_dim) @ (B, heads, head_dim, N) = (B, heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Output: (B, heads, N, N) @ (B, heads, N, head_dim) = (B, heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# Test attention
mha = MultiHeadAttention(embed_dim=768, num_heads=12)
attn_out = mha(patches)
print(f"Attention output: {attn_out.shape}")


# ============================================
# 3. Transformer Block
# ============================================
print("\n[3] Transformer Block")
print("-" * 40)


class MLP(nn.Module):
    """MLP Block with GELU activation"""
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# Test transformer block
block = TransformerBlock(embed_dim=768, num_heads=12)
block_out = block(patches)
print(f"Transformer block output: {block_out.shape}")


# ============================================
# 4. Vision Transformer (Full Model)
# ============================================
print("\n[4] Vision Transformer Model")
print("-" * 40)


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT)"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Position Embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)

        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification Head
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        B = x.shape[0]

        # Patch Embedding
        x = self.patch_embed(x)  # (B, N, D)

        # Add CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Add Position Embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if return_features:
            return x

        # CLS Token for classification
        cls_output = x[:, 0]
        return self.head(cls_output)


# Create different ViT variants
def vit_tiny(num_classes=1000):
    return VisionTransformer(
        embed_dim=192, depth=12, num_heads=3, num_classes=num_classes
    )


def vit_small(num_classes=1000):
    return VisionTransformer(
        embed_dim=384, depth=12, num_heads=6, num_classes=num_classes
    )


def vit_base(num_classes=1000):
    return VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, num_classes=num_classes
    )


# Test ViT
model = vit_tiny(num_classes=10)
test_output = model(test_img)
print(f"ViT-Tiny input: {test_img.shape}")
print(f"ViT-Tiny output: {test_output.shape}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================
# 5. CIFAR-10 Training
# ============================================
print("\n[5] Training on CIFAR-10")
print("-" * 40)


# Custom ViT for CIFAR-10 (32x32 images)
class ViTForCIFAR(nn.Module):
    """ViT adapted for CIFAR-10 (32x32 images)"""
    def __init__(self, num_classes=10):
        super().__init__()
        # Smaller patch size for 32x32 images
        self.vit = VisionTransformer(
            img_size=32,
            patch_size=4,  # 32/4 = 8x8 = 64 patches
            in_channels=3,
            num_classes=num_classes,
            embed_dim=256,
            depth=6,
            num_heads=8,
            mlp_ratio=2.0,
            dropout=0.1
        )

    def forward(self, x):
        return self.vit(x)


def train_vit_cifar10(epochs=10, batch_size=128, lr=1e-3):
    """Train ViT on CIFAR-10"""
    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('data', train=False, transform=transform_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = ViTForCIFAR(num_classes=10).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Training
    train_losses = []
    test_accs = []

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        # Evaluate
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        test_accs.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs}: Loss={train_losses[-1]:.4f}, Acc={accuracy:.2f}%")

        scheduler.step()

    return model, train_losses, test_accs


# Train (reduced epochs for demo)
print("\nStarting training...")
model, losses, accs = train_vit_cifar10(epochs=5)


# ============================================
# 6. Visualizations
# ============================================
print("\n[6] Visualizations")
print("-" * 40)

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(losses)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')

ax2.plot(accs)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Test Accuracy')

plt.tight_layout()
plt.savefig('vit_training.png', dpi=150)
plt.close()
print("Training curves saved to vit_training.png")


# Visualize position embeddings
def visualize_position_embeddings(model, filename='vit_pos_embed.png'):
    """Visualize learned position embeddings"""
    pos_embed = model.vit.pos_embed.detach().cpu()
    # Remove CLS token
    pos_embed = pos_embed[0, 1:]  # (N, D)

    # Compute similarity
    pos_embed_norm = pos_embed / pos_embed.norm(dim=1, keepdim=True)
    similarity = pos_embed_norm @ pos_embed_norm.T

    # Get grid size
    num_patches = pos_embed.shape[0]
    grid_size = int(num_patches ** 0.5)

    # Visualize similarity for corner and center patches
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    patch_indices = [
        (0, "Top-Left"),
        (grid_size - 1, "Top-Right"),
        (num_patches // 2 - grid_size // 2, "Center"),
        (num_patches - grid_size, "Bottom-Left"),
        (num_patches - 1, "Bottom-Right"),
        (grid_size // 2, "Top-Center")
    ]

    for idx, (patch_idx, name) in enumerate(patch_indices):
        ax = axes[idx // 3, idx % 3]
        sim = similarity[patch_idx].reshape(grid_size, grid_size)
        im = ax.imshow(sim.numpy(), cmap='viridis')
        ax.set_title(f'{name} (patch {patch_idx})')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle('Position Embedding Similarity')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Position embeddings saved to {filename}")


visualize_position_embeddings(model)


# ============================================
# 7. Attention Visualization
# ============================================
print("\n[7] Attention Visualization")
print("-" * 40)


class ViTWithAttention(nn.Module):
    """ViT that returns attention weights"""
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.vit.vit.patch_embed(x)
        cls_tokens = self.vit.vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.vit.vit.pos_embed

        # Get attention from first block
        attn_weights = []

        for block in self.vit.vit.blocks:
            # Extract attention weights manually
            norm_x = block.norm1(x)
            B, N, C = norm_x.shape
            qkv = block.attn.qkv(norm_x).reshape(B, N, 3, block.attn.num_heads, block.attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            attn_weights.append(attn)

            x = x + block.attn(block.norm1(x))
            x = x + block.mlp(block.norm2(x))

        return attn_weights


# Get attention for a sample image
model.eval()
test_data = datasets.CIFAR10('data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
]))

sample_img, label = test_data[0]
sample_img = sample_img.unsqueeze(0).to(device)

vit_attn = ViTWithAttention(model)
with torch.no_grad():
    attentions = vit_attn(sample_img)

# Visualize CLS token attention from last layer
attn_last = attentions[-1][0]  # (heads, N, N)
cls_attn = attn_last[:, 0, 1:]  # Attention from CLS to patches

# Average over heads
cls_attn_avg = cls_attn.mean(0).cpu()
grid_size = int(cls_attn_avg.shape[0] ** 0.5)
cls_attn_map = cls_attn_avg.reshape(grid_size, grid_size)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Original image
orig_img = sample_img[0].cpu().permute(1, 2, 0)
orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
axes[0].imshow(orig_img)
axes[0].set_title(f'Original (Label: {test_data.classes[label]})')
axes[0].axis('off')

# Attention map
axes[1].imshow(cls_attn_map.numpy(), cmap='hot')
axes[1].set_title('CLS Token Attention')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('vit_attention.png', dpi=150)
plt.close()
print("Attention visualization saved to vit_attention.png")


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Vision Transformer Summary")
print("=" * 60)

summary = """
Key Components:
1. Patch Embedding: Image -> Patches -> Linear projection
2. CLS Token: Learnable token for classification
3. Position Embedding: Learnable position information
4. Transformer Blocks: Multi-Head Attention + MLP
5. Classification Head: Linear layer on CLS output

ViT Variants:
- ViT-Tiny: 192 dim, 3 heads, 12 layers (~5M params)
- ViT-Small: 384 dim, 6 heads, 12 layers (~22M params)
- ViT-Base: 768 dim, 12 heads, 12 layers (~86M params)
- ViT-Large: 1024 dim, 16 heads, 24 layers (~307M params)

Training Tips:
1. Use AdamW optimizer with weight decay
2. Learning rate warmup + cosine decay
3. Strong data augmentation (RandAugment, Mixup)
4. Gradient clipping
5. Pre-training on large datasets helps significantly

Output Files:
- vit_training.png: Training loss and accuracy curves
- vit_pos_embed.png: Position embedding similarity
- vit_attention.png: CLS token attention visualization
"""
print(summary)
print("=" * 60)
