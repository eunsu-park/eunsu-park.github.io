"""
Vision Transformer (ViT) vs CNN Comparison on CIFAR-10

This script compares Vision Transformers and Convolutional Neural Networks
on image classification. It implements both architectures from scratch and
provides detailed performance comparisons.

Key Concepts:
- Vision Transformer: Patch embedding, positional encoding, transformer blocks
- CNN: Convolutional layers, batch normalization, residual connections
- Performance metrics: Accuracy, training time, parameter count
- Visualization: Attention maps (ViT) vs feature maps (CNN)

Requirements:
    pip install torch torchvision matplotlib numpy
"""

import argparse
import time
from typing import Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, height, width)
        x = self.projection(x)  # (B, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, embed_dim = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, seq_len, 3*embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        output = self.projection(attn_output)

        return output, attn_weights.mean(dim=1)  # Return averaged attention across heads


class TransformerBlock(nn.Module):
    """Transformer encoder block with self-attention and MLP."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection
        attn_output, attn_weights = self.attn(self.norm1(x))
        x = x + attn_output

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights


class VisionTransformer(nn.Module):
    """Simple Vision Transformer for CIFAR-10."""

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3,
                 num_classes: int = 10, embed_dim: int = 128, depth: int = 6,
                 num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        num_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer blocks
        attn_weights_list = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attn_weights_list.append(attn_weights)

        # Classification using CLS token
        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)

        return logits, attn_weights_list


class SimpleCNN(nn.Module):
    """Simple CNN baseline for comparison."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Convolutional blocks
        self.conv1 = self._make_conv_block(3, 64)
        self.conv2 = self._make_conv_block(64, 128)
        self.conv3 = self._make_conv_block(128, 256)
        self.conv4 = self._make_conv_block(256, 512)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass (handle both ViT and CNN outputs)
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # ViT returns (logits, attention_weights)

        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, dataloader: DataLoader,
             criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    eval_loss = running_loss / len(dataloader)
    eval_acc = 100.0 * correct / total

    return eval_loss, eval_acc


def visualize_attention(model: VisionTransformer, dataloader: DataLoader,
                       device: torch.device, save_path: str = "vit_attention.png"):
    """Visualize attention maps from ViT."""
    model.eval()

    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images.to(device)

    with torch.no_grad():
        _, attn_weights_list = model(images[:4])  # Use first 4 images

    # Visualize attention from last layer
    attn_weights = attn_weights_list[-1][:4].cpu()  # (4, seq_len, seq_len)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for idx in range(4):
        # Original image
        img = images[idx].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        axes[0, idx].imshow(img)
        axes[0, idx].set_title(f"Image {idx + 1}")
        axes[0, idx].axis("off")

        # Attention map (CLS token attending to patches)
        attn_map = attn_weights[idx, 0, 1:].reshape(8, 8)  # Skip CLS token itself
        axes[1, idx].imshow(attn_map, cmap="viridis")
        axes[1, idx].set_title("CLS Token Attention")
        axes[1, idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Attention visualization saved to {save_path}")


def plot_comparison(vit_history: dict, cnn_history: dict,
                   save_path: str = "comparison.png"):
    """Plot training comparison between ViT and CNN."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Training loss
    axes[0].plot(vit_history["train_loss"], label="ViT", marker="o")
    axes[0].plot(cnn_history["train_loss"], label="CNN", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Validation accuracy
    axes[1].plot(vit_history["val_acc"], label="ViT", marker="o")
    axes[1].plot(cnn_history["val_acc"], label="CNN", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Comparison plot saved to {save_path}")


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preparation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Initialize models
    vit_model = VisionTransformer(
        img_size=32, patch_size=4, num_classes=10,
        embed_dim=128, depth=6, num_heads=4
    ).to(device)

    cnn_model = SimpleCNN(num_classes=10).to(device)

    print(f"\nViT Parameters: {count_parameters(vit_model):,}")
    print(f"CNN Parameters: {count_parameters(cnn_model):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    vit_optimizer = torch.optim.AdamW(vit_model.parameters(), lr=args.lr, weight_decay=0.05)
    cnn_optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=args.lr, weight_decay=0.05)

    # Learning rate schedulers
    vit_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(vit_optimizer, T_max=args.epochs)
    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cnn_optimizer, T_max=args.epochs)

    # Training histories
    vit_history = {"train_loss": [], "val_acc": []}
    cnn_history = {"train_loss": [], "val_acc": []}

    # Training loop
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train ViT
        vit_start = time.time()
        vit_loss, vit_train_acc = train_epoch(
            vit_model, train_loader, criterion, vit_optimizer, device
        )
        vit_time = time.time() - vit_start
        vit_scheduler.step()

        # Train CNN
        cnn_start = time.time()
        cnn_loss, cnn_train_acc = train_epoch(
            cnn_model, train_loader, criterion, cnn_optimizer, device
        )
        cnn_time = time.time() - cnn_start
        cnn_scheduler.step()

        # Evaluate
        _, vit_val_acc = evaluate(vit_model, test_loader, criterion, device)
        _, cnn_val_acc = evaluate(cnn_model, test_loader, criterion, device)

        # Store history
        vit_history["train_loss"].append(vit_loss)
        vit_history["val_acc"].append(vit_val_acc)
        cnn_history["train_loss"].append(cnn_loss)
        cnn_history["val_acc"].append(cnn_val_acc)

        # Print results
        print(f"ViT - Loss: {vit_loss:.4f}, Train Acc: {vit_train_acc:.2f}%, "
              f"Val Acc: {vit_val_acc:.2f}%, Time: {vit_time:.2f}s")
        print(f"CNN - Loss: {cnn_loss:.4f}, Train Acc: {cnn_train_acc:.2f}%, "
              f"Val Acc: {cnn_val_acc:.2f}%, Time: {cnn_time:.2f}s")

    # Final comparison
    print("\n" + "="*50)
    print("Final Results")
    print("="*50)
    print(f"ViT - Best Val Acc: {max(vit_history['val_acc']):.2f}%")
    print(f"CNN - Best Val Acc: {max(cnn_history['val_acc']):.2f}%")

    # Visualizations
    visualize_attention(vit_model, test_loader, device)
    plot_comparison(vit_history, cnn_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT vs CNN Comparison on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    args = parser.parse_args()
    main(args)
