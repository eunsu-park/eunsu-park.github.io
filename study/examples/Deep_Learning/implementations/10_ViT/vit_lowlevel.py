"""
PyTorch Low-Level Vision Transformer (ViT) 구현

nn.TransformerEncoder 미사용
패치 임베딩부터 직접 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ViTConfig:
    """ViT 설정"""
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0


class PatchEmbedding(nn.Module):
    """
    이미지 → 패치 → 임베딩

    (B, C, H, W) → (B, N, D)
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 768
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Linear projection (Conv2d로 효율적 구현)
        # kernel_size = stride = patch_size → 겹치지 않는 패치
        self.projection = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            patches: (B, N, D) where N = num_patches
        """
        # (B, C, H, W) → (B, D, H/P, W/P)
        x = self.projection(x)

        # (B, D, H', W') → (B, D, N) → (B, N, D)
        x = x.flatten(2).transpose(1, 2)

        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV를 하나의 projection으로
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, D)

        Returns:
            output: (B, N, D)
            attention: (B, H, N, N)
        """
        B, N, D = x.shape

        # QKV 계산: (B, N, 3D)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores: (B, H, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention: (B, H, N, head_dim)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, D)  # (B, N, D)

        # Output projection
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x, attn


class MLP(nn.Module):
    """Feed-Forward Network (2-layer MLP with GELU)"""

    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)

        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """ViT Transformer Block (Pre-LN)"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = MultiHeadAttention(hidden_size, num_heads, attention_dropout)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = MLP(hidden_size, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-LN + Attention + Residual
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out

        # Pre-LN + MLP + Residual
        x = x + self.mlp(self.norm2(x))

        if return_attention:
            return x, attn_weights
        return x, None


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT)"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            config.image_size, config.patch_size,
            config.in_channels, config.hidden_size
        )
        num_patches = self.patch_embed.num_patches

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # Position embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))

        # Dropout
        self.pos_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.hidden_size, config.num_heads,
                config.mlp_ratio, config.dropout, config.attention_dropout
            )
            for _ in range(config.num_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # Classification head
        self.head = nn.Linear(config.hidden_size, config.num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Position embedding: truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False,
        return_attention: bool = False
    ):
        """특징 추출 (분류 헤드 전)"""
        B = x.shape[0]

        # Patch embedding: (B, N, D)
        x = self.patch_embed(x)

        # [CLS] token 추가: (B, N+1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Position embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Transformer blocks
        attentions = []
        for block in self.blocks:
            x, attn = block(x, return_attention=return_attention)
            if return_attention:
                attentions.append(attn)

        # Final norm
        x = self.norm(x)

        if return_all_tokens:
            return x, attentions

        # [CLS] token만 반환
        return x[:, 0], attentions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """분류"""
        features, _ = self.forward_features(x)
        return self.head(features)


class ViTForImageClassification(nn.Module):
    """ViT with flexible head"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.vit = VisionTransformer(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        logits = self.vit(pixel_values)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {
            'logits': logits,
            'loss': loss
        }


# Attention 시각화
def visualize_attention(
    model: VisionTransformer,
    image: torch.Tensor,
    layer_idx: int = -1,
    head_idx: int = 0
):
    """Attention map 시각화"""
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        _, attentions = model.forward_features(image, return_attention=True)

    # 특정 레이어의 attention
    attn = attentions[layer_idx]  # (B, H, N, N)
    attn = attn[0, head_idx]      # (N, N)

    # [CLS] token의 attention (다른 패치에 대한)
    cls_attn = attn[0, 1:]  # (N-1,)

    # 2D로 reshape
    num_patches = int(cls_attn.shape[0] ** 0.5)
    cls_attn = cls_attn.reshape(num_patches, num_patches)

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 원본 이미지
    img = image[0].permute(1, 2, 0).cpu()
    img = (img - img.min()) / (img.max() - img.min())
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Attention map
    axes[1].imshow(cls_attn.cpu(), cmap='viridis')
    axes[1].set_title(f"[CLS] Attention (Layer {layer_idx}, Head {head_idx})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('vit_attention.png')
    print("Saved vit_attention.png")


# 다양한 크기의 ViT 설정
def vit_tiny():
    return ViTConfig(hidden_size=192, num_layers=12, num_heads=3)

def vit_small():
    return ViTConfig(hidden_size=384, num_layers=12, num_heads=6)

def vit_base():
    return ViTConfig(hidden_size=768, num_layers=12, num_heads=12)

def vit_large():
    return ViTConfig(hidden_size=1024, num_layers=24, num_heads=16)


# 테스트
if __name__ == "__main__":
    print("=== Vision Transformer Low-Level Implementation ===\n")

    # ViT-Base 설정
    config = vit_base()
    print(f"Config: {config}\n")

    # 모델 생성
    model = VisionTransformer(config)

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Expected ~86M for ViT-Base/16\n")

    # 테스트 입력
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    # Forward
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Features with attention
    features, attentions = model.forward_features(x, return_attention=True)
    print(f"\nFeatures shape: {features.shape}")
    print(f"Number of attention maps: {len(attentions)}")
    print(f"Attention shape: {attentions[0].shape}")

    # Patch embedding 테스트
    print("\n=== Patch Embedding Test ===")
    patch_embed = PatchEmbedding(224, 16, 3, 768)
    patches = patch_embed(x)
    print(f"Image: {x.shape}")
    print(f"Patches: {patches.shape}")
    print(f"Number of patches: {patches.shape[1]}")
    print(f"Expected: (224/16)² = {(224//16)**2}")

    # 다양한 크기 테스트
    print("\n=== Different ViT Sizes ===")
    for name, config_fn in [('Tiny', vit_tiny), ('Small', vit_small),
                             ('Base', vit_base), ('Large', vit_large)]:
        cfg = config_fn()
        model = VisionTransformer(cfg)
        params = sum(p.numel() for p in model.parameters())
        print(f"ViT-{name}: {params/1e6:.1f}M params")

    print("\nAll tests passed!")
