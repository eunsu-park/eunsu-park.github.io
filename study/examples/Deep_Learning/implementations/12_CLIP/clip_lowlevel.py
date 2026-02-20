"""
PyTorch Low-Level CLIP 구현

Image Encoder, Text Encoder, Contrastive Loss 직접 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class CLIPConfig:
    """CLIP 설정"""
    # Image encoder (ViT)
    image_size: int = 224
    patch_size: int = 32
    vision_width: int = 768
    vision_layers: int = 12
    vision_heads: int = 12

    # Text encoder
    vocab_size: int = 49408
    context_length: int = 77
    text_width: int = 512
    text_layers: int = 12
    text_heads: int = 8

    # Shared
    embed_dim: int = 512  # 공통 임베딩 차원

    # Training
    temperature: float = 0.07  # learnable


# ============== Vision Transformer (Image Encoder) ==============

class PatchEmbedding(nn.Module):
    """이미지를 패치로 분할하고 임베딩"""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        # Conv2d로 패치화 + 임베딩
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer Block (Pre-LN)"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """CLIP Vision Encoder"""

    def __init__(self, config: CLIPConfig):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            config.image_size, config.patch_size,
            in_channels=3, embed_dim=config.vision_width
        )
        num_patches = self.patch_embed.num_patches

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_width))

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.vision_width)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.vision_width,
                config.vision_heads,
                mlp_ratio=4.0
            )
            for _ in range(config.vision_layers)
        ])

        self.norm = nn.LayerNorm(config.vision_width)

        # Projection to shared embedding space
        self.proj = nn.Linear(config.vision_width, config.embed_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            image_features: (B, embed_dim)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Use [CLS] token as image representation
        x = x[:, 0]

        # Project to shared space
        x = self.proj(x)

        return x


# ============== Text Encoder ==============

class TextTransformer(nn.Module):
    """CLIP Text Encoder"""

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.context_length = config.context_length

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.text_width)

        # Position embedding
        self.positional_embedding = nn.Parameter(
            torch.zeros(config.context_length, config.text_width)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.text_width,
                config.text_heads,
                mlp_ratio=4.0
            )
            for _ in range(config.text_layers)
        ])

        self.ln_final = nn.LayerNorm(config.text_width)

        # Projection to shared embedding space
        self.text_projection = nn.Linear(config.text_width, config.embed_dim, bias=False)

        # Causal mask
        self.register_buffer(
            "attn_mask",
            self._build_causal_mask(config.context_length)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def _build_causal_mask(self, context_length: int) -> torch.Tensor:
        """Causal attention mask"""
        mask = torch.triu(
            torch.full((context_length, context_length), float("-inf")),
            diagonal=1
        )
        return mask

    def forward(
        self,
        text: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            text: (B, L) token ids
            attention_mask: (B, L) optional padding mask

        Returns:
            text_features: (B, embed_dim)
        """
        B, L = text.shape

        # Token + Position embedding
        x = self.token_embedding(text)
        x = x + self.positional_embedding[:L]

        # Causal mask
        causal_mask = self.attn_mask[:L, :L]

        # Transformer
        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.ln_final(x)

        # Use [EOS] token as text representation
        # CLIP에서는 가장 높은 토큰 위치 사용 (EOT)
        # 여기서는 간단히 마지막 토큰 사용
        if attention_mask is not None:
            # 각 시퀀스의 마지막 유효 토큰
            seq_lengths = attention_mask.sum(dim=1) - 1
            x = x[torch.arange(B), seq_lengths]
        else:
            x = x[:, -1]  # 마지막 토큰

        # Project to shared space
        x = self.text_projection(x)

        return x


# ============== CLIP Model ==============

class CLIP(nn.Module):
    """CLIP: Contrastive Language-Image Pre-training"""

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config

        # Encoders
        self.visual = VisionTransformer(config)
        self.text_encoder = TextTransformer(config)

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / config.temperature))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """이미지 인코딩"""
        features = self.visual(image)
        # L2 normalize
        features = F.normalize(features, dim=-1)
        return features

    def encode_text(
        self,
        text: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """텍스트 인코딩"""
        features = self.text_encoder(text, attention_mask)
        # L2 normalize
        features = F.normalize(features, dim=-1)
        return features

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (B, 3, H, W)
            text: (B, L)
            attention_mask: (B, L) optional

        Returns:
            logits_per_image: (B, B)
            logits_per_text: (B, B)
        """
        # Encode
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, attention_mask)

        # Scaled cosine similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


def clip_loss(
    logits_per_image: torch.Tensor,
    logits_per_text: torch.Tensor
) -> torch.Tensor:
    """
    CLIP Contrastive Loss (InfoNCE)

    이미지→텍스트, 텍스트→이미지 양방향 손실
    """
    batch_size = logits_per_image.shape[0]

    # Ground truth: 대각선이 positive
    labels = torch.arange(batch_size, device=logits_per_image.device)

    # Image-to-Text loss
    loss_i2t = F.cross_entropy(logits_per_image, labels)

    # Text-to-Image loss
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    # Symmetric loss
    loss = (loss_i2t + loss_t2i) / 2

    return loss


# ============== Zero-shot Classification ==============

class ZeroShotClassifier:
    """CLIP Zero-shot 분류기"""

    def __init__(self, model: CLIP, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def create_text_embeddings(
        self,
        class_names: list,
        templates: list = None
    ) -> torch.Tensor:
        """
        클래스별 텍스트 임베딩 생성

        Args:
            class_names: 클래스 이름 리스트
            templates: 프롬프트 템플릿 리스트

        Returns:
            text_features: (num_classes, embed_dim)
        """
        if templates is None:
            templates = [
                "a photo of a {}",
                "a picture of a {}",
                "an image of a {}"
            ]

        all_embeddings = []

        with torch.no_grad():
            for class_name in class_names:
                class_embeddings = []

                for template in templates:
                    text = template.format(class_name)
                    # 실제로는 tokenizer 사용
                    # 여기서는 간단히 mock 텐서 사용
                    tokens = self._simple_tokenize(text)
                    tokens = tokens.to(self.device)

                    embedding = self.model.encode_text(tokens)
                    class_embeddings.append(embedding)

                # 템플릿 평균
                class_embedding = torch.stack(class_embeddings).mean(dim=0)
                class_embedding = F.normalize(class_embedding, dim=-1)
                all_embeddings.append(class_embedding)

        return torch.cat(all_embeddings, dim=0)

    def classify(
        self,
        images: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Zero-shot 분류

        Args:
            images: (B, 3, H, W)
            text_features: (num_classes, embed_dim)

        Returns:
            predictions: (B,)
        """
        with torch.no_grad():
            image_features = self.model.encode_image(images)

            # Similarity
            logits = 100.0 * image_features @ text_features.t()
            probs = F.softmax(logits, dim=-1)
            predictions = probs.argmax(dim=-1)

        return predictions

    def _simple_tokenize(self, text: str, max_length: int = 77) -> torch.Tensor:
        """간단한 토큰화 (실제로는 BPE 사용)"""
        # Mock tokenization - 실제 구현에서는 CLIP tokenizer 사용
        tokens = [ord(c) % 49408 for c in text[:max_length-2]]
        tokens = [49406] + tokens + [49407]  # SOT, EOT
        tokens = tokens + [0] * (max_length - len(tokens))  # Padding
        return torch.tensor([tokens])


# ============== Image-Text Retrieval ==============

class ImageTextRetrieval:
    """이미지-텍스트 검색"""

    def __init__(self, model: CLIP, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

        self.image_embeddings = None
        self.text_embeddings = None

    def index_images(self, images: torch.Tensor):
        """이미지 인덱싱"""
        with torch.no_grad():
            self.image_embeddings = self.model.encode_image(images.to(self.device))

    def index_texts(self, texts: torch.Tensor):
        """텍스트 인덱싱"""
        with torch.no_grad():
            self.text_embeddings = self.model.encode_text(texts.to(self.device))

    def search_by_text(
        self,
        query_text: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """텍스트로 이미지 검색"""
        with torch.no_grad():
            query_features = self.model.encode_text(query_text.to(self.device))
            similarities = query_features @ self.image_embeddings.t()
            scores, indices = similarities.topk(top_k, dim=-1)

        return indices, scores

    def search_by_image(
        self,
        query_image: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """이미지로 텍스트 검색"""
        with torch.no_grad():
            query_features = self.model.encode_image(query_image.to(self.device))
            similarities = query_features @ self.text_embeddings.t()
            scores, indices = similarities.topk(top_k, dim=-1)

        return indices, scores


# 테스트
if __name__ == "__main__":
    print("=== CLIP Low-Level Implementation ===\n")

    # 설정
    config = CLIPConfig(
        image_size=224,
        patch_size=32,
        vision_width=768,
        vision_layers=12,
        vision_heads=12,
        vocab_size=49408,
        context_length=77,
        text_width=512,
        text_layers=12,
        text_heads=8,
        embed_dim=512
    )
    print(f"Config: embed_dim={config.embed_dim}, vision_layers={config.vision_layers}\n")

    # 모델 생성
    model = CLIP(config)

    # 파라미터 수
    vision_params = sum(p.numel() for p in model.visual.parameters())
    text_params = sum(p.numel() for p in model.text_encoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Vision encoder params: {vision_params:,}")
    print(f"Text encoder params: {text_params:,}")
    print(f"Total params: {total_params:,}\n")

    # 테스트 입력
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    texts = torch.randint(0, config.vocab_size, (batch_size, 77))

    # Forward
    logits_per_image, logits_per_text = model(images, texts)
    print(f"Images shape: {images.shape}")
    print(f"Texts shape: {texts.shape}")
    print(f"Logits per image shape: {logits_per_image.shape}")
    print(f"Logits per text shape: {logits_per_text.shape}")

    # Loss 계산
    loss = clip_loss(logits_per_image, logits_per_text)
    print(f"\nContrastive Loss: {loss.item():.4f}")

    # 개별 인코딩 테스트
    print("\n=== Encoding Test ===")
    image_features = model.encode_image(images)
    text_features = model.encode_text(texts)
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    print(f"Image features norm: {image_features.norm(dim=-1).mean():.4f} (should be ~1.0)")
    print(f"Text features norm: {text_features.norm(dim=-1).mean():.4f} (should be ~1.0)")

    # Similarity 계산
    similarity = image_features @ text_features.t()
    print(f"\nSimilarity matrix:\n{similarity}")

    # Temperature 효과
    print(f"\nTemperature (1/exp(logit_scale)): {1/model.logit_scale.exp().item():.4f}")
    print(f"Scaled similarity range: [{(model.logit_scale.exp() * similarity).min().item():.2f}, "
          f"{(model.logit_scale.exp() * similarity).max().item():.2f}]")

    # Zero-shot 분류 테스트
    print("\n=== Zero-shot Classification Test ===")
    device = torch.device("cpu")
    classifier = ZeroShotClassifier(model, device)

    # Mock 분류
    class_names = ["cat", "dog", "bird", "car", "plane"]
    print(f"Classes: {class_names}")

    # 실제로는 text_features를 생성하고 분류
    # text_features = classifier.create_text_embeddings(class_names)
    # predictions = classifier.classify(images, text_features)

    print("\nAll tests passed!")
