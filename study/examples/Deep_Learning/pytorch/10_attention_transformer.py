"""
10. Attention과 Transformer

Attention 메커니즘과 Transformer를 PyTorch로 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("PyTorch Attention & Transformer")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================
# 1. Scaled Dot-Product Attention
# ============================================
print("\n[1] Scaled Dot-Product Attention")
print("-" * 40)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: (batch, seq_q, d_k)
        K: (batch, seq_k, d_k)
        V: (batch, seq_k, d_v)
        mask: (batch, seq_q, seq_k) or broadcastable
    Returns:
        output: (batch, seq_q, d_v)
        attention_weights: (batch, seq_q, seq_k)
    """
    d_k = K.size(-1)

    # 스코어 계산
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 마스킹
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # 가중 합
    output = torch.matmul(attention_weights, V)

    return output, attention_weights

# 테스트
batch_size = 2
seq_len = 5
d_k = 8

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Q, K, V: ({batch_size}, {seq_len}, {d_k})")
print(f"Output: {output.shape}")
print(f"Attention Weights: {weights.shape}")
print(f"Weights sum (should be 1): {weights[0, 0].sum().item():.4f}")


# ============================================
# 2. Multi-Head Attention
# ============================================
print("\n[2] Multi-Head Attention")
print("-" * 40)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 선형 변환
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        # 헤드 분할: (batch, seq, d_model) → (batch, heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 헤드 결합
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # 출력 변환
        output = self.W_O(attn_output)

        return output, attn_weights

# 테스트
mha = MultiHeadAttention(d_model=64, num_heads=8)
x = torch.randn(2, 10, 64)
output, weights = mha(x, x, x)
print(f"입력: {x.shape}")
print(f"출력: {output.shape}")
print(f"Attention Weights: {weights.shape}")


# ============================================
# 3. Positional Encoding
# ============================================
print("\n[3] Positional Encoding")
print("-" * 40)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Positional Encoding 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# 시각화
pe = PositionalEncoding(d_model=64)
positions = pe.pe[0, :50, :].numpy()

plt.figure(figsize=(12, 4))
plt.imshow(positions.T, aspect='auto', cmap='RdBu')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Positional Encoding')
plt.colorbar()
plt.savefig('positional_encoding.png', dpi=100)
plt.close()
print("그래프 저장: positional_encoding.png")


# ============================================
# 4. Feed Forward Network
# ============================================
print("\n[4] Feed Forward Network")
print("-" * 40)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

ff = FeedForward(d_model=64, d_ff=256)
x = torch.randn(2, 10, 64)
output = ff(x)
print(f"FFN 입력: {x.shape} → 출력: {output.shape}")


# ============================================
# 5. Transformer Encoder Layer
# ============================================
print("\n[5] Transformer Encoder Layer")
print("-" * 40)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention + Residual + LayerNorm
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x

encoder_layer = TransformerEncoderLayer(d_model=64, num_heads=8, d_ff=256)
x = torch.randn(2, 10, 64)
output = encoder_layer(x)
print(f"인코더 층 입력: {x.shape} → 출력: {output.shape}")


# ============================================
# 6. Full Transformer Encoder
# ============================================
print("\n[6] Transformer Encoder")
print("-" * 40)

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

encoder = TransformerEncoder(num_layers=6, d_model=64, num_heads=8, d_ff=256)
x = torch.randn(2, 10, 64)
output = encoder(x)
print(f"Transformer Encoder 출력: {output.shape}")
print(f"파라미터 수: {sum(p.numel() for p in encoder.parameters()):,}")


# ============================================
# 7. Transformer 분류기
# ============================================
print("\n[7] Transformer 분류기")
print("-" * 40)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes,
                 d_ff=None, max_len=512, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.fc = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        # x: (batch, seq) - 토큰 인덱스
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.encoder(x, mask)

        # 평균 풀링 또는 [CLS] 토큰
        x = x.mean(dim=1)

        return self.fc(x)

model = TransformerClassifier(
    vocab_size=10000,
    d_model=128,
    num_heads=8,
    num_layers=4,
    num_classes=5
)

x = torch.randint(0, 10000, (4, 32))
output = model(x)
print(f"분류기 입력: {x.shape}")
print(f"분류기 출력: {output.shape}")
print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")


# ============================================
# 8. PyTorch 내장 Transformer
# ============================================
print("\n[8] PyTorch 내장 Transformer")
print("-" * 40)

# nn.TransformerEncoder 사용
pytorch_encoder_layer = nn.TransformerEncoderLayer(
    d_model=64,
    nhead=8,
    dim_feedforward=256,
    dropout=0.1,
    batch_first=True  # PyTorch 1.9+
)

pytorch_encoder = nn.TransformerEncoder(pytorch_encoder_layer, num_layers=6)

x = torch.randn(2, 10, 64)
output = pytorch_encoder(x)
print(f"PyTorch Transformer 출력: {output.shape}")


# ============================================
# 9. Attention 시각화
# ============================================
print("\n[9] Attention 시각화")
print("-" * 40)

def visualize_attention(attention_weights, tokens=None):
    """Attention 가중치 시각화"""
    weights = attention_weights[0, 0].detach().numpy()  # 첫 배치, 첫 헤드

    plt.figure(figsize=(8, 6))
    plt.imshow(weights, cmap='Blues')
    plt.colorbar()

    if tokens:
        plt.xticks(range(len(tokens)), tokens, rotation=45)
        plt.yticks(range(len(tokens)), tokens)

    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.title('Attention Weights')
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=100)
    plt.close()
    print("그래프 저장: attention_visualization.png")

# 예시 attention 시각화
mha = MultiHeadAttention(d_model=64, num_heads=8)
x = torch.randn(1, 6, 64)
_, weights = mha(x, x, x)
visualize_attention(weights, ['The', 'cat', 'sat', 'on', 'mat', '.'])


# ============================================
# 10. Causal Mask (디코더용)
# ============================================
print("\n[10] Causal Mask")
print("-" * 40)

def create_causal_mask(size):
    """미래 토큰을 볼 수 없게 하는 마스크"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0  # True: 참조 가능, False: 마스킹

mask = create_causal_mask(5)
print(f"Causal Mask (5x5):\n{mask.int()}")


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("Attention & Transformer 정리")
print("=" * 60)

summary = """
Scaled Dot-Product Attention:
    scores = Q @ K.T / sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V

Multi-Head Attention:
    - 여러 헤드가 다른 관계 학습
    - 각 헤드: d_k = d_model / num_heads

Transformer Encoder:
    - Self-Attention + FFN
    - Residual + LayerNorm
    - Positional Encoding

PyTorch 내장:
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
    encoder = nn.TransformerEncoder(encoder_layer, num_layers)

핵심 하이퍼파라미터:
    - d_model: 모델 차원 (512)
    - num_heads: 헤드 수 (8)
    - d_ff: FFN 차원 (2048)
    - num_layers: 층 수 (6)
"""
print(summary)
print("=" * 60)
