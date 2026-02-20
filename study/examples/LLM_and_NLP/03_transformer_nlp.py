"""
03. Transformer NLP 예제

Transformer 아키텍처 복습 및 NLP 적용
"""

print("=" * 60)
print("Transformer NLP")
print("=" * 60)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    # ============================================
    # 1. Self-Attention
    # ============================================
    print("\n[1] Self-Attention")
    print("-" * 40)

    def scaled_dot_product_attention(Q, K, V, mask=None):
        """Scaled Dot-Product Attention"""
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    # 테스트
    batch, seq_len, d_model = 2, 5, 64
    Q = torch.randn(batch, seq_len, d_model)
    K = torch.randn(batch, seq_len, d_model)
    V = torch.randn(batch, seq_len, d_model)

    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"입력 shape: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"출력 shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")


    # ============================================
    # 2. Multi-Head Attention
    # ============================================
    print("\n[2] Multi-Head Attention")
    print("-" * 40)

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

        def forward(self, x, mask=None):
            batch_size, seq_len, _ = x.shape

            Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

            attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

            return self.W_o(attn_output)

    mha = MultiHeadAttention(d_model=64, num_heads=8)
    x = torch.randn(2, 10, 64)
    output = mha(x)
    print(f"Multi-Head Attention: {x.shape} → {output.shape}")


    # ============================================
    # 3. Causal Mask (GPT 스타일)
    # ============================================
    print("\n[3] Causal Mask")
    print("-" * 40)

    def create_causal_mask(seq_len):
        """미래 토큰 마스킹"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask

    mask = create_causal_mask(5)
    print("Causal Mask (5x5):")
    print(mask)


    # ============================================
    # 4. Positional Encoding
    # ============================================
    print("\n[4] Positional Encoding")
    print("-" * 40)

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    pe = PositionalEncoding(d_model=64)
    x = torch.randn(2, 10, 64)
    output = pe(x)
    print(f"Positional Encoding: {x.shape} → {output.shape}")


    # ============================================
    # 5. Transformer Encoder Block
    # ============================================
    print("\n[5] Transformer Encoder Block")
    print("-" * 40)

    class TransformerBlock(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
            super().__init__()
            self.attention = MultiHeadAttention(d_model, num_heads)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            # Self-Attention + Residual
            attn_out = self.attention(x, mask)
            x = self.norm1(x + self.dropout(attn_out))

            # FFN + Residual
            ffn_out = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_out))

            return x

    block = TransformerBlock(d_model=64, num_heads=8, d_ff=256)
    x = torch.randn(2, 10, 64)
    output = block(x)
    print(f"Transformer Block: {x.shape} → {output.shape}")


    # ============================================
    # 6. 텍스트 분류 Transformer
    # ============================================
    print("\n[6] 텍스트 분류 Transformer")
    print("-" * 40)

    class TransformerClassifier(nn.Module):
        def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = PositionalEncoding(d_model)
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_model * 4)
                for _ in range(num_layers)
            ])
            self.fc = nn.Linear(d_model, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            x = self.pos_encoding(x)
            for block in self.blocks:
                x = block(x)
            x = x.mean(dim=1)  # 평균 풀링
            return self.fc(x)

    model = TransformerClassifier(
        vocab_size=10000, d_model=128, num_heads=4, num_layers=2, num_classes=2
    )
    x = torch.randint(0, 10000, (4, 32))  # (batch, seq)
    output = model(x)
    print(f"입력: {x.shape}")
    print(f"출력: {output.shape}")
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")


    # ============================================
    # 7. PyTorch 내장 Transformer
    # ============================================
    print("\n[7] PyTorch 내장 Transformer")
    print("-" * 40)

    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    x = torch.randn(32, 100, 512)  # (batch, seq, d_model)
    output = encoder(x)
    print(f"PyTorch Transformer: {x.shape} → {output.shape}")


    # ============================================
    # 정리
    # ============================================
    print("\n" + "=" * 60)
    print("Transformer 정리")
    print("=" * 60)

    summary = """
핵심 구성요소:
    1. Self-Attention: Q @ K.T / sqrt(d_k) → softmax → @ V
    2. Multi-Head: 여러 헤드로 분할 후 결합
    3. Positional Encoding: 위치 정보 추가
    4. FFN: Linear → GELU → Linear
    5. Residual + LayerNorm

BERT vs GPT:
    - BERT: 양방향 (인코더), 마스크 없음
    - GPT: 단방향 (디코더), Causal Mask
"""
    print(summary)

except ImportError as e:
    print(f"PyTorch 미설치: {e}")
