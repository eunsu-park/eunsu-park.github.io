"""
Transformer - PyTorch Low-Level 구현

이 파일은 Transformer를 PyTorch 기본 연산만으로 구현합니다.
nn.TransformerEncoder, nn.MultiheadAttention 등 고수준 API를 사용하지 않고
직접 attention과 FFN을 구현합니다.

논문: "Attention Is All You Need" (Vaswani et al., 2017)

학습 목표:
1. Scaled Dot-Product Attention 구현
2. Multi-Head Attention 구현
3. Positional Encoding 구현
4. Encoder/Decoder 블록 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention

    Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Args:
        query: (batch, n_heads, seq_len, d_k)
        key: (batch, n_heads, seq_len, d_k)
        value: (batch, n_heads, seq_len, d_v)
        mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
        dropout: Dropout layer

    Returns:
        output: (batch, n_heads, seq_len, d_v)
        attention_weights: (batch, n_heads, seq_len, seq_len)
    """
    d_k = query.size(-1)

    # 1. QK^T: Query와 Key의 유사도 계산
    # (batch, heads, seq, d_k) @ (batch, heads, d_k, seq) → (batch, heads, seq, seq)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # 2. Scaling: √d_k로 나눔 (softmax 안정성)
    scores = scores / math.sqrt(d_k)

    # 3. Masking (optional)
    if mask is not None:
        # mask가 True인 위치를 -inf로 설정 (softmax 후 0이 됨)
        scores = scores.masked_fill(mask, float('-inf'))

    # 4. Softmax: 확률 분포로 변환
    attention_weights = F.softmax(scores, dim=-1)

    # 5. Dropout (학습 시)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # 6. Weighted sum of values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttentionLowLevel(nn.Module):
    """
    Multi-Head Attention (Low-Level 구현)

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

    nn.MultiheadAttention을 사용하지 않고 직접 구현
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: 모델 차원
            n_heads: attention head 수
            dropout: dropout 비율
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 각 head의 차원

        # Q, K, V projection (합쳐서 한 번에)
        # nn.Linear 대신 직접 파라미터 관리도 가능
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: (batch, seq_len) or (batch, seq_len, seq_len)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size = query.size(0)

        # 1. Linear projections
        # (batch, seq, d_model) → (batch, seq, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Split into multiple heads
        # (batch, seq, d_model) → (batch, seq, n_heads, d_k) → (batch, n_heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 3. Mask 차원 조정 (broadcasting을 위해)
        if mask is not None:
            if mask.dim() == 2:
                # (batch, seq) → (batch, 1, 1, seq)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (batch, seq, seq) → (batch, 1, seq, seq)
                mask = mask.unsqueeze(1)

        # 4. Attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask, self.dropout)

        # 5. Concat heads
        # (batch, n_heads, seq, d_k) → (batch, seq, n_heads, d_k) → (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # 6. Output projection
        output = self.W_o(attn_output)

        return output


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 위치 인코딩 미리 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # 학습되지 않는 버퍼로 등록
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x + PE: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class FeedForwardLowLevel(nn.Module):
    """
    Position-wise Feed-Forward Network

    FFN(x) = GELU(xW_1 + b_1)W_2 + b_2

    보통 d_ff = 4 * d_model (expansion)
    """

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        # GELU activation (원래 논문은 ReLU지만 현대는 GELU 선호)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block

    구조:
    x → LayerNorm → MultiHeadAttention → Dropout → Add(x) →
      → LayerNorm → FeedForward → Dropout → Add(x) → output
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttentionLowLevel(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardLowLevel(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: padding mask

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Pre-norm (현대적 방식, 원래 논문은 Post-norm)
        # Self-Attention + Residual
        normed = self.norm1(x)
        attn_out = self.attention(normed, normed, normed, mask)
        x = x + self.dropout(attn_out)

        # Feed-Forward + Residual
        normed = self.norm2(x)
        ff_out = self.feed_forward(normed)
        x = x + self.dropout(ff_out)

        return x


class TransformerDecoderBlock(nn.Module):
    """
    Transformer Decoder Block

    구조:
    x → LayerNorm → MaskedSelfAttention → Add(x) →
      → LayerNorm → CrossAttention(encoder_output) → Add(x) →
      → LayerNorm → FeedForward → Add(x) → output
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadAttentionLowLevel(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttentionLowLevel(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardLowLevel(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: torch.Tensor = None,
        cross_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: decoder input (batch, tgt_len, d_model)
            encoder_output: encoder output (batch, src_len, d_model)
            self_mask: causal mask for self-attention
            cross_mask: padding mask for cross-attention

        Returns:
            output: (batch, tgt_len, d_model)
        """
        # Masked Self-Attention
        normed = self.norm1(x)
        attn_out = self.self_attention(normed, normed, normed, self_mask)
        x = x + self.dropout(attn_out)

        # Cross-Attention (query: decoder, key/value: encoder)
        normed = self.norm2(x)
        cross_out = self.cross_attention(normed, encoder_output, encoder_output, cross_mask)
        x = x + self.dropout(cross_out)

        # Feed-Forward
        normed = self.norm3(x)
        ff_out = self.feed_forward(normed)
        x = x + self.dropout(ff_out)

        return x


class TransformerLowLevel(nn.Module):
    """
    전체 Transformer 모델 (Encoder-Decoder)

    번역, 요약 등 seq2seq 태스크용
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Scaling factor for embeddings
        self.scale = math.sqrt(d_model)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Causal mask: 미래 토큰을 못 보게 하는 마스크

        Returns:
            mask: (seq_len, seq_len) - True = 마스킹
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encoder forward pass

        Args:
            src: source tokens (batch, src_len)
            src_mask: padding mask

        Returns:
            encoder_output: (batch, src_len, d_model)
        """
        # Embedding + Positional Encoding
        x = self.src_embedding(src) * self.scale
        x = self.pos_encoding(x)

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        x = self.encoder_norm(x)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Decoder forward pass

        Args:
            tgt: target tokens (batch, tgt_len)
            encoder_output: (batch, src_len, d_model)
            tgt_mask: causal mask
            memory_mask: cross-attention mask

        Returns:
            decoder_output: (batch, tgt_len, d_model)
        """
        # Embedding + Positional Encoding
        x = self.tgt_embedding(tgt) * self.scale
        x = self.pos_encoding(x)

        # Causal mask
        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(tgt.size(1), tgt.device)

        # Decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, memory_mask)

        x = self.decoder_norm(x)
        return x

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        전체 forward pass

        Args:
            src: source tokens (batch, src_len)
            tgt: target tokens (batch, tgt_len)

        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        logits = self.output_projection(decoder_output)
        return logits


def main():
    """테스트 실행"""
    print("=" * 60)
    print("Transformer - PyTorch Low-Level 구현")
    print("=" * 60)

    # 설정
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 256
    n_heads = 8
    n_layers = 4
    batch_size = 2
    src_len = 10
    tgt_len = 8

    # 모델 생성
    model = TransformerLowLevel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_layers,
        n_decoder_layers=n_layers,
    )

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n모델 파라미터 수: {total_params:,}")

    # 더미 데이터
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

    print(f"\nInput shapes:")
    print(f"  Source: {src.shape}")
    print(f"  Target: {tgt.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt)

    print(f"\nOutput shape: {logits.shape}")
    print(f"  Expected: (batch={batch_size}, tgt_len={tgt_len}, vocab={tgt_vocab_size})")

    # Attention 패턴 시각화 (optional)
    print("\n테스트 완료!")


if __name__ == "__main__":
    main()
