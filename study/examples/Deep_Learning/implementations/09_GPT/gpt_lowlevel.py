"""
PyTorch Low-Level GPT-2 구현

nanoGPT 스타일의 간결한 구현
Pre-LayerNorm, Causal Attention, Weight Tying
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """GPT-2 설정"""
    vocab_size: int = 50257
    block_size: int = 1024  # max sequence length
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Causal Self-Attention (Masked Multi-Head Attention)"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Q, K, V를 하나의 projection으로 (효율성)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask (미래 토큰 참조 차단)
        # register_buffer: 학습하지 않지만 state_dict에 포함
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (batch, seq_len, n_embd)
            use_cache: KV cache 사용 여부 (생성 시)
            past_kv: 이전 K, V 캐시

        Returns:
            y: (batch, seq_len, n_embd)
            present_kv: 현재 K, V (캐싱용)
        """
        B, T, C = x.shape

        # Q, K, V 계산 (하나의 matmul로)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Multi-head reshape: (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # KV Cache 처리 (생성 시 효율화)
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        # Attention scores
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T_kv) → (B, n_head, T, T_kv)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask 적용
        T_kv = k.size(2)
        # 현재 위치부터 시작하는 마스크 (KV cache 고려)
        mask = self.bias[:, :, T_kv - T:T_kv, :T_kv]
        att = att.masked_fill(mask == 0, float('-inf'))

        # Softmax + Dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = torch.matmul(att, v)  # (B, n_head, T, head_dim)

        # Reshape back: (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + dropout
        y = self.resid_dropout(self.c_proj(y))

        return y, present_kv


class MLP(nn.Module):
    """Feed-Forward Network (GPT-2 스타일)"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # 4x 확장
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x, approximate='tanh')  # GPT-2는 tanh approximation
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer Block (Pre-LN)"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN + Residual
        attn_out, present_kv = self.attn(
            self.ln_1(x), use_cache=use_cache, past_kv=past_kv
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


class GPT(nn.Module):
    """GPT-2 Model"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),  # token embedding
            'wpe': nn.Embedding(config.block_size, config.n_embd),  # position embedding
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })

        # LM Head (weight tying with wte)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer['wte'].weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[list] = None
    ):
        """
        Args:
            idx: (batch, seq_len) 토큰 인덱스
            targets: (batch, seq_len) 타겟 (학습 시)
            use_cache: KV cache 사용
            past_key_values: 이전 KV cache 리스트

        Returns:
            logits, loss, present_key_values
        """
        device = idx.device
        B, T = idx.shape

        # Position IDs
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2)
            pos = torch.arange(past_length, past_length + T, device=device)
        else:
            pos = torch.arange(0, T, device=device)

        # Embeddings
        tok_emb = self.transformer['wte'](idx)  # (B, T, n_embd)
        pos_emb = self.transformer['wpe'](pos)  # (T, n_embd)
        x = self.transformer['drop'](tok_emb + pos_emb)

        # Transformer blocks
        present_key_values = [] if use_cache else None

        for i, block in enumerate(self.transformer['h']):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, use_cache=use_cache, past_kv=past_kv)
            if use_cache:
                present_key_values.append(present_kv)

        # Final layer norm
        x = self.transformer['ln_f'](x)

        # LM Head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )

        return {
            'logits': logits,
            'loss': loss,
            'past_key_values': present_key_values
        }

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        텍스트 생성

        Args:
            idx: (batch, seq_len) 시작 토큰
            max_new_tokens: 생성할 토큰 수
            temperature: 샘플링 온도
            top_k: Top-K 샘플링
            top_p: Nucleus (Top-P) 샘플링
            use_cache: KV cache 사용

        Returns:
            idx: (batch, seq_len + max_new_tokens)
        """
        past_key_values = None

        for _ in range(max_new_tokens):
            # 컨텍스트 자르기 (block_size 초과 방지)
            if past_key_values is None:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            else:
                idx_cond = idx[:, -1:]  # 마지막 토큰만 (캐시 사용 시)

            # Forward
            outputs = self(idx_cond, use_cache=use_cache, past_key_values=past_key_values)
            logits = outputs['logits'][:, -1, :]  # 마지막 위치
            past_key_values = outputs['past_key_values']

            # Temperature
            logits = logits / temperature

            # Top-K 필터링
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-P (Nucleus) 필터링
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Top-P 초과하는 토큰 제거
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # 샘플링
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # 결과에 추가
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# 테스트
if __name__ == "__main__":
    print("=== GPT-2 Low-Level Implementation Test ===\n")

    # GPT-2 Small 설정
    config = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1
    )

    # 모델 생성
    model = GPT(config)

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Expected ~117M for GPT-2 Small\n")

    # 테스트 입력
    batch_size, seq_len = 2, 64
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward
    outputs = model(idx, targets=targets)

    print("Forward pass:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")

    # 생성 테스트
    print("\n=== Generation Test ===")
    start_tokens = torch.tensor([[50256]])  # <|endoftext|>

    generated = model.generate(
        start_tokens,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50
    )

    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()[:25]}...")

    # KV Cache 테스트
    print("\n=== KV Cache Test ===")
    import time

    # Without cache
    torch.manual_seed(42)
    start = time.time()
    gen_no_cache = model.generate(start_tokens, max_new_tokens=50, use_cache=False)
    time_no_cache = time.time() - start

    # With cache
    torch.manual_seed(42)
    start = time.time()
    gen_with_cache = model.generate(start_tokens, max_new_tokens=50, use_cache=True)
    time_with_cache = time.time() - start

    print(f"Without cache: {time_no_cache:.3f}s")
    print(f"With cache: {time_with_cache:.3f}s")
    print(f"Speedup: {time_no_cache / time_with_cache:.2f}x")

    # 결과 일치 확인
    assert torch.equal(gen_no_cache, gen_with_cache), "Cache results should match!"
    print("Cache results match!")

    print("\nAll tests passed!")
