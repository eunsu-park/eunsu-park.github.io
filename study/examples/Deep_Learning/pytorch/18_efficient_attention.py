"""
18. Efficient Attention Mechanisms

Implementation of various attention mechanisms including:
- Standard Multi-Head Attention
- Flash Attention (via PyTorch 2.0+)
- Sparse Attention patterns
- Position encodings (Sinusoidal, RoPE, ALiBi)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
import time

print("=" * 60)
print("Efficient Attention Mechanisms")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")


# ============================================
# 1. Standard Multi-Head Attention
# ============================================
print("\n[1] Standard Multi-Head Attention")
print("-" * 40)


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention implementation"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, return_attention=False):
        batch_size, seq_len, _ = query.size()

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into heads: (batch, seq, d_model) -> (batch, heads, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.W_o(attention_output)

        if return_attention:
            return output, attention_weights
        return output


# Test
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 100, 512)
out = mha(x, x, x)
print(f"Input: {x.shape}")
print(f"Output: {out.shape}")


# ============================================
# 2. PyTorch 2.0+ Scaled Dot-Product Attention
# ============================================
print("\n[2] PyTorch Scaled Dot-Product Attention")
print("-" * 40)


class EfficientMultiHeadAttention(nn.Module):
    """Multi-Head Attention using PyTorch's scaled_dot_product_attention

    Automatically uses Flash Attention when available
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, mask=None, is_causal=False):
        batch_size, seq_len, _ = query.size()

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch's efficient attention
        dropout_p = self.dropout if self.training else 0.0
        attention_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )

        # Reshape back
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)

        return self.W_o(attention_output)


# Test efficient attention
efficient_mha = EfficientMultiHeadAttention(d_model=512, num_heads=8)
out_efficient = efficient_mha(x, x, x)
print(f"Efficient MHA output: {out_efficient.shape}")


# ============================================
# 3. Attention Complexity Analysis
# ============================================
print("\n[3] Complexity Analysis")
print("-" * 40)


def analyze_complexity(seq_lengths, d_model=512, num_heads=8):
    """Analyze time and memory complexity for different sequence lengths"""
    results = []

    for seq_len in seq_lengths:
        # Theoretical complexity
        time_complexity = seq_len ** 2 * d_model  # O(n^2 * d)
        space_complexity = seq_len ** 2 * num_heads  # attention matrix

        # Memory in GB (float32)
        memory_gb = space_complexity * 4 / (1024 ** 3)

        results.append({
            'seq_len': seq_len,
            'time_ops': time_complexity,
            'memory_gb': memory_gb
        })

    return results


seq_lengths = [128, 256, 512, 1024, 2048, 4096]
complexity = analyze_complexity(seq_lengths)

print("Sequence Length | Time Ops (M) | Memory (GB)")
print("-" * 45)
for r in complexity:
    print(f"{r['seq_len']:>14} | {r['time_ops']/1e6:>11.2f} | {r['memory_gb']:.4f}")


# ============================================
# 4. Sparse Attention Patterns
# ============================================
print("\n[4] Sparse Attention Patterns")
print("-" * 40)


def create_local_mask(seq_len, window_size):
    """Create local (sliding window) attention mask"""
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = 1
    return mask


def create_strided_mask(seq_len, stride):
    """Create strided attention mask"""
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        indices = list(range(0, seq_len, stride))
        mask[i, indices] = 1
    return mask


def create_causal_mask(seq_len):
    """Create causal (autoregressive) mask"""
    return torch.tril(torch.ones(seq_len, seq_len))


def create_bigbird_mask(seq_len, window_size=64, num_global=2, num_random=3):
    """Create BigBird sparse attention mask"""
    mask = torch.zeros(seq_len, seq_len)

    # Local attention
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = 1

    # Global tokens
    mask[:num_global, :] = 1
    mask[:, :num_global] = 1

    # Random connections
    for i in range(seq_len):
        random_indices = torch.randperm(seq_len)[:num_random]
        mask[i, random_indices] = 1

    return mask


# Visualize masks
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
seq_len = 64

masks = [
    ("Local (window=16)", create_local_mask(seq_len, 16)),
    ("Strided (stride=8)", create_strided_mask(seq_len, 8)),
    ("Causal", create_causal_mask(seq_len)),
    ("BigBird", create_bigbird_mask(seq_len, 16, 2, 3))
]

for ax, (name, mask) in zip(axes.flat, masks):
    ax.imshow(mask, cmap='Blues')
    ax.set_title(name)
    ax.axis('off')

plt.tight_layout()
plt.savefig('attention_masks.png', dpi=150)
plt.close()
print("Attention masks saved to attention_masks.png")


class LocalAttention(nn.Module):
    """Local (Sliding Window) Attention"""
    def __init__(self, d_model, num_heads, window_size=256):
        super().__init__()
        self.window_size = window_size
        self.attention = MultiHeadAttention(d_model, num_heads)

    def forward(self, x):
        seq_len = x.size(1)
        mask = create_local_mask(seq_len, self.window_size).to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
        return self.attention(x, x, x, mask=mask)


# ============================================
# 5. Position Encodings
# ============================================
print("\n[5] Position Encodings")
print("-" * 40)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Position Encoding (original Transformer)"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    """Learned Position Embeddings"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embedding(positions)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)

    Used in LLaMA, GPT-NeoX, etc.
    """
    def __init__(self, dim, max_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        self._set_cos_sin_cache(max_len)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len):
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed


class ALiBiPositionalBias(nn.Module):
    """Attention with Linear Biases (ALiBi)

    Used in MPT, BLOOM, etc.
    No learned parameters, good length extrapolation.
    """
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)

    def _get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return torch.tensor(get_slopes_power_of_2(n))
        else:
            closest_power = 2 ** math.floor(math.log2(n))
            return torch.tensor(
                get_slopes_power_of_2(closest_power) +
                get_slopes_power_of_2(2 * closest_power)[0::2][:n - closest_power]
            )

    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions.abs().unsqueeze(0).float()

        alibi = -self.slopes.unsqueeze(1).unsqueeze(1).to(device) * relative_positions

        return alibi


# Visualize position encodings
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Sinusoidal
sin_pe = SinusoidalPositionalEncoding(64)
pe_matrix = sin_pe.pe[0, :100, :].numpy()
axes[0].imshow(pe_matrix.T, aspect='auto', cmap='RdBu')
axes[0].set_title('Sinusoidal PE')
axes[0].set_xlabel('Position')
axes[0].set_ylabel('Dimension')

# ALiBi
alibi = ALiBiPositionalBias(8)
alibi_bias = alibi(100, 'cpu')
axes[1].imshow(alibi_bias[0].numpy(), aspect='auto', cmap='RdBu')
axes[1].set_title('ALiBi Bias (Head 0)')
axes[1].set_xlabel('Key Position')
axes[1].set_ylabel('Query Position')

# ALiBi all heads
axes[2].imshow(alibi_bias.mean(0).numpy(), aspect='auto', cmap='RdBu')
axes[2].set_title('ALiBi Bias (Mean)')
axes[2].set_xlabel('Key Position')
axes[2].set_ylabel('Query Position')

plt.tight_layout()
plt.savefig('position_encodings.png', dpi=150)
plt.close()
print("Position encodings visualization saved to position_encodings.png")


# ============================================
# 6. Attention with ALiBi
# ============================================
print("\n[6] Attention with ALiBi")
print("-" * 40)


class ALiBiMultiHeadAttention(nn.Module):
    """Multi-Head Attention with ALiBi position bias"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.alibi = ALiBiPositionalBias(num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()

        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores with ALiBi bias
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        alibi_bias = self.alibi(seq_len, query.device)
        scores = scores + alibi_bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)

        return self.W_o(attention_output)


# Test ALiBi attention
alibi_mha = ALiBiMultiHeadAttention(d_model=512, num_heads=8)
out_alibi = alibi_mha(x, x, x)
print(f"ALiBi MHA output: {out_alibi.shape}")


# ============================================
# 7. Benchmark
# ============================================
print("\n[7] Performance Benchmark")
print("-" * 40)


def benchmark_attention(attn_module, batch_size, seq_len, d_model, num_runs=10):
    """Benchmark attention module"""
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = attn_module(x, x, x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timing
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = attn_module(x, x, x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.time() - start) / num_runs

    return elapsed * 1000  # ms


# Only benchmark if not too slow
if device.type == 'cuda':
    print("\nBenchmarking on GPU...")
    configs = [
        (8, 256, 512),
        (8, 512, 512),
        (8, 1024, 512),
    ]

    results = []
    for batch, seq, dim in configs:
        standard_mha = MultiHeadAttention(dim, 8).to(device)
        efficient_mha = EfficientMultiHeadAttention(dim, 8).to(device)

        time_standard = benchmark_attention(standard_mha, batch, seq, dim)
        time_efficient = benchmark_attention(efficient_mha, batch, seq, dim)

        results.append({
            'config': f"({batch}, {seq}, {dim})",
            'standard': time_standard,
            'efficient': time_efficient,
            'speedup': time_standard / time_efficient
        })

    print("\nConfig            | Standard (ms) | Efficient (ms) | Speedup")
    print("-" * 60)
    for r in results:
        print(f"{r['config']:16} | {r['standard']:12.2f} | {r['efficient']:13.2f} | {r['speedup']:.2f}x")
else:
    print("GPU not available. Skipping benchmark.")


# ============================================
# 8. Attention Visualization
# ============================================
print("\n[8] Attention Visualization")
print("-" * 40)


def visualize_attention_weights(attention_weights, tokens=None, filename='attention_vis.png'):
    """Visualize attention weights heatmap"""
    num_heads = attention_weights.size(1)
    ncols = 4
    nrows = (num_heads + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.flatten()

    for head in range(num_heads):
        weights = attention_weights[0, head].cpu().detach().numpy()
        ax = axes[head]
        im = ax.imshow(weights, cmap='Blues')
        ax.set_title(f'Head {head}')

        if tokens and len(tokens) <= 10:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticklabels(tokens)
        else:
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')

    for i in range(num_heads, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Attention visualization saved to {filename}")


# Generate sample attention
mha_vis = MultiHeadAttention(d_model=64, num_heads=8)
x_vis = torch.randn(1, 10, 64)
_, attn_weights = mha_vis(x_vis, x_vis, x_vis, return_attention=True)
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'then', 'left', '.']
visualize_attention_weights(attn_weights, tokens, 'attention_heads.png')


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Efficient Attention Summary")
print("=" * 60)

summary = """
Key Concepts:
1. Standard Attention: O(n^2) time and space
2. Flash Attention: O(n^2) time, O(n) space via tiling
3. Sparse Attention: O(n * k) where k << n

Attention Patterns:
- Local: Sliding window attention
- Strided: Fixed-stride sparse pattern
- BigBird: Local + global + random
- Causal: Autoregressive masking

Position Encodings:
- Sinusoidal: Fixed, no parameters
- Learned: Trainable embeddings
- RoPE: Rotary, good relative position
- ALiBi: Linear bias, best extrapolation

PyTorch Tips:
1. Use F.scaled_dot_product_attention (PyTorch 2.0+)
2. Enable Flash Attention when possible
3. Use is_causal=True for autoregressive

Memory Comparison (seq_len=4096, heads=12):
- Standard: ~1.5 GB (attention matrix)
- Flash: ~0.1 GB (no full matrix storage)

Output Files:
- attention_masks.png: Sparse attention patterns
- position_encodings.png: PE visualizations
- attention_heads.png: Multi-head attention weights
"""
print(summary)
print("=" * 60)
