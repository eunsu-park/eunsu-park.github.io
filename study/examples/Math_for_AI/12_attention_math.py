"""
Attention Mechanism Mathematics

This script demonstrates:
- Scaled dot-product attention from scratch
- Multi-head attention implementation
- Positional encoding (sinusoidal)
- Visualization of attention weights
- Comparison with PyTorch nn.MultiheadAttention

Attention mechanism:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Where:
- Q: Query matrix
- K: Key matrix
- V: Value matrix
- d_k: Dimension of keys (for scaling)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query matrix (batch, seq_len_q, d_k)
        K: Key matrix (batch, seq_len_k, d_k)
        V: Value matrix (batch, seq_len_k, d_v)
        mask: Optional mask (batch, seq_len_q, seq_len_k)

    Returns:
        output: Attention output (batch, seq_len_q, d_v)
        attention_weights: Attention weights (batch, seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]

    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)

    # Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Compute weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer.

    Multi-head attention allows the model to attend to information
    from different representation subspaces.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """
        Split last dimension into (num_heads, d_k).
        Transpose to (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """
        Inverse of split_heads.
        Transpose and reshape back to (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2)
        return x.reshape(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass.

        Args:
            Q, K, V: Input tensors (batch, seq_len, d_model)
            mask: Optional mask

        Returns:
            output: Attention output
            attention_weights: Attention weights (for visualization)
        """
        # Linear projections
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (batch, num_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (batch, num_heads, seq_len_v, d_k)

        # Apply attention
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads
        output = self.combine_heads(output)  # (batch, seq_len_q, d_model)

        # Final linear projection
        output = self.W_o(output)

        return output, attention_weights


def positional_encoding(seq_len, d_model):
    """
    Generate sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: Sequence length
        d_model: Model dimension

    Returns:
        Positional encoding matrix (seq_len, d_model)
    """
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return torch.FloatTensor(pe)


def visualize_positional_encoding():
    """
    Visualize sinusoidal positional encoding.
    """
    print("=== Positional Encoding ===\n")

    seq_len = 100
    d_model = 128

    pe = positional_encoding(seq_len, d_model)
    print(f"Positional encoding shape: {pe.shape}\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap
    im = axes[0].imshow(pe.numpy().T, cmap='RdBu', aspect='auto', interpolation='nearest')
    axes[0].set_title('Positional Encoding Heatmap')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Dimension')
    plt.colorbar(im, ax=axes[0])

    # Sample dimensions
    axes[1].plot(pe[:, 4].numpy(), label='dim 4')
    axes[1].plot(pe[:, 5].numpy(), label='dim 5')
    axes[1].plot(pe[:, 16].numpy(), label='dim 16')
    axes[1].plot(pe[:, 32].numpy(), label='dim 32')
    axes[1].set_title('Positional Encoding for Selected Dimensions')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Encoding Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/positional_encoding.png', dpi=150, bbox_inches='tight')
    print("Plot saved to /tmp/positional_encoding.png\n")
    plt.close()


def visualize_attention_weights():
    """
    Visualize attention weights for a simple example.
    """
    print("=== Attention Weights Visualization ===\n")

    # Create simple input
    seq_len = 10
    d_model = 64
    batch_size = 1

    # Random input
    X = torch.randn(batch_size, seq_len, d_model)

    # Add positional encoding
    pe = positional_encoding(seq_len, d_model)
    X = X + pe.unsqueeze(0)

    # Self-attention
    Q = K = V = X

    # Compute attention
    output, attn_weights = scaled_dot_product_attention(Q, K, V)

    print(f"Input shape: {X.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Output shape: {output.shape}\n")

    # Visualize attention matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(attn_weights[0].detach().numpy(), cmap='viridis',
                xticklabels=range(seq_len), yticklabels=range(seq_len),
                cbar_kws={'label': 'Attention Weight'}, ax=ax)
    ax.set_title('Self-Attention Weights')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')

    plt.tight_layout()
    plt.savefig('/tmp/attention_weights.png', dpi=150, bbox_inches='tight')
    print("Plot saved to /tmp/attention_weights.png\n")
    plt.close()


def compare_with_pytorch():
    """
    Compare custom implementation with PyTorch nn.MultiheadAttention.
    """
    print("=== Comparison with PyTorch nn.MultiheadAttention ===\n")

    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 8

    # Random input
    X = torch.randn(batch_size, seq_len, d_model)

    # Custom implementation
    custom_mha = MultiHeadAttention(d_model, num_heads)
    output_custom, attn_custom = custom_mha(X, X, X)

    # PyTorch implementation
    # Note: PyTorch expects (seq_len, batch, d_model), so we transpose
    pytorch_mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    # Copy weights from custom to PyTorch for fair comparison
    # (In practice, they would be trained, so outputs would differ)
    with torch.no_grad():
        pytorch_mha.in_proj_weight.copy_(torch.cat([
            custom_mha.W_q.weight,
            custom_mha.W_k.weight,
            custom_mha.W_v.weight
        ], dim=0))
        pytorch_mha.in_proj_bias.copy_(torch.cat([
            custom_mha.W_q.bias,
            custom_mha.W_k.bias,
            custom_mha.W_v.bias
        ], dim=0))
        pytorch_mha.out_proj.weight.copy_(custom_mha.W_o.weight)
        pytorch_mha.out_proj.bias.copy_(custom_mha.W_o.bias)

    output_pytorch, attn_pytorch = pytorch_mha(X, X, X, average_attn_weights=False)

    print(f"Custom output shape: {output_custom.shape}")
    print(f"PyTorch output shape: {output_pytorch.shape}")
    print(f"Outputs close: {torch.allclose(output_custom, output_pytorch, atol=1e-5)}\n")

    print(f"Custom attention shape: {attn_custom.shape}")
    print(f"PyTorch attention shape: {attn_pytorch.shape}\n")

    # The outputs should be very close if weights are identical
    diff = torch.abs(output_custom - output_pytorch).max().item()
    print(f"Max absolute difference: {diff:.6f}\n")


def demonstrate_causal_masking():
    """
    Demonstrate causal (autoregressive) masking for decoder-style attention.
    """
    print("=== Causal Masking (Decoder Attention) ===\n")

    seq_len = 6
    d_model = 32
    batch_size = 1

    X = torch.randn(batch_size, seq_len, d_model)
    Q = K = V = X

    # Create causal mask (lower triangular)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
    print("Causal mask (lower triangular):")
    print(causal_mask[0].numpy().astype(int))
    print()

    # Apply attention with mask
    output, attn_weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

    print(f"Attention weights with causal mask:")
    print(f"Shape: {attn_weights.shape}\n")

    # Visualize masked attention
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Without mask
    _, attn_no_mask = scaled_dot_product_attention(Q, K, V, mask=None)
    sns.heatmap(attn_no_mask[0].detach().numpy(), cmap='viridis',
                xticklabels=range(seq_len), yticklabels=range(seq_len),
                cbar_kws={'label': 'Attention Weight'}, ax=axes[0])
    axes[0].set_title('Attention Without Mask')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')

    # With causal mask
    sns.heatmap(attn_weights[0].detach().numpy(), cmap='viridis',
                xticklabels=range(seq_len), yticklabels=range(seq_len),
                cbar_kws={'label': 'Attention Weight'}, ax=axes[1])
    axes[1].set_title('Attention With Causal Mask')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')

    plt.tight_layout()
    plt.savefig('/tmp/causal_masking.png', dpi=150, bbox_inches='tight')
    print("Plot saved to /tmp/causal_masking.png\n")
    plt.close()


def demonstrate_cross_attention():
    """
    Demonstrate cross-attention (encoder-decoder attention).
    """
    print("=== Cross-Attention (Encoder-Decoder) ===\n")

    batch_size = 1
    encoder_seq_len = 8
    decoder_seq_len = 5
    d_model = 64

    # Encoder output (keys and values)
    encoder_output = torch.randn(batch_size, encoder_seq_len, d_model)

    # Decoder hidden states (queries)
    decoder_hidden = torch.randn(batch_size, decoder_seq_len, d_model)

    # Cross-attention: decoder queries attend to encoder keys/values
    Q = decoder_hidden
    K = V = encoder_output

    output, attn_weights = scaled_dot_product_attention(Q, K, V)

    print(f"Encoder sequence length: {encoder_seq_len}")
    print(f"Decoder sequence length: {decoder_seq_len}")
    print(f"Cross-attention weights shape: {attn_weights.shape}")
    print(f"(decoder_seq_len x encoder_seq_len)\n")

    # Visualize
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(attn_weights[0].detach().numpy(), cmap='viridis',
                xticklabels=range(encoder_seq_len),
                yticklabels=range(decoder_seq_len),
                cbar_kws={'label': 'Attention Weight'}, ax=ax)
    ax.set_title('Cross-Attention Weights (Decoder → Encoder)')
    ax.set_xlabel('Encoder Position')
    ax.set_ylabel('Decoder Position')

    plt.tight_layout()
    plt.savefig('/tmp/cross_attention.png', dpi=150, bbox_inches='tight')
    print("Plot saved to /tmp/cross_attention.png\n")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Attention Mechanism Mathematics")
    print("=" * 60)
    print()

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Run demonstrations
    visualize_positional_encoding()

    visualize_attention_weights()

    compare_with_pytorch()

    demonstrate_causal_masking()

    demonstrate_cross_attention()

    print("=" * 60)
    print("Summary:")
    print("- Attention allows dynamic weighting of input elements")
    print("- Scaled dot-product prevents gradient issues with large d_k")
    print("- Multi-head attention captures different representation subspaces")
    print("- Positional encoding injects sequence order information")
    print("- Causal masking enables autoregressive generation")
    print("- Cross-attention connects encoder and decoder")
    print("=" * 60)
