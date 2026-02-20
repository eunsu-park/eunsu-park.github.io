"""
Tensor Operations and Einstein Summation (einsum)

This script demonstrates:
- Tensor creation and manipulation in NumPy and PyTorch
- Einstein summation notation (einsum) for efficient operations
- Broadcasting rules and examples
- Numerical stability techniques

Einstein summation is a concise notation for tensor operations:
- Implicit mode: repeated indices are summed
- Explicit mode: specify output indices
- Used extensively in deep learning (attention, tensor contractions)
"""

import numpy as np
import torch
import torch.nn.functional as F


def tensor_basics():
    """
    Basic tensor operations in NumPy and PyTorch.
    """
    print("=== Tensor Basics ===\n")

    # NumPy
    np_array = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"NumPy array shape: {np_array.shape}")
    print(f"NumPy array:\n{np_array}\n")

    # PyTorch
    torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print(f"PyTorch tensor shape: {torch_tensor.shape}")
    print(f"PyTorch tensor:\n{torch_tensor}\n")

    # Conversion
    np_from_torch = torch_tensor.numpy()
    torch_from_np = torch.from_numpy(np_array)
    print(f"Converted NumPy → PyTorch → NumPy: {np.array_equal(np_array, np_from_torch)}\n")


def einsum_examples():
    """
    Comprehensive examples of einsum operations.

    Einsum notation:
    - 'i,i->': dot product (sum over i)
    - 'ij,jk->ik': matrix multiplication
    - 'ij->ji': transpose
    - 'ii->i': diagonal extraction
    - 'ij->': sum all elements
    """
    print("=== Einstein Summation (einsum) Examples ===\n")

    # Example 1: Dot product
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    dot_manual = np.sum(a * b)
    dot_einsum = np.einsum('i,i->', a, b)
    print(f"1. Dot product:")
    print(f"   Manual: {dot_manual}")
    print(f"   Einsum 'i,i->': {dot_einsum}\n")

    # Example 2: Matrix multiplication
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 5)

    matmul_manual = np.matmul(A, B)
    matmul_einsum = np.einsum('ij,jk->ik', A, B)
    print(f"2. Matrix multiplication (3x4 @ 4x5 = 3x5):")
    print(f"   Close match: {np.allclose(matmul_manual, matmul_einsum)}\n")

    # Example 3: Transpose
    C = np.random.randn(3, 4)
    transpose_manual = C.T
    transpose_einsum = np.einsum('ij->ji', C)
    print(f"3. Transpose:")
    print(f"   Close match: {np.allclose(transpose_manual, transpose_einsum)}\n")

    # Example 4: Trace (sum of diagonal)
    D = np.random.randn(4, 4)
    trace_manual = np.trace(D)
    trace_einsum = np.einsum('ii->', D)
    print(f"4. Trace (sum of diagonal):")
    print(f"   Manual: {trace_manual:.4f}")
    print(f"   Einsum 'ii->': {trace_einsum:.4f}\n")

    # Example 5: Batch matrix multiplication
    batch_A = np.random.randn(8, 3, 4)  # batch of 8 matrices (3x4)
    batch_B = np.random.randn(8, 4, 5)  # batch of 8 matrices (4x5)

    batch_matmul_manual = np.matmul(batch_A, batch_B)
    batch_matmul_einsum = np.einsum('bij,bjk->bik', batch_A, batch_B)
    print(f"5. Batch matrix multiplication (8 x [3x4 @ 4x5]):")
    print(f"   Close match: {np.allclose(batch_matmul_manual, batch_matmul_einsum)}\n")

    # Example 6: Outer product
    x = np.array([1, 2, 3])
    y = np.array([4, 5])
    outer_manual = np.outer(x, y)
    outer_einsum = np.einsum('i,j->ij', x, y)
    print(f"6. Outer product:")
    print(f"   Close match: {np.allclose(outer_manual, outer_einsum)}\n")

    # Example 7: Hadamard (element-wise) product and sum
    E = np.random.randn(3, 4)
    F = np.random.randn(3, 4)
    hadamard_sum = np.sum(E * F)
    hadamard_einsum = np.einsum('ij,ij->', E, F)
    print(f"7. Hadamard product and sum:")
    print(f"   Manual: {hadamard_sum:.4f}")
    print(f"   Einsum 'ij,ij->': {hadamard_einsum:.4f}\n")


def attention_with_einsum():
    """
    Implement scaled dot-product attention using einsum.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    This is more efficient than explicit reshaping and matmul.
    """
    print("=== Attention Mechanism with Einsum ===\n")

    batch_size = 2
    seq_len_q = 10
    seq_len_k = 12
    d_model = 64

    # Query, Key, Value
    Q = torch.randn(batch_size, seq_len_q, d_model)
    K = torch.randn(batch_size, seq_len_k, d_model)
    V = torch.randn(batch_size, seq_len_k, d_model)

    # Method 1: Manual matmul
    scores_manual = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_model)
    attn_weights_manual = F.softmax(scores_manual, dim=-1)
    output_manual = torch.matmul(attn_weights_manual, V)

    # Method 2: Using einsum
    scores_einsum = torch.einsum('bqd,bkd->bqk', Q, K) / np.sqrt(d_model)
    attn_weights_einsum = F.softmax(scores_einsum, dim=-1)
    output_einsum = torch.einsum('bqk,bkd->bqd', attn_weights_einsum, V)

    print(f"Attention output shape: {output_manual.shape}")
    print(f"Manual vs Einsum match: {torch.allclose(output_manual, output_einsum, atol=1e-6)}\n")

    # Multi-head attention (einsum is cleaner here)
    num_heads = 8
    d_k = d_model // num_heads

    # Split into multiple heads: (batch, seq, d_model) -> (batch, num_heads, seq, d_k)
    Q_heads = Q.reshape(batch_size, seq_len_q, num_heads, d_k).transpose(1, 2)
    K_heads = K.reshape(batch_size, seq_len_k, num_heads, d_k).transpose(1, 2)
    V_heads = V.reshape(batch_size, seq_len_k, num_heads, d_k).transpose(1, 2)

    # Compute attention per head using einsum
    scores_heads = torch.einsum('bhqd,bhkd->bhqk', Q_heads, K_heads) / np.sqrt(d_k)
    attn_weights_heads = F.softmax(scores_heads, dim=-1)
    output_heads = torch.einsum('bhqk,bhkd->bhqd', attn_weights_heads, V_heads)

    # Concatenate heads
    output_multihead = output_heads.transpose(1, 2).reshape(batch_size, seq_len_q, d_model)

    print(f"Multi-head attention output shape: {output_multihead.shape}")
    print(f"Attention weights shape (per head): {attn_weights_heads.shape}\n")


def broadcasting_examples():
    """
    Demonstrate broadcasting rules in NumPy and PyTorch.

    Broadcasting rules:
    1. If arrays have different ranks, prepend 1s to smaller rank
    2. Arrays are compatible if dimensions are equal or one is 1
    3. Result shape is element-wise maximum
    """
    print("=== Broadcasting Examples ===\n")

    # Example 1: Scalar + array
    a = np.array([1, 2, 3])
    b = 5
    result = a + b
    print(f"1. Scalar + array: {a} + {b} = {result}\n")

    # Example 2: Row vector + column vector (outer sum)
    row = np.array([[1, 2, 3]])  # (1, 3)
    col = np.array([[10], [20], [30]])  # (3, 1)
    result = row + col  # (3, 3)
    print(f"2. Row + column (outer sum):")
    print(f"{result}\n")

    # Example 3: Broadcasting in batch operations
    batch_data = np.random.randn(32, 10)  # 32 samples, 10 features
    feature_mean = np.mean(batch_data, axis=0, keepdims=True)  # (1, 10)
    centered_data = batch_data - feature_mean  # Broadcast (1, 10) to (32, 10)
    print(f"3. Batch normalization:")
    print(f"   Data shape: {batch_data.shape}")
    print(f"   Mean shape: {feature_mean.shape}")
    print(f"   Centered data shape: {centered_data.shape}")
    print(f"   Mean after centering: {np.mean(centered_data, axis=0)[:3]} (should be ~0)\n")

    # Example 4: Broadcasting with einsum
    A = np.random.randn(5, 3)
    b = np.random.randn(3)

    # Add bias using broadcasting
    result_broadcast = A + b

    # Add bias using einsum (less natural here, but possible)
    result_einsum = A + np.einsum('i->i', b)

    print(f"4. Matrix + vector broadcasting:")
    print(f"   Close match: {np.allclose(result_broadcast, result_einsum)}\n")


def numerical_stability():
    """
    Demonstrate numerical stability techniques.

    Common issues:
    - Overflow/underflow in exp()
    - Log of small numbers
    - Division by zero
    """
    print("=== Numerical Stability Techniques ===\n")

    # Problem 1: Softmax overflow
    logits = np.array([1000, 1001, 1002])  # Large values

    # Naive softmax (will overflow)
    try:
        naive_softmax = np.exp(logits) / np.sum(np.exp(logits))
        print(f"Naive softmax: {naive_softmax}")
    except:
        print("Naive softmax: OVERFLOW!")

    # Stable softmax (subtract max)
    max_logit = np.max(logits)
    stable_softmax = np.exp(logits - max_logit) / np.sum(np.exp(logits - max_logit))
    print(f"Stable softmax: {stable_softmax}")
    print(f"Sum: {np.sum(stable_softmax):.6f}\n")

    # Problem 2: Log-sum-exp trick
    x = np.array([1000, 1001, 1002])

    # Naive log(sum(exp(x))) will overflow
    max_x = np.max(x)
    log_sum_exp_stable = max_x + np.log(np.sum(np.exp(x - max_x)))
    print(f"Log-sum-exp (stable): {log_sum_exp_stable:.4f}")

    # Verify with scipy (if available)
    from scipy.special import logsumexp
    print(f"Scipy logsumexp: {logsumexp(x):.4f}\n")

    # Problem 3: Numerical precision in small differences
    a = torch.tensor([1e10, 1.0, 1e-10])
    b = torch.tensor([1e10, 1.0, 0.0])

    # Use torch.allclose with tolerance
    print(f"Arrays close (default tol): {torch.allclose(a, b)}")
    print(f"Arrays close (rtol=1e-5, atol=1e-8): {torch.allclose(a, b, rtol=1e-5, atol=1e-8)}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Tensor Operations and Einstein Summation")
    print("=" * 60)
    print()

    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # Run demonstrations
    tensor_basics()
    print()

    einsum_examples()
    print()

    attention_with_einsum()
    print()

    broadcasting_examples()
    print()

    numerical_stability()

    print("=" * 60)
    print("Summary:")
    print("- Einsum provides concise notation for tensor operations")
    print("- Broadcasting enables efficient element-wise operations")
    print("- Numerical stability is critical for deep learning")
    print("- PyTorch and NumPy have similar tensor APIs")
    print("=" * 60)
