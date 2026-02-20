"""
Foundation Models - Quantization Techniques

Demonstrates INT8/INT4 quantization for model compression.
Implements symmetric and asymmetric quantization schemes.
Shows quantization error analysis and calibration concepts.

Requires: PyTorch, NumPy
"""

import torch
import numpy as np


def symmetric_quantize(tensor, num_bits=8):
    """
    Symmetric quantization: maps [-α, α] to [-2^(b-1), 2^(b-1)-1]

    Args:
        tensor: Input tensor
        num_bits: Number of bits (8 for INT8, 4 for INT4)

    Returns:
        Tuple of (quantized tensor, scale factor)
    """
    # Determine quantization range
    q_min = -(2 ** (num_bits - 1))
    q_max = 2 ** (num_bits - 1) - 1

    # Find scale: α / 2^(b-1)
    alpha = tensor.abs().max()
    scale = alpha / q_max

    # Quantize: round(x / scale)
    quantized = torch.round(tensor / scale).clamp(q_min, q_max)

    return quantized, scale


def symmetric_dequantize(quantized, scale):
    """
    Dequantize symmetric quantized tensor.

    Args:
        quantized: Quantized tensor
        scale: Scale factor from quantization

    Returns:
        Dequantized tensor
    """
    return quantized * scale


def asymmetric_quantize(tensor, num_bits=8):
    """
    Asymmetric quantization: maps [min, max] to [0, 2^b - 1]

    Args:
        tensor: Input tensor
        num_bits: Number of bits

    Returns:
        Tuple of (quantized tensor, scale, zero_point)
    """
    # Quantization range
    q_min = 0
    q_max = 2 ** num_bits - 1

    # Calibration: find min and max
    r_min = tensor.min()
    r_max = tensor.max()

    # Compute scale and zero point
    scale = (r_max - r_min) / (q_max - q_min)
    zero_point = q_min - torch.round(r_min / scale)
    zero_point = zero_point.clamp(q_min, q_max)

    # Quantize: round(x / scale + zero_point)
    quantized = torch.round(tensor / scale + zero_point).clamp(q_min, q_max)

    return quantized, scale, zero_point


def asymmetric_dequantize(quantized, scale, zero_point):
    """
    Dequantize asymmetric quantized tensor.

    Args:
        quantized: Quantized tensor
        scale: Scale factor
        zero_point: Zero point offset

    Returns:
        Dequantized tensor
    """
    return (quantized - zero_point) * scale


def compute_quantization_error(original, dequantized):
    """
    Compute quantization error metrics.

    Args:
        original: Original tensor
        dequantized: Dequantized tensor

    Returns:
        Dictionary of error metrics
    """
    error = (original - dequantized).abs()

    metrics = {
        'max_error': error.max().item(),
        'mean_error': error.mean().item(),
        'mse': ((original - dequantized) ** 2).mean().item(),
        'snr_db': 10 * torch.log10((original ** 2).mean() / ((error ** 2).mean() + 1e-10)).item(),
    }

    return metrics


def per_channel_quantize(tensor, num_bits=8, dim=0):
    """
    Per-channel (per-output) quantization for better accuracy.

    Args:
        tensor: Input tensor (e.g., weight matrix)
        num_bits: Number of bits
        dim: Channel dimension

    Returns:
        Tuple of (quantized, scales)
    """
    # Move channel dim to front
    tensor = tensor.transpose(0, dim) if dim != 0 else tensor
    out_channels = tensor.shape[0]

    quantized_channels = []
    scales = []

    q_min = -(2 ** (num_bits - 1))
    q_max = 2 ** (num_bits - 1) - 1

    # Quantize each channel independently
    for i in range(out_channels):
        channel = tensor[i]
        alpha = channel.abs().max()
        scale = alpha / q_max

        quant_channel = torch.round(channel / scale).clamp(q_min, q_max)
        quantized_channels.append(quant_channel)
        scales.append(scale)

    quantized = torch.stack(quantized_channels, dim=0)

    # Transpose back
    quantized = quantized.transpose(0, dim) if dim != 0 else quantized

    return quantized, torch.tensor(scales)


def per_channel_dequantize(quantized, scales, dim=0):
    """Dequantize per-channel quantized tensor."""
    quantized = quantized.transpose(0, dim) if dim != 0 else quantized
    out_channels = quantized.shape[0]

    dequantized_channels = []
    for i in range(out_channels):
        channel = quantized[i] * scales[i]
        dequantized_channels.append(channel)

    dequantized = torch.stack(dequantized_channels, dim=0)
    dequantized = dequantized.transpose(0, dim) if dim != 0 else dequantized

    return dequantized


# ============================================================
# Demonstrations
# ============================================================

def demo_symmetric_quantization():
    """Demonstrate symmetric quantization."""
    print("=" * 60)
    print("DEMO 1: Symmetric Quantization")
    print("=" * 60)

    # Create random tensor
    torch.manual_seed(42)
    tensor = torch.randn(1000) * 10

    print(f"\nOriginal tensor:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Range: [{tensor.min():.2f}, {tensor.max():.2f}]")
    print(f"  Mean: {tensor.mean():.2f}, Std: {tensor.std():.2f}")

    # Quantize to INT8 and INT4
    for num_bits in [8, 4]:
        quantized, scale = symmetric_quantize(tensor, num_bits=num_bits)
        dequantized = symmetric_dequantize(quantized, scale)

        metrics = compute_quantization_error(tensor, dequantized)

        print(f"\nINT{num_bits} Quantization:")
        print(f"  Scale: {scale:.6f}")
        print(f"  Quantized range: [{quantized.min():.0f}, {quantized.max():.0f}]")
        print(f"  Max error: {metrics['max_error']:.4f}")
        print(f"  Mean error: {metrics['mean_error']:.4f}")
        print(f"  SNR: {metrics['snr_db']:.2f} dB")


def demo_asymmetric_quantization():
    """Demonstrate asymmetric quantization."""
    print("\n" + "=" * 60)
    print("DEMO 2: Asymmetric Quantization")
    print("=" * 60)

    # Create tensor with asymmetric range (e.g., activations)
    torch.manual_seed(42)
    tensor = torch.relu(torch.randn(1000) * 5 + 2)  # Mostly positive

    print(f"\nOriginal tensor (ReLU activations):")
    print(f"  Range: [{tensor.min():.2f}, {tensor.max():.2f}]")
    print(f"  Mean: {tensor.mean():.2f}, Std: {tensor.std():.2f}")

    # Compare symmetric vs asymmetric
    for method in ['symmetric', 'asymmetric']:
        if method == 'symmetric':
            quantized, scale = symmetric_quantize(tensor, num_bits=8)
            dequantized = symmetric_dequantize(quantized, scale)
            print(f"\nSymmetric INT8:")
            print(f"  Scale: {scale:.6f}")
        else:
            quantized, scale, zero_point = asymmetric_quantize(tensor, num_bits=8)
            dequantized = asymmetric_dequantize(quantized, scale, zero_point)
            print(f"\nAsymmetric INT8:")
            print(f"  Scale: {scale:.6f}, Zero point: {zero_point:.0f}")

        metrics = compute_quantization_error(tensor, dequantized)
        print(f"  Max error: {metrics['max_error']:.4f}")
        print(f"  Mean error: {metrics['mean_error']:.4f}")
        print(f"  SNR: {metrics['snr_db']:.2f} dB")


def demo_per_channel_quantization():
    """Demonstrate per-channel quantization for weight matrices."""
    print("\n" + "=" * 60)
    print("DEMO 3: Per-Channel Quantization")
    print("=" * 60)

    # Simulate weight matrix with different channel statistics
    torch.manual_seed(42)
    weight = torch.randn(256, 512)

    # Make some channels have larger magnitude
    weight[:64] *= 10
    weight[64:128] *= 0.1

    print(f"\nWeight matrix:")
    print(f"  Shape: {weight.shape}")
    print(f"  Overall range: [{weight.min():.2f}, {weight.max():.2f}]")
    print(f"  Channel 0 range: [{weight[0].min():.2f}, {weight[0].max():.2f}]")
    print(f"  Channel 100 range: [{weight[100].min():.2f}, {weight[100].max():.2f}]")

    # Per-tensor quantization
    quant_tensor, scale_tensor = symmetric_quantize(weight, num_bits=8)
    dequant_tensor = symmetric_dequantize(quant_tensor, scale_tensor)
    metrics_tensor = compute_quantization_error(weight, dequant_tensor)

    print(f"\nPer-tensor quantization:")
    print(f"  Single scale: {scale_tensor:.6f}")
    print(f"  Max error: {metrics_tensor['max_error']:.4f}")
    print(f"  SNR: {metrics_tensor['snr_db']:.2f} dB")

    # Per-channel quantization
    quant_channel, scales_channel = per_channel_quantize(weight, num_bits=8, dim=0)
    dequant_channel = per_channel_dequantize(quant_channel, scales_channel, dim=0)
    metrics_channel = compute_quantization_error(weight, dequant_channel)

    print(f"\nPer-channel quantization:")
    print(f"  Scale range: [{scales_channel.min():.6f}, {scales_channel.max():.6f}]")
    print(f"  Max error: {metrics_channel['max_error']:.4f}")
    print(f"  SNR: {metrics_channel['snr_db']:.2f} dB")
    print(f"  Improvement: {metrics_channel['snr_db'] - metrics_tensor['snr_db']:.2f} dB")


def demo_bit_depth_comparison():
    """Compare different quantization bit depths."""
    print("\n" + "=" * 60)
    print("DEMO 4: Bit Depth Comparison")
    print("=" * 60)

    torch.manual_seed(42)
    tensor = torch.randn(10000) * 5

    print(f"\nOriginal tensor: {tensor.numel()} values")
    print(f"  FP32 size: {tensor.numel() * 4 / 1024:.2f} KB")

    print("\n" + "-" * 60)
    print(f"{'Bits':<8} {'Size (KB)':<12} {'SNR (dB)':<12} {'Compression':<12}")
    print("-" * 60)

    for num_bits in [2, 4, 8, 16]:
        quantized, scale = symmetric_quantize(tensor, num_bits=num_bits)
        dequantized = symmetric_dequantize(quantized, scale)

        metrics = compute_quantization_error(tensor, dequantized)

        # Approximate size (quantized values + scale)
        size_kb = (tensor.numel() * num_bits / 8 + 4) / 1024
        compression = (tensor.numel() * 4) / (tensor.numel() * num_bits / 8 + 4)

        print(f"{num_bits:<8} {size_kb:<12.2f} {metrics['snr_db']:<12.2f} {compression:<12.2f}x")


def demo_calibration():
    """Demonstrate calibration for quantization."""
    print("\n" + "=" * 60)
    print("DEMO 5: Calibration Strategies")
    print("=" * 60)

    # Generate data with outliers
    torch.manual_seed(42)
    tensor = torch.randn(1000) * 2
    tensor[torch.randint(0, 1000, (10,))] = torch.randn(10) * 50  # Outliers

    print(f"\nData statistics:")
    print(f"  Min: {tensor.min():.2f}, Max: {tensor.max():.2f}")
    print(f"  Mean: {tensor.mean():.2f}, Std: {tensor.std():.2f}")
    print(f"  99th percentile: {torch.quantile(tensor.abs(), 0.99):.2f}")

    # Strategy 1: Min-max (uses full range)
    quant1, scale1 = symmetric_quantize(tensor, num_bits=8)
    dequant1 = symmetric_dequantize(quant1, scale1)
    metrics1 = compute_quantization_error(tensor, dequant1)

    print(f"\nMin-max calibration:")
    print(f"  Clipping range: ±{tensor.abs().max():.2f}")
    print(f"  SNR: {metrics1['snr_db']:.2f} dB")

    # Strategy 2: Percentile-based (clip outliers)
    percentile = 99
    clip_value = torch.quantile(tensor.abs(), percentile / 100)
    tensor_clipped = tensor.clamp(-clip_value, clip_value)

    quant2, scale2 = symmetric_quantize(tensor_clipped, num_bits=8)
    dequant2 = symmetric_dequantize(quant2, scale2)
    metrics2 = compute_quantization_error(tensor_clipped, dequant2)

    print(f"\n{percentile}th percentile calibration:")
    print(f"  Clipping range: ±{clip_value:.2f}")
    print(f"  SNR: {metrics2['snr_db']:.2f} dB")
    print(f"  Values clipped: {(tensor.abs() > clip_value).sum().item()}")


def demo_quantized_matmul():
    """Demonstrate quantized matrix multiplication."""
    print("\n" + "=" * 60)
    print("DEMO 6: Quantized Matrix Multiplication")
    print("=" * 60)

    # Create matrices
    torch.manual_seed(42)
    A = torch.randn(64, 128) * 2
    B = torch.randn(128, 256) * 3

    # FP32 matmul
    C_fp32 = torch.matmul(A, B)

    print(f"\nMatrix shapes: A {A.shape} @ B {B.shape} = C {C_fp32.shape}")

    # Quantize matrices
    A_quant, A_scale = symmetric_quantize(A, num_bits=8)
    B_quant, B_scale = symmetric_quantize(B, num_bits=8)

    # Quantized matmul (in practice, use INT8 kernel)
    # Here we dequantize for demonstration
    A_dequant = symmetric_dequantize(A_quant, A_scale)
    B_dequant = symmetric_dequantize(B_quant, B_scale)
    C_quant = torch.matmul(A_dequant, B_dequant)

    # Compare results
    error = (C_fp32 - C_quant).abs()

    print(f"\nResults:")
    print(f"  FP32 result range: [{C_fp32.min():.2f}, {C_fp32.max():.2f}]")
    print(f"  INT8 result range: [{C_quant.min():.2f}, {C_quant.max():.2f}]")
    print(f"  Max error: {error.max():.4f}")
    print(f"  Mean error: {error.mean():.4f}")
    print(f"  Relative error: {(error / C_fp32.abs()).mean():.4%}")

    # Memory savings
    fp32_size = (A.numel() + B.numel() + C_fp32.numel()) * 4
    int8_size = (A.numel() + B.numel()) * 1 + C_fp32.numel() * 4 + 8  # scales
    print(f"\nMemory usage:")
    print(f"  FP32: {fp32_size / 1024:.2f} KB")
    print(f"  INT8: {int8_size / 1024:.2f} KB")
    print(f"  Savings: {(1 - int8_size/fp32_size) * 100:.1f}%")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Foundation Models: Quantization")
    print("=" * 60)

    demo_symmetric_quantization()
    demo_asymmetric_quantization()
    demo_per_channel_quantization()
    demo_bit_depth_comparison()
    demo_calibration()
    demo_quantized_matmul()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. Quantization maps FP32 → INT8/INT4 for compression")
    print("2. Symmetric: best for weights (centered at 0)")
    print("3. Asymmetric: best for activations (arbitrary range)")
    print("4. Per-channel: better accuracy for heterogeneous data")
    print("5. Calibration: clip outliers to reduce quantization error")
    print("6. INT8 provides ~4x compression with minimal accuracy loss")
    print("=" * 60)
