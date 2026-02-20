"""
NumPy로 구현한 Convolution 연산

이 파일에서는 Convolution의 forward/backward를 순수 NumPy로 구현합니다.
"""

import numpy as np
from typing import Tuple, Optional


def conv2d_naive(
    input: np.ndarray,
    kernel: np.ndarray,
    bias: Optional[np.ndarray] = None,
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:
    """
    2D Convolution (naive implementation with loops)

    Args:
        input: (N, C_in, H, W) - 배치 입력
        kernel: (C_out, C_in, K_h, K_w) - 필터
        bias: (C_out,) - 편향
        stride: 스트라이드
        padding: 패딩

    Returns:
        output: (N, C_out, H_out, W_out)
    """
    N, C_in, H, W = input.shape
    C_out, _, K_h, K_w = kernel.shape

    # 패딩 적용
    if padding > 0:
        input_padded = np.pad(
            input,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )
    else:
        input_padded = input

    # 출력 크기 계산
    H_out = (H + 2 * padding - K_h) // stride + 1
    W_out = (W + 2 * padding - K_w) // stride + 1

    output = np.zeros((N, C_out, H_out, W_out))

    # Convolution 연산 (6중 루프 - 매우 느림)
    for n in range(N):                          # 배치
        for c_out in range(C_out):              # 출력 채널
            for h in range(H_out):              # 출력 높이
                for w in range(W_out):          # 출력 너비
                    # 수용 영역
                    h_start = h * stride
                    h_end = h_start + K_h
                    w_start = w * stride
                    w_end = w_start + K_w

                    # 수용 영역과 커널의 element-wise 곱의 합
                    receptive_field = input_padded[n, :, h_start:h_end, w_start:w_end]
                    output[n, c_out, h, w] = np.sum(receptive_field * kernel[c_out])

    # 편향 추가
    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)

    return output


def im2col(
    input: np.ndarray,
    kernel_size: Tuple[int, int],
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:
    """
    im2col: 이미지를 행렬로 변환 (효율적인 convolution을 위해)

    Convolution을 행렬 곱셈으로 변환:
    - 각 수용 영역을 열 벡터로 변환
    - 커널을 행 벡터로 변환
    - 행렬 곱셈으로 convolution 수행

    Args:
        input: (N, C, H, W)
        kernel_size: (K_h, K_w)
        stride: 스트라이드
        padding: 패딩

    Returns:
        col: (N, C * K_h * K_w, H_out * W_out)
    """
    N, C, H, W = input.shape
    K_h, K_w = kernel_size

    # 패딩
    if padding > 0:
        input_padded = np.pad(
            input,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )
    else:
        input_padded = input

    H_padded, W_padded = input_padded.shape[2], input_padded.shape[3]

    # 출력 크기
    H_out = (H_padded - K_h) // stride + 1
    W_out = (W_padded - K_w) // stride + 1

    # im2col 행렬
    col = np.zeros((N, C, K_h, K_w, H_out, W_out))

    for h in range(K_h):
        h_max = h + stride * H_out
        for w in range(K_w):
            w_max = w + stride * W_out
            col[:, :, h, w, :, :] = input_padded[:, :, h:h_max:stride, w:w_max:stride]

    # (N, C, K_h, K_w, H_out, W_out) → (N, C*K_h*K_w, H_out*W_out)
    col = col.transpose(0, 1, 2, 3, 4, 5).reshape(N, C * K_h * K_w, H_out * W_out)

    return col


def col2im(
    col: np.ndarray,
    input_shape: Tuple[int, int, int, int],
    kernel_size: Tuple[int, int],
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:
    """
    col2im: im2col의 역연산

    Backward pass에서 gradient를 원래 이미지 형태로 복원

    Args:
        col: (N, C * K_h * K_w, H_out * W_out)
        input_shape: (N, C, H, W) 원본 입력 shape
        kernel_size: (K_h, K_w)
        stride: 스트라이드
        padding: 패딩

    Returns:
        input_grad: (N, C, H, W)
    """
    N, C, H, W = input_shape
    K_h, K_w = kernel_size

    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    H_out = (H_padded - K_h) // stride + 1
    W_out = (W_padded - K_w) // stride + 1

    # col reshape: (N, C*K_h*K_w, H_out*W_out) → (N, C, K_h, K_w, H_out, W_out)
    col = col.reshape(N, C, K_h, K_w, H_out, W_out)

    # 출력 배열 (패딩 포함)
    input_padded = np.zeros((N, C, H_padded, W_padded))

    # 누적 (stride 위치에 값 더하기)
    for h in range(K_h):
        h_max = h + stride * H_out
        for w in range(K_w):
            w_max = w + stride * W_out
            input_padded[:, :, h:h_max:stride, w:w_max:stride] += col[:, :, h, w, :, :]

    # 패딩 제거
    if padding > 0:
        return input_padded[:, :, padding:-padding, padding:-padding]
    return input_padded


def conv2d_im2col(
    input: np.ndarray,
    kernel: np.ndarray,
    bias: Optional[np.ndarray] = None,
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:
    """
    im2col을 사용한 효율적인 Convolution

    연산: Y = W · col(X) + b
    """
    N, C_in, H, W = input.shape
    C_out, _, K_h, K_w = kernel.shape

    # im2col 변환
    col = im2col(input, (K_h, K_w), stride, padding)  # (N, C_in*K_h*K_w, H_out*W_out)

    # 커널을 행렬로 변환
    kernel_mat = kernel.reshape(C_out, -1)  # (C_out, C_in*K_h*K_w)

    # 행렬 곱셈
    H_out = (H + 2 * padding - K_h) // stride + 1
    W_out = (W + 2 * padding - K_w) // stride + 1

    # (C_out, C_in*K_h*K_w) @ (N, C_in*K_h*K_w, H_out*W_out)
    # → (N, C_out, H_out*W_out)
    output = np.zeros((N, C_out, H_out * W_out))
    for n in range(N):
        output[n] = kernel_mat @ col[n]

    # Reshape
    output = output.reshape(N, C_out, H_out, W_out)

    # 편향
    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)

    return output


class Conv2dNumpy:
    """
    NumPy Convolution 레이어 (학습 가능)

    forward/backward 모두 구현
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Kaiming (He) 초기화
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * scale
        self.bias = np.zeros(out_channels)

        # Gradient 저장
        self.weight_grad = None
        self.bias_grad = None

        # Backward를 위한 캐시
        self.cache = {}

    def forward(self, input: np.ndarray) -> np.ndarray:
        """Forward pass"""
        N, C, H, W = input.shape

        # im2col
        col = im2col(input, (self.kernel_size, self.kernel_size),
                     self.stride, self.padding)

        # 캐시 저장
        self.cache['input_shape'] = input.shape
        self.cache['col'] = col

        # 행렬 곱셈
        kernel_mat = self.weight.reshape(self.out_channels, -1)

        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        output = np.zeros((N, self.out_channels, H_out * W_out))
        for n in range(N):
            output[n] = kernel_mat @ col[n]

        output = output.reshape(N, self.out_channels, H_out, W_out)
        output += self.bias.reshape(1, -1, 1, 1)

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass

        Args:
            grad_output: ∂L/∂Y (N, C_out, H_out, W_out)

        Returns:
            grad_input: ∂L/∂X (N, C_in, H, W)
        """
        N, C_out, H_out, W_out = grad_output.shape
        input_shape = self.cache['input_shape']
        col = self.cache['col']

        # Bias gradient: ∂L/∂b = Σ ∂L/∂Y
        self.bias_grad = np.sum(grad_output, axis=(0, 2, 3))

        # grad_output을 행렬로 변환
        grad_output_mat = grad_output.reshape(N, C_out, -1)  # (N, C_out, H_out*W_out)

        # Weight gradient: ∂L/∂W = ∂L/∂Y · col(X)^T
        kernel_mat = self.weight.reshape(self.out_channels, -1)
        self.weight_grad = np.zeros_like(kernel_mat)

        for n in range(N):
            self.weight_grad += grad_output_mat[n] @ col[n].T

        self.weight_grad = self.weight_grad.reshape(self.weight.shape)

        # Input gradient: ∂L/∂X = col2im(W^T · ∂L/∂Y)
        grad_col = np.zeros_like(col)
        for n in range(N):
            grad_col[n] = kernel_mat.T @ grad_output_mat[n]

        grad_input = col2im(
            grad_col, input_shape,
            (self.kernel_size, self.kernel_size),
            self.stride, self.padding
        )

        return grad_input

    def update(self, lr: float):
        """가중치 업데이트"""
        self.weight -= lr * self.weight_grad
        self.bias -= lr * self.bias_grad


# 테스트
if __name__ == "__main__":
    np.random.seed(42)

    # 테스트 입력
    N, C_in, H, W = 2, 3, 8, 8
    C_out, K = 4, 3

    input = np.random.randn(N, C_in, H, W)
    kernel = np.random.randn(C_out, C_in, K, K)
    bias = np.random.randn(C_out)

    # Naive vs im2col 비교
    output_naive = conv2d_naive(input, kernel, bias, stride=1, padding=1)
    output_im2col = conv2d_im2col(input, kernel, bias, stride=1, padding=1)

    print("Output shape:", output_naive.shape)
    print("Naive vs im2col 차이:", np.max(np.abs(output_naive - output_im2col)))

    # Conv2dNumpy 테스트
    conv = Conv2dNumpy(C_in, C_out, K, stride=1, padding=1)
    output = conv.forward(input)
    print("\nConv2dNumpy output shape:", output.shape)

    # Backward 테스트
    grad_output = np.random.randn(*output.shape)
    grad_input = conv.backward(grad_output)
    print("Grad input shape:", grad_input.shape)
    print("Weight grad shape:", conv.weight_grad.shape)

    # Gradient check
    def numerical_gradient(f, x, h=1e-5):
        """수치 미분으로 gradient 검증"""
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

        while not it.finished:
            idx = it.multi_index
            old_val = x[idx]

            x[idx] = old_val + h
            fxh1 = f()

            x[idx] = old_val - h
            fxh2 = f()

            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = old_val

            it.iternext()

        return grad

    print("\n=== Gradient Check ===")

    # 작은 입력으로 gradient check
    small_input = np.random.randn(1, 2, 4, 4)
    small_conv = Conv2dNumpy(2, 2, 3, stride=1, padding=1)

    def loss_fn():
        out = small_conv.forward(small_input)
        return np.sum(out ** 2)

    # Analytical gradient
    output = small_conv.forward(small_input)
    grad_output = 2 * output  # d(sum(x^2))/dx = 2x
    grad_input = small_conv.backward(grad_output)

    # Numerical gradient (입력에 대해)
    num_grad = numerical_gradient(loss_fn, small_input)

    print("Input gradient check:")
    print(f"  Max diff: {np.max(np.abs(grad_input - num_grad)):.2e}")

    # Weight gradient check
    def loss_fn_weight():
        out = small_conv.forward(small_input)
        return np.sum(out ** 2)

    num_grad_weight = numerical_gradient(loss_fn_weight, small_conv.weight)

    # Backward로 계산
    output = small_conv.forward(small_input)
    small_conv.backward(2 * output)

    print("Weight gradient check:")
    print(f"  Max diff: {np.max(np.abs(small_conv.weight_grad - num_grad_weight)):.2e}")
