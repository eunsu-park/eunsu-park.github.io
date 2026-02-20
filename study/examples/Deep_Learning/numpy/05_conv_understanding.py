"""
05. 합성곱 이해 - NumPy 버전 (교육용)

합성곱 연산의 원리를 NumPy로 이해합니다.
실제 CNN 학습에는 PyTorch를 사용하세요!

이 파일은 합성곱이 어떻게 동작하는지 이해하기 위한 것입니다.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("NumPy 합성곱 이해 (교육용)")
print("=" * 60)


# ============================================
# 1. 기본 2D 합성곱
# ============================================
print("\n[1] 기본 2D 합성곱")
print("-" * 40)

def conv2d_basic(image, kernel):
    """
    가장 기본적인 2D 합성곱 구현

    Args:
        image: 2D 배열 (H, W)
        kernel: 2D 배열 (kH, kW)

    Returns:
        출력 (H-kH+1, W-kW+1)
    """
    h, w = image.shape
    kh, kw = kernel.shape
    oh, ow = h - kh + 1, w - kw + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            # 영역 추출
            region = image[i:i+kh, j:j+kw]
            # 요소별 곱셈 후 합산
            output[i, j] = np.sum(region * kernel)

    return output

# 테스트
image = np.array([
    [1, 2, 3, 0],
    [0, 1, 2, 3],
    [3, 0, 1, 2],
    [2, 3, 0, 1]
], dtype=float)

kernel = np.array([
    [1, 0],
    [0, -1]
], dtype=float)

output = conv2d_basic(image, kernel)
print(f"입력 이미지 (4×4):\n{image}")
print(f"\n커널 (2×2):\n{kernel}")
print(f"\n출력 (3×3):\n{output}")
print(f"\n예시 계산 (좌상단):")
print(f"  {image[0,0]}×{kernel[0,0]} + {image[0,1]}×{kernel[0,1]} + {image[1,0]}×{kernel[1,0]} + {image[1,1]}×{kernel[1,1]}")
print(f"  = 1×1 + 2×0 + 0×0 + 1×(-1) = 0")


# ============================================
# 2. 패딩과 스트라이드
# ============================================
print("\n[2] 패딩과 스트라이드")
print("-" * 40)

def conv2d_with_padding(image, kernel, padding=0, stride=1):
    """패딩과 스트라이드를 지원하는 합성곱"""
    # 패딩 적용
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)

    h, w = image.shape
    kh, kw = kernel.shape
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            si, sj = i * stride, j * stride
            region = image[si:si+kh, sj:sj+kw]
            output[i, j] = np.sum(region * kernel)

    return output

# 테스트
image = np.ones((4, 4))
kernel = np.ones((3, 3))

print("입력: 4×4, 커널: 3×3")
for p in [0, 1]:
    for s in [1, 2]:
        out = conv2d_with_padding(image, kernel, padding=p, stride=s)
        print(f"  padding={p}, stride={s} → 출력: {out.shape}")


# ============================================
# 3. 에지 검출 필터
# ============================================
print("\n[3] 에지 검출 필터")
print("-" * 40)

# 샘플 이미지 생성
def create_sample_image():
    """간단한 패턴 이미지 생성"""
    img = np.zeros((8, 8))
    img[2:6, 2:6] = 1  # 중앙 사각형
    return img

image = create_sample_image()

# 에지 검출 필터들
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

laplacian = np.array([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]])

# 필터 적용
edge_x = conv2d_with_padding(image, sobel_x, padding=1)
edge_y = conv2d_with_padding(image, sobel_y, padding=1)
edge_laplace = conv2d_with_padding(image, laplacian, padding=1)

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 1].imshow(sobel_x, cmap='RdBu')
axes[0, 1].set_title('Sobel X Filter')
axes[0, 2].imshow(sobel_y, cmap='RdBu')
axes[0, 2].set_title('Sobel Y Filter')
axes[1, 0].imshow(edge_x, cmap='gray')
axes[1, 0].set_title('Sobel X Edge')
axes[1, 1].imshow(edge_y, cmap='gray')
axes[1, 1].set_title('Sobel Y Edge')
axes[1, 2].imshow(edge_laplace, cmap='gray')
axes[1, 2].set_title('Laplacian Edge')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('numpy_edge_detection.png', dpi=100)
plt.close()
print("에지 검출 저장: numpy_edge_detection.png")


# ============================================
# 4. 풀링 연산
# ============================================
print("\n[4] 풀링 연산")
print("-" * 40)

def max_pool2d(image, pool_size=2, stride=2):
    """Max Pooling 구현"""
    h, w = image.shape
    oh = (h - pool_size) // stride + 1
    ow = (w - pool_size) // stride + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            si, sj = i * stride, j * stride
            region = image[si:si+pool_size, sj:sj+pool_size]
            output[i, j] = np.max(region)

    return output

def avg_pool2d(image, pool_size=2, stride=2):
    """Average Pooling 구현"""
    h, w = image.shape
    oh = (h - pool_size) // stride + 1
    ow = (w - pool_size) // stride + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            si, sj = i * stride, j * stride
            region = image[si:si+pool_size, sj:sj+pool_size]
            output[i, j] = np.mean(region)

    return output

# 테스트
image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=float)

print(f"입력:\n{image}")
print(f"\nMax Pooling (2×2):\n{max_pool2d(image)}")
print(f"\nAvg Pooling (2×2):\n{avg_pool2d(image)}")


# ============================================
# 5. 다채널 합성곱
# ============================================
print("\n[5] 다채널 합성곱")
print("-" * 40)

def conv2d_multichannel(image, kernels, bias=0):
    """
    다채널 합성곱 (RGB 이미지 등)

    Args:
        image: (C, H, W) - C개 채널
        kernels: (C, kH, kW) - 각 채널용 커널
        bias: 편향

    Returns:
        출력: (H-kH+1, W-kW+1)
    """
    c, h, w = image.shape
    _, kh, kw = kernels.shape
    oh, ow = h - kh + 1, w - kw + 1

    output = np.zeros((oh, ow))

    # 각 채널에 대해 합성곱 후 합산
    for ch in range(c):
        output += conv2d_basic(image[ch], kernels[ch])

    return output + bias

# RGB 이미지 예시
rgb_image = np.random.rand(3, 8, 8)  # (C, H, W)
kernels = np.random.rand(3, 3, 3)    # (C, kH, kW)

output = conv2d_multichannel(rgb_image, kernels)
print(f"입력: {rgb_image.shape} (3채널)")
print(f"커널: {kernels.shape} (채널별 3×3)")
print(f"출력: {output.shape}")


# ============================================
# 6. 여러 필터 적용
# ============================================
print("\n[6] 여러 필터 적용")
print("-" * 40)

def conv2d_layer(image, filters, biases):
    """
    Conv 층 시뮬레이션

    Args:
        image: (C_in, H, W)
        filters: (C_out, C_in, kH, kW)
        biases: (C_out,)

    Returns:
        출력: (C_out, oH, oW)
    """
    c_out, c_in, kh, kw = filters.shape
    _, h, w = image.shape
    oh, ow = h - kh + 1, w - kw + 1

    output = np.zeros((c_out, oh, ow))

    for f in range(c_out):
        output[f] = conv2d_multichannel(image, filters[f], biases[f])

    return output

# 예시: 3채널 입력 → 8채널 출력
image = np.random.rand(3, 16, 16)
filters = np.random.rand(8, 3, 3, 3)  # 8개 필터
biases = np.zeros(8)

output = conv2d_layer(image, filters, biases)
print(f"입력: {image.shape}")
print(f"필터: {filters.shape}")
print(f"출력: {output.shape}")


# ============================================
# 7. CNN 순전파 시뮬레이션
# ============================================
print("\n[7] CNN 순전파 시뮬레이션")
print("-" * 40)

def relu(x):
    return np.maximum(0, x)

def simple_cnn_forward(image):
    """
    간단한 CNN 순전파

    입력 (1, 8, 8) → Conv (2, 6, 6) → Pool (2, 3, 3) → FC → 출력
    """
    # Conv1: 1→2 채널, 3×3 커널
    filters1 = np.random.randn(2, 1, 3, 3) * 0.5
    biases1 = np.zeros(2)

    conv1_out = conv2d_layer(image, filters1, biases1)
    relu1_out = relu(conv1_out)
    print(f"  Conv1 후: {relu1_out.shape}")

    # MaxPool: 2×2
    pool_out = np.zeros((2, 3, 3))
    for c in range(2):
        pool_out[c] = max_pool2d(relu1_out[c], 2, 2)
    print(f"  Pool 후: {pool_out.shape}")

    # Flatten
    flat = pool_out.flatten()
    print(f"  Flatten: {flat.shape}")

    # FC
    fc_weights = np.random.randn(10, 18) * 0.5
    fc_bias = np.zeros(10)
    output = fc_weights @ flat + fc_bias
    print(f"  FC 출력: {output.shape}")

    return output

# 테스트
image = np.random.rand(1, 8, 8)
print(f"입력: {image.shape}")
output = simple_cnn_forward(image)


# ============================================
# 왜 PyTorch를 사용해야 하는가?
# ============================================
print("\n" + "=" * 60)
print("NumPy CNN의 한계")
print("=" * 60)

limitations = """
NumPy 구현의 문제점:

1. 속도
   - 순수 Python 루프는 매우 느림
   - 28×28 MNIST도 수천 배 느림
   - GPU 가속 불가능

2. 역전파
   - 합성곱 역전파 구현이 복잡
   - im2col 등 최적화 필요
   - 실수하기 쉬움

3. 메모리
   - 비효율적인 메모리 사용
   - 배치 처리 어려움

4. 기능
   - BatchNorm, Dropout 구현 복잡
   - 다양한 층/연산 부족

PyTorch 사용 이유:
   ✓ cuDNN으로 최적화된 합성곱
   ✓ 자동 미분 (역전파 자동)
   ✓ GPU 지원
   ✓ 풍부한 레이어/함수 제공
"""
print(limitations)


# ============================================
# 정리
# ============================================
print("=" * 60)
print("합성곱 핵심 정리")
print("=" * 60)

summary = """
합성곱 연산:
    output[i,j] = Σ input[i+m, j+n] × kernel[m, n]

출력 크기:
    output_size = (input - kernel + 2×padding) / stride + 1

풀링:
    - MaxPool: 영역 내 최대값 선택
    - AvgPool: 영역 내 평균

다채널:
    - 각 채널에 별도 커널 적용 후 합산
    - 여러 필터 = 여러 출력 채널

학습:
    - 커널의 가중치가 학습됨
    - 역전파로 최적화

NumPy로 배운 것:
    1. 합성곱의 수학적 정의
    2. 패딩과 스트라이드의 효과
    3. 풀링의 동작 원리
    4. 다채널 처리 방식

실전에서는 PyTorch!
"""
print(summary)
print("=" * 60)
