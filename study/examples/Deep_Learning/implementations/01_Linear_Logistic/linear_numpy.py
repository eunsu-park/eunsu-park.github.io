"""
Linear Regression - NumPy From-Scratch 구현

이 파일은 선형 회귀를 순수 NumPy로 구현합니다.
프레임워크 없이 gradient descent를 직접 구현하여
딥러닝의 기본 원리를 이해합니다.

학습 목표:
1. Forward pass: y_hat = Xw + b
2. Loss 계산: MSE = (1/2n) * ||y - y_hat||^2
3. Backward pass: gradient 계산
4. Weight update: w = w - lr * dw
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionNumpy:
    """
    NumPy로 구현한 Linear Regression

    수학적 배경:
    - 모델: ŷ = Xw + b
    - 손실: L = (1/2n) Σ(y - ŷ)²
    - 그래디언트:
        ∂L/∂w = (1/n) X^T (ŷ - y)
        ∂L/∂b = (1/n) Σ(ŷ - y)
    """

    def __init__(self, input_dim: int, output_dim: int = 1):
        """
        Args:
            input_dim: 입력 특성 수
            output_dim: 출력 차원 (기본 1)
        """
        # Xavier/He 초기화: 분산을 2/n으로 유지
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))

        # 그래디언트 저장용
        self.dW = None
        self.db = None

        # Forward에서 캐시 (backward에서 사용)
        self._cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: ŷ = Xw + b

        Args:
            X: 입력 데이터 (batch_size, input_dim)

        Returns:
            y_hat: 예측값 (batch_size, output_dim)
        """
        # 입력 캐시 (backward에서 필요)
        self._cache['X'] = X

        # 선형 변환: y = Xw + b
        y_hat = np.dot(X, self.W) + self.b

        return y_hat

    def compute_loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Mean Squared Error 손실 계산

        L = (1/2n) Σ(y - ŷ)²

        Args:
            y: 실제값 (batch_size, output_dim)
            y_hat: 예측값 (batch_size, output_dim)

        Returns:
            loss: 스칼라 손실값
        """
        n = y.shape[0]
        loss = (1 / (2 * n)) * np.sum((y - y_hat) ** 2)
        return loss

    def backward(self, y: np.ndarray, y_hat: np.ndarray) -> None:
        """
        Backward pass: 그래디언트 계산

        Chain Rule 적용:
        ∂L/∂w = ∂L/∂ŷ × ∂ŷ/∂w
               = (1/n)(ŷ - y) × X^T
               = (1/n) X^T (ŷ - y)

        ∂L/∂b = ∂L/∂ŷ × ∂ŷ/∂b
               = (1/n) Σ(ŷ - y)

        Args:
            y: 실제값
            y_hat: 예측값
        """
        X = self._cache['X']
        n = y.shape[0]

        # 오차
        error = y_hat - y  # (batch_size, output_dim)

        # 그래디언트 계산
        # ∂L/∂W = (1/n) X^T @ error
        self.dW = (1 / n) * np.dot(X.T, error)

        # ∂L/∂b = (1/n) Σerror (각 출력 차원별)
        self.db = (1 / n) * np.sum(error, axis=0, keepdims=True)

    def update(self, lr: float) -> None:
        """
        가중치 업데이트 (Gradient Descent)

        w = w - η × ∂L/∂w
        b = b - η × ∂L/∂b

        Args:
            lr: learning rate
        """
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 1000,
        verbose: bool = True
    ) -> list:
        """
        모델 학습

        Args:
            X: 학습 데이터 (n_samples, n_features)
            y: 타겟값 (n_samples, 1) 또는 (n_samples,)
            lr: learning rate
            epochs: 학습 반복 횟수
            verbose: 진행 상황 출력 여부

        Returns:
            losses: 에폭별 손실 리스트
        """
        # y shape 보정
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        losses = []

        for epoch in range(epochs):
            # 1. Forward pass
            y_hat = self.forward(X)

            # 2. Loss 계산
            loss = self.compute_loss(y, y_hat)
            losses.append(loss)

            # 3. Backward pass (gradient 계산)
            self.backward(y, y_hat)

            # 4. Weight update
            self.update(lr)

            # 진행 상황 출력
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측"""
        return self.forward(X)


def generate_sample_data(n_samples: int = 100, n_features: int = 1, noise: float = 0.1):
    """
    테스트용 샘플 데이터 생성

    y = 2x + 3 + noise
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    # 실제 가중치 (학습으로 찾아야 할 값)
    true_w = np.array([[2.0]])
    true_b = 3.0

    y = np.dot(X, true_w) + true_b + noise * np.random.randn(n_samples, 1)

    return X, y, true_w, true_b


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Linear Regression - NumPy From-Scratch 구현")
    print("=" * 60)

    # 1. 데이터 생성
    print("\n1. 샘플 데이터 생성")
    X, y, true_w, true_b = generate_sample_data(n_samples=100, noise=0.1)
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   True w: {true_w.flatten()}, True b: {true_b}")

    # 2. 모델 생성
    print("\n2. 모델 초기화")
    model = LinearRegressionNumpy(input_dim=1, output_dim=1)
    print(f"   Initial W: {model.W.flatten()}")
    print(f"   Initial b: {model.b.flatten()}")

    # 3. 학습
    print("\n3. 학습 시작")
    losses = model.fit(X, y, lr=0.1, epochs=100, verbose=True)

    # 4. 결과 확인
    print("\n4. 학습 결과")
    print(f"   Learned W: {model.W.flatten()}")
    print(f"   Learned b: {model.b.flatten()}")
    print(f"   True W: {true_w.flatten()}")
    print(f"   True b: {true_b}")
    print(f"   Final Loss: {losses[-1]:.6f}")

    # 5. 시각화
    print("\n5. 시각화")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss 곡선
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)

    # 데이터와 예측 직선
    y_pred = model.predict(X)
    sorted_idx = np.argsort(X.flatten())
    axes[1].scatter(X, y, alpha=0.5, label='Data')
    axes[1].plot(X[sorted_idx], y_pred[sorted_idx], 'r-', linewidth=2, label='Prediction')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('y')
    axes[1].set_title('Linear Regression Fit')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('linear_regression_result.png', dpi=150)
    plt.show()
    print("   결과 이미지 저장: linear_regression_result.png")


if __name__ == "__main__":
    main()
