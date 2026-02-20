"""
Multi-Layer Perceptron - NumPy From-Scratch 구현

이 파일은 MLP를 순수 NumPy로 구현합니다.
Backpropagation 알고리즘을 직접 구현하여
딥러닝의 핵심 원리를 이해합니다.

학습 목표:
1. Forward pass: 다층 신경망의 순전파
2. Backward pass: Chain rule을 이용한 역전파
3. Activation functions: ReLU, Sigmoid, Tanh
4. Weight initialization: Xavier, He 초기화
"""

import numpy as np


class ActivationFunctions:
    """활성화 함수와 그 미분"""

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z):
        # 수치 안정성을 위해 클리핑
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        s = ActivationFunctions.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def softmax(z):
        # 수치 안정성: 최대값을 빼줌
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class Layer:
    """
    단일 Fully Connected Layer

    z = Wx + b (선형 변환)
    a = σ(z)   (활성화)
    """

    def __init__(self, input_dim: int, output_dim: int, activation: str = 'relu'):
        """
        Args:
            input_dim: 입력 차원
            output_dim: 출력 차원
            activation: 'relu', 'sigmoid', 'tanh', 'none'
        """
        # He 초기화 (ReLU용)
        if activation == 'relu':
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        else:
            # Xavier 초기화
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1.0 / input_dim)

        self.b = np.zeros((1, output_dim))

        self.activation = activation
        self._get_activation_fn()

        # Gradients
        self.dW = None
        self.db = None

        # Cache (for backward)
        self.cache = {}

    def _get_activation_fn(self):
        """활성화 함수 설정"""
        activations = {
            'relu': (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
            'sigmoid': (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
            'tanh': (ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
            'none': (lambda x: x, lambda x: np.ones_like(x)),
        }
        self.act_fn, self.act_derivative = activations[self.activation]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Args:
            x: 입력 (batch_size, input_dim)

        Returns:
            a: 활성화 출력 (batch_size, output_dim)
        """
        # 캐시 저장 (backward에서 사용)
        self.cache['x'] = x

        # 선형 변환: z = Wx + b
        z = np.dot(x, self.W) + self.b
        self.cache['z'] = z

        # 활성화: a = σ(z)
        a = self.act_fn(z)
        self.cache['a'] = a

        return a

    def backward(self, da: np.ndarray) -> np.ndarray:
        """
        Backward pass

        Args:
            da: 출력의 gradient (batch_size, output_dim)

        Returns:
            dx: 입력의 gradient (batch_size, input_dim)
        """
        x = self.cache['x']
        z = self.cache['z']
        batch_size = x.shape[0]

        # ∂L/∂z = ∂L/∂a × ∂a/∂z = da × σ'(z)
        dz = da * self.act_derivative(z)

        # ∂L/∂W = x^T × ∂L/∂z
        self.dW = np.dot(x.T, dz) / batch_size

        # ∂L/∂b = sum(∂L/∂z)
        self.db = np.sum(dz, axis=0, keepdims=True) / batch_size

        # ∂L/∂x = ∂L/∂z × W^T (다음 레이어로 전파)
        dx = np.dot(dz, self.W.T)

        return dx


class MLPNumpy:
    """
    Multi-Layer Perceptron (NumPy 구현)

    사용 예:
        model = MLPNumpy([784, 256, 128, 10], activations=['relu', 'relu', 'none'])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    """

    def __init__(self, layer_dims: list, activations: list = None):
        """
        Args:
            layer_dims: 각 레이어의 차원 [input, hidden1, hidden2, ..., output]
            activations: 각 레이어의 활성화 함수 (마지막 레이어 제외)
        """
        self.layers = []
        n_layers = len(layer_dims) - 1

        if activations is None:
            activations = ['relu'] * (n_layers - 1) + ['none']

        for i in range(n_layers):
            layer = Layer(layer_dims[i], layer_dims[i + 1], activations[i])
            self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """전체 네트워크 forward pass"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad: np.ndarray) -> None:
        """전체 네트워크 backward pass"""
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Cross-entropy loss (분류용)

        L = -1/n × Σ y_true × log(y_pred)
        """
        eps = 1e-15  # 수치 안정성
        y_pred = np.clip(y_pred, eps, 1 - eps)

        if y_true.ndim == 1:
            # Sparse labels → one-hot
            n_classes = y_pred.shape[1]
            y_true_onehot = np.zeros((len(y_true), n_classes))
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            y_true = y_true_onehot

        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss

    def compute_loss_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Cross-entropy gradient (softmax 출력 가정)

        ∂L/∂z = y_pred - y_true (softmax + CE의 경우 간단해짐)
        """
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_true_onehot = np.zeros((len(y_true), n_classes))
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            y_true = y_true_onehot

        return y_pred - y_true

    def update_weights(self, lr: float) -> None:
        """SGD 가중치 업데이트"""
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
        verbose: bool = True
    ) -> list:
        """
        모델 학습

        Args:
            X: 학습 데이터 (n_samples, n_features)
            y: 레이블 (n_samples,) 또는 (n_samples, n_classes)
            epochs: 에폭 수
            lr: learning rate
            batch_size: 배치 크기
            verbose: 진행 상황 출력

        Returns:
            losses: 에폭별 손실 리스트
        """
        n_samples = X.shape[0]
        losses = []

        for epoch in range(epochs):
            # 셔플
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices] if y.ndim == 1 else y[indices]

            epoch_loss = 0

            # 미니배치 학습
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward
                y_pred = self.forward(X_batch)

                # Softmax (마지막 레이어가 none일 경우)
                y_pred = ActivationFunctions.softmax(y_pred)

                # Loss
                loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += loss * len(X_batch)

                # Backward
                loss_grad = self.compute_loss_gradient(y_batch, y_pred)
                self.backward(loss_grad)

                # Update
                self.update_weights(lr)

            epoch_loss /= n_samples
            losses.append(epoch_loss)

            if verbose and (epoch + 1) % (epochs // 10) == 0:
                accuracy = self.evaluate(X, y)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측"""
        logits = self.forward(X)
        probs = ActivationFunctions.softmax(logits)
        return np.argmax(probs, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """정확도 평가"""
        predictions = self.predict(X)
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        return np.mean(predictions == y)


def load_mnist_sample(n_samples=1000):
    """MNIST 샘플 데이터 생성 (테스트용)"""
    np.random.seed(42)

    # 간단한 가상 데이터 (실제로는 MNIST 로드)
    n_classes = 10
    n_features = 784  # 28x28

    X = np.random.randn(n_samples, n_features) * 0.5
    y = np.random.randint(0, n_classes, n_samples)

    # 클래스별로 약간의 패턴 추가
    for i in range(n_classes):
        mask = y == i
        X[mask, i * 78:(i + 1) * 78] += 1.0

    return X, y


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Multi-Layer Perceptron - NumPy From-Scratch 구현")
    print("=" * 60)

    # 1. 데이터 생성
    print("\n1. 샘플 데이터 생성")
    X_train, y_train = load_mnist_sample(n_samples=1000)
    X_test, y_test = load_mnist_sample(n_samples=200)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # 2. 모델 생성
    print("\n2. MLP 모델 초기화")
    model = MLPNumpy(
        layer_dims=[784, 128, 64, 10],
        activations=['relu', 'relu', 'none']
    )
    print(f"   Layers: {[l.W.shape for l in model.layers]}")

    # 3. 학습
    print("\n3. 학습 시작")
    losses = model.fit(
        X_train, y_train,
        epochs=50,
        lr=0.1,
        batch_size=32,
        verbose=True
    )

    # 4. 평가
    print("\n4. 평가 결과")
    train_acc = model.evaluate(X_train, y_train)
    test_acc = model.evaluate(X_test, y_test)
    print(f"   Train Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")

    # 5. 시각화
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss 곡선
        axes[0].plot(losses)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True)

        # 가중치 분포 (첫 번째 레이어)
        axes[1].hist(model.layers[0].W.flatten(), bins=50, alpha=0.7)
        axes[1].set_xlabel('Weight Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('First Layer Weight Distribution')
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('mlp_result.png', dpi=150)
        plt.show()
        print("\n결과 이미지 저장: mlp_result.png")

    except ImportError:
        print("\n(matplotlib 없음, 시각화 생략)")


if __name__ == "__main__":
    main()
