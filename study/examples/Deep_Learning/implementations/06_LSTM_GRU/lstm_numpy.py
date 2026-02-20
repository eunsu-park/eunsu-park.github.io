"""
NumPy LSTM From-Scratch 구현

모든 게이트 연산과 BPTT를 직접 구현
"""

import numpy as np
from typing import Tuple, Dict, List


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 활성화 (수치 안정성 고려)"""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def sigmoid_derivative(s: np.ndarray) -> np.ndarray:
    """Sigmoid의 derivative: s * (1 - s)"""
    return s * (1 - s)


def tanh_derivative(t: np.ndarray) -> np.ndarray:
    """Tanh의 derivative: 1 - t^2"""
    return 1 - t ** 2


class LSTMCellNumPy:
    """
    단일 LSTM Cell (NumPy 구현)

    수식:
        f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
        i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
        c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
        h_t = o_t ⊙ tanh(c_t)
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Xavier 초기화
        concat_size = input_size + hidden_size
        scale = np.sqrt(2.0 / (concat_size + hidden_size))

        # 4개 게이트를 하나의 가중치로 관리 (효율성)
        # 순서: forget, input, candidate, output
        self.W = np.random.randn(4 * hidden_size, concat_size) * scale
        self.b = np.zeros(4 * hidden_size)

        # Gradient 저장
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Forward pass 캐시
        self.cache = {}

    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass

        Args:
            x: (batch_size, input_size) 현재 입력
            h_prev: (batch_size, hidden_size) 이전 hidden
            c_prev: (batch_size, hidden_size) 이전 cell

        Returns:
            h_t: (batch_size, hidden_size) 현재 hidden
            c_t: (batch_size, hidden_size) 현재 cell
        """
        batch_size = x.shape[0]
        H = self.hidden_size

        # Concatenate [h_prev, x]
        concat = np.concatenate([h_prev, x], axis=1)  # (batch, hidden+input)

        # 모든 게이트 한번에 계산
        gates = concat @ self.W.T + self.b  # (batch, 4*hidden)

        # 분리
        f_gate = sigmoid(gates[:, 0:H])           # Forget gate
        i_gate = sigmoid(gates[:, H:2*H])         # Input gate
        c_tilde = np.tanh(gates[:, 2*H:3*H])      # Candidate
        o_gate = sigmoid(gates[:, 3*H:4*H])       # Output gate

        # Cell state 업데이트
        c_t = f_gate * c_prev + i_gate * c_tilde

        # Hidden state
        h_t = o_gate * np.tanh(c_t)

        # Backward를 위한 캐시
        self.cache = {
            'x': x,
            'h_prev': h_prev,
            'c_prev': c_prev,
            'concat': concat,
            'f_gate': f_gate,
            'i_gate': i_gate,
            'c_tilde': c_tilde,
            'o_gate': o_gate,
            'c_t': c_t,
            'h_t': h_t,
            'tanh_c_t': np.tanh(c_t),
        }

        return h_t, c_t

    def backward(
        self,
        dh_next: np.ndarray,
        dc_next: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass (BPTT 한 스텝)

        Args:
            dh_next: (batch_size, hidden_size) 다음 시점에서 온 h gradient
            dc_next: (batch_size, hidden_size) 다음 시점에서 온 c gradient

        Returns:
            dx: (batch_size, input_size) 입력에 대한 gradient
            dh_prev: (batch_size, hidden_size) 이전 hidden gradient
            dc_prev: (batch_size, hidden_size) 이전 cell gradient
        """
        cache = self.cache
        H = self.hidden_size

        # Cell state gradient (두 경로에서 옴)
        # 1. dh_next → o_gate → tanh(c_t) → c_t
        # 2. dc_next (다음 시점에서 직접)
        do = dh_next * cache['tanh_c_t']
        dc = dh_next * cache['o_gate'] * tanh_derivative(cache['tanh_c_t'])
        dc = dc + dc_next  # 두 경로 합침

        # 각 게이트 gradient
        df = dc * cache['c_prev']
        di = dc * cache['c_tilde']
        dc_tilde = dc * cache['i_gate']

        # 이전 cell state gradient (핵심: forget gate를 통해 직접 전파)
        dc_prev = dc * cache['f_gate']

        # 활성화 함수 derivative
        df_gate = df * sigmoid_derivative(cache['f_gate'])
        di_gate = di * sigmoid_derivative(cache['i_gate'])
        dc_tilde_gate = dc_tilde * tanh_derivative(cache['c_tilde'])
        do_gate = do * sigmoid_derivative(cache['o_gate'])

        # 모든 게이트 gradient 합치기
        dgates = np.concatenate([df_gate, di_gate, dc_tilde_gate, do_gate], axis=1)

        # 가중치 gradient
        self.dW += dgates.T @ cache['concat']
        self.db += dgates.sum(axis=0)

        # Concat gradient → h_prev, x gradient
        dconcat = dgates @ self.W
        dh_prev = dconcat[:, :H]
        dx = dconcat[:, H:]

        return dx, dh_prev, dc_prev

    def zero_grad(self):
        """Gradient 초기화"""
        self.dW.fill(0)
        self.db.fill(0)


class LSTMNumPy:
    """
    전체 LSTM (여러 시점)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 레이어별 LSTM Cell
        self.cells = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.cells.append(LSTMCellNumPy(in_size, hidden_size))

    def forward(
        self,
        x: np.ndarray,
        h_0: np.ndarray = None,
        c_0: np.ndarray = None
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward pass (전체 시퀀스)

        Args:
            x: (seq_len, batch_size, input_size)
            h_0: (num_layers, batch_size, hidden_size) 초기 hidden
            c_0: (num_layers, batch_size, hidden_size) 초기 cell

        Returns:
            output: (seq_len, batch_size, hidden_size) 모든 시점의 hidden
            (h_n, c_n): 마지막 hidden/cell
        """
        seq_len, batch_size, _ = x.shape

        # 초기 상태
        if h_0 is None:
            h_0 = np.zeros((self.num_layers, batch_size, self.hidden_size))
        if c_0 is None:
            c_0 = np.zeros((self.num_layers, batch_size, self.hidden_size))

        # 출력 저장
        outputs = []
        h_states = [h_0[i] for i in range(self.num_layers)]
        c_states = [c_0[i] for i in range(self.num_layers)]

        # 시간에 따른 캐시 (backward용)
        self.time_cache = []

        for t in range(seq_len):
            layer_input = x[t]

            for layer_idx, cell in enumerate(self.cells):
                h_states[layer_idx], c_states[layer_idx] = cell.forward(
                    layer_input, h_states[layer_idx], c_states[layer_idx]
                )
                layer_input = h_states[layer_idx]

            outputs.append(h_states[-1])
            self.time_cache.append([cell.cache.copy() for cell in self.cells])

        output = np.stack(outputs, axis=0)
        h_n = np.stack(h_states, axis=0)
        c_n = np.stack(c_states, axis=0)

        return output, (h_n, c_n)

    def backward(self, doutput: np.ndarray) -> np.ndarray:
        """
        Backward pass (BPTT)

        Args:
            doutput: (seq_len, batch_size, hidden_size) 출력에 대한 gradient

        Returns:
            dx: (seq_len, batch_size, input_size) 입력에 대한 gradient
        """
        seq_len, batch_size, _ = doutput.shape

        # Gradient 초기화
        for cell in self.cells:
            cell.zero_grad()

        dx = np.zeros((seq_len, batch_size, self.input_size))

        # 레이어별 gradient 전파
        dh_next = [np.zeros((batch_size, self.hidden_size))
                   for _ in range(self.num_layers)]
        dc_next = [np.zeros((batch_size, self.hidden_size))
                   for _ in range(self.num_layers)]

        # 시간 역순
        for t in reversed(range(seq_len)):
            # 마지막 레이어에 출력 gradient 더함
            dh_next[-1] += doutput[t]

            # 레이어 역순 (깊은 레이어 → 얕은 레이어)
            for layer_idx in reversed(range(self.num_layers)):
                cell = self.cells[layer_idx]
                cell.cache = self.time_cache[t][layer_idx]

                dx_layer, dh_prev, dc_prev = cell.backward(
                    dh_next[layer_idx], dc_next[layer_idx]
                )

                # 다음 시점으로 전파
                dh_next[layer_idx] = dh_prev
                dc_next[layer_idx] = dc_prev

                # 이전 레이어로 전파
                if layer_idx > 0:
                    dh_next[layer_idx - 1] += dx_layer
                else:
                    dx[t] = dx_layer

        return dx

    def parameters(self) -> List[np.ndarray]:
        """모든 파라미터 반환"""
        params = []
        for cell in self.cells:
            params.extend([cell.W, cell.b])
        return params

    def gradients(self) -> List[np.ndarray]:
        """모든 gradient 반환"""
        grads = []
        for cell in self.cells:
            grads.extend([cell.dW, cell.db])
        return grads


def sgd_update(params: List[np.ndarray], grads: List[np.ndarray], lr: float):
    """SGD 업데이트"""
    for param, grad in zip(params, grads):
        param -= lr * grad


def clip_gradients(grads: List[np.ndarray], max_norm: float = 5.0):
    """Gradient clipping (exploding gradient 방지)"""
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for g in grads:
            g *= scale


# 간단한 테스트
def test_lstm():
    print("=== LSTM NumPy Test ===\n")

    # 하이퍼파라미터
    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20
    num_layers = 2

    # 모델
    lstm = LSTMNumPy(input_size, hidden_size, num_layers)

    # 더미 입력
    x = np.random.randn(seq_len, batch_size, input_size)

    # Forward
    output, (h_n, c_n) = lstm.forward(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"h_n shape: {h_n.shape}")
    print(f"c_n shape: {c_n.shape}")

    # Backward (마지막 출력에 대한 loss 가정)
    loss = np.sum(output[-1] ** 2)  # 더미 loss
    doutput = np.zeros_like(output)
    doutput[-1] = 2 * output[-1]

    dx = lstm.backward(doutput)

    print(f"\ndx shape: {dx.shape}")
    print(f"Gradient norms:")
    for i, (param, grad) in enumerate(zip(lstm.parameters(), lstm.gradients())):
        print(f"  Layer {i//2}, {'W' if i%2==0 else 'b'}: "
              f"param norm={np.linalg.norm(param):.4f}, "
              f"grad norm={np.linalg.norm(grad):.4f}")


def train_sequence_classification():
    """간단한 시퀀스 분류 예제"""
    print("\n=== Sequence Classification ===\n")

    np.random.seed(42)

    # 데이터: 시퀀스의 평균이 양수면 1, 음수면 0
    def generate_data(n_samples, seq_len, input_size):
        X = np.random.randn(n_samples, seq_len, input_size)
        y = (X.mean(axis=(1, 2)) > 0).astype(int)
        return X, y

    X_train, y_train = generate_data(100, 10, 5)
    X_test, y_test = generate_data(20, 10, 5)

    # 모델
    lstm = LSTMNumPy(input_size=5, hidden_size=16, num_layers=1)

    # 출력 레이어
    W_out = np.random.randn(2, 16) * 0.1
    b_out = np.zeros(2)

    lr = 0.01
    epochs = 50

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for i in range(len(X_train)):
            x = X_train[i:i+1].transpose(1, 0, 2)  # (seq, 1, input)
            target = y_train[i]

            # Forward
            output, _ = lstm.forward(x)
            last_hidden = output[-1]  # (1, hidden)

            # 분류
            logits = last_hidden @ W_out.T + b_out
            probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()

            # Loss (cross entropy)
            loss = -np.log(probs[0, target] + 1e-7)
            total_loss += loss

            # Accuracy
            pred = logits.argmax()
            correct += (pred == target)

            # Backward
            dlogits = probs.copy()
            dlogits[0, target] -= 1

            dW_out = dlogits.T @ last_hidden
            db_out = dlogits.sum(axis=0)
            dlast_hidden = dlogits @ W_out

            doutput = np.zeros_like(output)
            doutput[-1] = dlast_hidden

            lstm.backward(doutput)

            # Gradient clipping
            clip_gradients(lstm.gradients())

            # Update
            sgd_update(lstm.parameters(), lstm.gradients(), lr)
            W_out -= lr * dW_out
            b_out -= lr * db_out

        if (epoch + 1) % 10 == 0:
            acc = correct / len(X_train)
            print(f"Epoch {epoch+1}: Loss={total_loss/len(X_train):.4f}, Acc={acc:.2f}")

    # 테스트
    test_correct = 0
    for i in range(len(X_test)):
        x = X_test[i:i+1].transpose(1, 0, 2)
        target = y_test[i]

        output, _ = lstm.forward(x)
        logits = output[-1] @ W_out.T + b_out
        pred = logits.argmax()
        test_correct += (pred == target)

    print(f"\nTest Accuracy: {test_correct/len(X_test):.2f}")


if __name__ == "__main__":
    test_lstm()
    train_sequence_classification()
