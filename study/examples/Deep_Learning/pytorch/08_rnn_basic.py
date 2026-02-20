"""
08. RNN 기초 (Recurrent Neural Networks)

순환 신경망의 기본 개념과 PyTorch 구현을 학습합니다.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("PyTorch RNN 기초")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")


# ============================================
# 1. RNN 기본 이해
# ============================================
print("\n[1] RNN 기본 이해")
print("-" * 40)

# 단순 RNN 셀 수동 구현
class SimpleRNNCell:
    """RNN 셀 수동 구현 (이해용)"""
    def __init__(self, input_size, hidden_size):
        # 가중치 초기화
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b = np.zeros(hidden_size)

    def forward(self, x, h_prev):
        """
        x: 현재 입력 (input_size,)
        h_prev: 이전 은닉 상태 (hidden_size,)
        """
        h_new = np.tanh(x @ self.W_xh + h_prev @ self.W_hh + self.b)
        return h_new

# 테스트
cell = SimpleRNNCell(input_size=3, hidden_size=5)
h = np.zeros(5)

print("수동 RNN 셀 순전파:")
for t in range(4):
    x = np.random.randn(3)
    h = cell.forward(x, h)
    print(f"  t={t}: h = {h[:3]}...")


# ============================================
# 2. PyTorch nn.RNN
# ============================================
print("\n[2] PyTorch nn.RNN")
print("-" * 40)

# RNN 레이어 생성
rnn = nn.RNN(
    input_size=10,    # 입력 특성 차원
    hidden_size=20,   # 은닉 상태 차원
    num_layers=2,     # RNN 층 수
    batch_first=True, # 입력: (batch, seq, feature)
    dropout=0.1       # 층 간 드롭아웃
)

# 입력 생성
batch_size = 4
seq_len = 8
x = torch.randn(batch_size, seq_len, 10)

# 순전파
output, h_n = rnn(x)

print(f"입력: {x.shape}")
print(f"output (모든 시간 은닉상태): {output.shape}")
print(f"h_n (마지막 은닉상태): {h_n.shape}")

# 초기 은닉 상태 지정
h0 = torch.zeros(2, batch_size, 20)  # (num_layers, batch, hidden)
output, h_n = rnn(x, h0)
print(f"\n초기 상태 지정: h0 shape = {h0.shape}")


# ============================================
# 3. 양방향 RNN
# ============================================
print("\n[3] 양방향 RNN")
print("-" * 40)

rnn_bi = nn.RNN(
    input_size=10,
    hidden_size=20,
    num_layers=1,
    batch_first=True,
    bidirectional=True
)

output_bi, h_n_bi = rnn_bi(x)

print(f"양방향 RNN:")
print(f"  output: {output_bi.shape}")  # (batch, seq, hidden*2)
print(f"  h_n: {h_n_bi.shape}")        # (2, batch, hidden)

# 정방향/역방향 분리
forward_out = output_bi[:, :, :20]
backward_out = output_bi[:, :, 20:]
print(f"  정방향 출력: {forward_out.shape}")
print(f"  역방향 출력: {backward_out.shape}")


# ============================================
# 4. RNN 분류기
# ============================================
print("\n[4] RNN 분류기")
print("-" * 40)

class RNNClassifier(nn.Module):
    """시퀀스 분류용 RNN"""
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq, features)
        output, h_n = self.rnn(x)

        # 마지막 층의 마지막 시간 은닉 상태
        last_hidden = h_n[-1]  # (batch, hidden)
        out = self.fc(last_hidden)
        return out

# 테스트
model = RNNClassifier(input_size=10, hidden_size=32, num_classes=5)
x = torch.randn(8, 15, 10)  # 8 샘플, 15 스텝, 10 특성
out = model(x)
print(f"분류기 입력: {x.shape}")
print(f"분류기 출력: {out.shape}")


# ============================================
# 5. 시계열 예측 (사인파)
# ============================================
print("\n[5] 시계열 예측 (사인파)")
print("-" * 40)

# 데이터 생성
def generate_sin_data(seq_len=50, n_samples=1000):
    X = []
    y = []
    for _ in range(n_samples):
        start = np.random.uniform(0, 2*np.pi)
        seq = np.sin(np.linspace(start, start + 4*np.pi, seq_len + 1))
        X.append(seq[:-1].reshape(-1, 1))
        y.append(seq[-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train, y_train = generate_sin_data(seq_len=50, n_samples=1000)
X_test, y_test = generate_sin_data(seq_len=50, n_samples=200)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

print(f"훈련 데이터: X={X_train.shape}, y={y_train.shape}")
print(f"테스트 데이터: X={X_test.shape}, y={y_test.shape}")

# 모델
class SinPredictor(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1]).squeeze(-1)

model = SinPredictor(hidden_size=32).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 학습
from torch.utils.data import DataLoader, TensorDataset

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64, shuffle=True
)

losses = []
for epoch in range(50):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(X_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()

        # 기울기 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        epoch_loss += loss.item()

    losses.append(epoch_loss / len(train_loader))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {losses[-1]:.6f}")

# 테스트
model.eval()
with torch.no_grad():
    X_test_dev = X_test.to(device)
    pred_test = model(X_test_dev)
    test_loss = criterion(pred_test, y_test.to(device))
    print(f"\n테스트 MSE: {test_loss.item():.6f}")

# 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test.numpy()[:100], pred_test.cpu().numpy()[:100], alpha=0.5)
plt.plot([-1, 1], [-1, 1], 'r--')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Prediction vs True')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rnn_sin_prediction.png', dpi=100)
plt.close()
print("그래프 저장: rnn_sin_prediction.png")


# ============================================
# 6. Many-to-Many RNN
# ============================================
print("\n[6] Many-to-Many RNN")
print("-" * 40)

class Seq2SeqRNN(nn.Module):
    """시퀀스 → 시퀀스"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        # 모든 시간 단계에 FC 적용
        out = self.fc(output)
        return out

model_s2s = Seq2SeqRNN(10, 20, 5)
x = torch.randn(4, 8, 10)
out = model_s2s(x)
print(f"Seq2Seq 입력: {x.shape}")
print(f"Seq2Seq 출력: {out.shape}")  # (4, 8, 5)


# ============================================
# 7. 가변 길이 시퀀스 처리
# ============================================
print("\n[7] 가변 길이 시퀀스")
print("-" * 40)

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 다양한 길이의 시퀀스 (패딩됨)
sequences = [
    torch.randn(5, 10),   # 길이 5
    torch.randn(3, 10),   # 길이 3
    torch.randn(7, 10),   # 길이 7
]
lengths = torch.tensor([5, 3, 7])

# 패딩 (가장 긴 시퀀스에 맞춤)
max_len = max(lengths)
padded = torch.zeros(3, max_len, 10)
for i, seq in enumerate(sequences):
    padded[i, :len(seq)] = seq

print(f"패딩된 시퀀스: {padded.shape}")
print(f"실제 길이: {lengths}")

# 패킹
rnn = nn.RNN(10, 20, batch_first=True)
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
packed_output, h_n = rnn(packed)

# 언패킹
output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
print(f"언패킹된 출력: {output.shape}")


# ============================================
# 8. 기울기 소실 시연
# ============================================
print("\n[8] 기울기 소실 시연")
print("-" * 40)

def check_gradients(model, seq_len):
    """시퀀스 길이에 따른 기울기 확인"""
    model.train()
    x = torch.randn(1, seq_len, 1, requires_grad=True)
    output, h_n = model.rnn(x)
    loss = h_n.sum()
    loss.backward()

    # 첫 번째 가중치의 기울기 크기
    grad_norm = model.rnn.weight_ih_l0.grad.norm().item()
    return grad_norm

model = SinPredictor(hidden_size=32)

print("시퀀스 길이에 따른 기울기 크기:")
for seq_len in [10, 50, 100, 200]:
    grad = check_gradients(model, seq_len)
    print(f"  길이 {seq_len:3d}: 기울기 norm = {grad:.6f}")


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("RNN 기초 정리")
print("=" * 60)

summary = """
RNN 핵심:
    h(t) = tanh(W_xh × x(t) + W_hh × h(t-1) + b)

PyTorch RNN:
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    output, h_n = rnn(x)
    # output: (batch, seq, hidden) - 모든 시간
    # h_n: (layers, batch, hidden) - 마지막만

분류 패턴:
    # 마지막 은닉 상태 사용
    output = fc(h_n[-1])

Seq2Seq 패턴:
    # 모든 시간 은닉 상태 사용
    output = fc(rnn_output)

주의사항:
1. 기울기 클리핑 사용
2. 긴 시퀀스 → LSTM/GRU 사용
3. batch_first 확인
4. 가변 길이 → pack_padded_sequence
"""
print(summary)
print("=" * 60)
