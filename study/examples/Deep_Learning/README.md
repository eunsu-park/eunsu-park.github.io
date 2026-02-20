# Deep_Learning 예제

Deep_Learning 폴더의 레슨에 해당하는 실행 가능한 예제 코드입니다.

## 폴더 구조

```
examples/
├── pytorch/                      # PyTorch 구현
│   ├── 01_tensor_autograd.py     # 텐서, 자동 미분
│   ├── 02_neural_network.py      # MLP, XOR 문제
│   ├── 03_backprop.py            # 역전파 시각화
│   ├── 04_training.py            # 학습 루프, 옵티마이저
│   ├── 05_cnn_basic.py           # CNN 기초
│   ├── 06_cnn_advanced.py        # ResNet, VGG
│   ├── 07_transfer_learning.py   # 전이 학습
│   ├── 08_rnn_basic.py           # RNN
│   ├── 09_lstm_gru.py            # LSTM, GRU
│   ├── 10_transformer.py         # Transformer
│   └── ...
│
└── numpy/                        # NumPy 순수 구현
    ├── 01_tensor_basics.py       # 텐서, 수동 미분
    ├── 02_neural_network_scratch.py  # MLP 순전파
    ├── 03_backprop_scratch.py    # 역전파 직접 구현
    ├── 04_training_scratch.py    # SGD 직접 구현
    └── 05_conv2d_scratch.py      # 합성곱 직접 구현
```

## PyTorch vs NumPy 구현 비교

| 레슨 | PyTorch | NumPy | 비교 포인트 |
|------|---------|-------|------------|
| 01 | 자동 미분 | 수동 미분 | `backward()` vs 직접 계산 |
| 02 | `nn.Module` | 클래스 직접 | 순전파 구조 |
| 03 | `loss.backward()` | 체인 룰 구현 | 역전파 원리 |
| 04 | `optim.Adam` | SGD 직접 구현 | 옵티마이저 원리 |
| 05 | `nn.Conv2d` | for 루프 | 합성곱 연산 |
| 06+ | PyTorch only | - | 복잡도로 인해 생략 |

## 실행 방법

### 환경 설정

```bash
# 가상환경 생성
python -m venv dl-env
source dl-env/bin/activate

# PyTorch 설치 (CUDA 지원)
pip install torch torchvision torchaudio

# 기타 패키지
pip install numpy matplotlib
```

### 실행

```bash
# PyTorch 예제
cd Deep_Learning/examples/pytorch
python 01_tensor_autograd.py

# NumPy 예제
cd Deep_Learning/examples/numpy
python 01_tensor_basics.py
```

## 학습 순서

### 1단계: 기초 (PyTorch + NumPy 비교)
```
pytorch/01 ←→ numpy/01  # 텐서, 미분
pytorch/02 ←→ numpy/02  # 신경망 순전파
pytorch/03 ←→ numpy/03  # 역전파
pytorch/04 ←→ numpy/04  # 학습
```

### 2단계: CNN (기초만 NumPy)
```
pytorch/05 ←→ numpy/05  # CNN 기초 (합성곱 이해)
pytorch/06              # CNN 심화 (PyTorch only)
pytorch/07              # 전이 학습 (PyTorch only)
```

### 3단계: 시퀀스 모델 (PyTorch only)
```
pytorch/08  # RNN
pytorch/09  # LSTM, GRU
pytorch/10  # Transformer
```

## NumPy 구현의 학습 가치

1. **01-02**: 텐서 연산과 순전파가 단순 행렬 곱임을 이해
2. **03**: 역전파가 체인 룰의 반복 적용임을 이해
3. **04**: 경사 하강법의 가중치 업데이트 원리 이해
4. **05**: 합성곱이 필터의 슬라이딩 윈도우임을 이해

## NumPy 구현이 어려운 시점

- **CNN 심화**: Skip Connection, Batch Normalization
- **RNN/LSTM**: 시간 축 역전파(BPTT), 게이트 구조
- **Transformer**: Multi-Head Attention, 위치 인코딩

→ 이 시점부터 PyTorch만 사용하여 실전에 집중

## 참고 자료

- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [CS231n (Stanford CNN)](http://cs231n.stanford.edu/)
- [3Blue1Brown Neural Networks](https://www.3blue1brown.com/topics/neural-networks)
