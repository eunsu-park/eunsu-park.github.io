"""
14. 실전 텍스트 분류 프로젝트

감성 분석을 위한 텍스트 분류 파이프라인 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import math

print("=" * 60)
print("실전 텍스트 분류 프로젝트")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ============================================
# 1. 텍스트 전처리
# ============================================
print("\n[1] 텍스트 전처리")
print("-" * 40)

def simple_tokenizer(text):
    """간단한 토크나이저"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

# 테스트
sample = "This is a SAMPLE sentence! With punctuation."
tokens = simple_tokenizer(sample)
print(f"원문: {sample}")
print(f"토큰: {tokens}")


# ============================================
# 2. 어휘 구축
# ============================================
print("\n[2] 어휘 구축")
print("-" * 40)

class Vocabulary:
    """텍스트 어휘 사전"""
    def __init__(self, min_freq=2):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.word_freq = Counter()
        self.min_freq = min_freq

    def build(self, texts, tokenizer):
        """어휘 구축"""
        for text in texts:
            tokens = tokenizer(text)
            self.word_freq.update(tokens)

        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        print(f"총 단어 수: {len(self.word_freq)}")
        print(f"어휘 크기 (min_freq={self.min_freq}): {len(self.word2idx)}")

    def encode(self, text, tokenizer, max_len=None):
        """텍스트를 인덱스로 변환"""
        tokens = tokenizer(text)
        indices = [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]
        if max_len:
            if len(indices) > max_len:
                indices = indices[:max_len]
            else:
                indices = indices + [self.word2idx['<pad>']] * (max_len - len(indices))
        return indices

    def __len__(self):
        return len(self.word2idx)


# ============================================
# 3. 샘플 데이터셋
# ============================================
print("\n[3] 샘플 데이터셋 생성")
print("-" * 40)

# 감성 분석용 샘플 데이터
positive_samples = [
    "This movie is absolutely amazing and wonderful",
    "I love this product it is fantastic",
    "Great experience highly recommended",
    "Excellent quality and fast delivery",
    "Best purchase I have ever made",
    "Wonderful service and friendly staff",
    "I am very happy with this item",
    "Perfect product exactly what I needed",
    "Amazing value for the price",
    "Outstanding performance and quality",
    "This is the best thing ever",
    "Incredible movie I loved every minute",
    "Superb quality and great design",
    "Highly satisfied with my purchase",
    "Fantastic product works perfectly",
] * 50  # 750 samples

negative_samples = [
    "Terrible product do not buy",
    "Worst experience of my life",
    "Very disappointed with the quality",
    "Complete waste of money",
    "Poor customer service",
    "The product broke after one day",
    "I hate this movie it was boring",
    "Never buying from here again",
    "Awful quality and slow delivery",
    "Extremely bad experience",
    "This is the worst product ever",
    "Horrible movie total waste of time",
    "Very poor quality disappointed",
    "Bad product not recommended",
    "Terrible service will not return",
] * 50  # 750 samples

texts = positive_samples + negative_samples
labels = [1] * len(positive_samples) + [0] * len(negative_samples)

# 셔플
indices = np.random.permutation(len(texts))
texts = [texts[i] for i in indices]
labels = [labels[i] for i in indices]

print(f"총 샘플 수: {len(texts)}")
print(f"긍정: {sum(labels)}, 부정: {len(labels) - sum(labels)}")

# 어휘 구축
vocab = Vocabulary(min_freq=2)
vocab.build(texts, simple_tokenizer)


# ============================================
# 4. PyTorch Dataset
# ============================================
print("\n[4] PyTorch Dataset")
print("-" * 40)

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.vocab.encode(text, self.tokenizer, self.max_len)
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# 데이터 분할
train_size = int(0.8 * len(texts))
train_texts, test_texts = texts[:train_size], texts[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

train_dataset = TextDataset(train_texts, train_labels, vocab, simple_tokenizer)
test_dataset = TextDataset(test_texts, test_labels, vocab, simple_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

# 샘플 확인
sample_x, sample_y = train_dataset[0]
print(f"샘플 입력 shape: {sample_x.shape}")
print(f"샘플 라벨: {sample_y.item()}")


# ============================================
# 5. 기본 텍스트 분류기 (임베딩 + 평균)
# ============================================
print("\n[5] 기본 텍스트 분류기")
print("-" * 40)

class SimpleClassifier(nn.Module):
    """임베딩 평균 기반 분류기"""
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq, embed)
        # 평균 풀링 (패딩 제외)
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.fc(pooled)

simple_model = SimpleClassifier(len(vocab), embed_dim=64, num_classes=2)
print(f"SimpleClassifier 파라미터: {sum(p.numel() for p in simple_model.parameters()):,}")


# ============================================
# 6. LSTM 분류기
# ============================================
print("\n[6] LSTM 분류기")
print("-" * 40)

class LSTMClassifier(nn.Module):
    """양방향 LSTM 분류기"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        output, (h_n, c_n) = self.lstm(embedded)

        # 양방향 마지막 은닉 상태 결합
        forward_last = h_n[-2]
        backward_last = h_n[-1]
        combined = torch.cat([forward_last, backward_last], dim=1)

        return self.fc(combined)

lstm_model = LSTMClassifier(len(vocab), embed_dim=64, hidden_dim=128, num_classes=2)
print(f"LSTMClassifier 파라미터: {sum(p.numel() for p in lstm_model.parameters()):,}")


# ============================================
# 7. Transformer 분류기
# ============================================
print("\n[7] Transformer 분류기")
print("-" * 40)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """Transformer 인코더 기반 분류기"""
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 num_classes, max_len=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # 패딩 마스크 생성
        padding_mask = (x == 0)

        # 임베딩 + 스케일링 + 위치 인코딩
        embedded = self.embedding(x) * math.sqrt(self.embed_dim)
        embedded = self.pos_encoder(embedded)

        # Transformer 인코더
        output = self.transformer(embedded, src_key_padding_mask=padding_mask)

        # 평균 풀링 (패딩 제외)
        mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.fc(pooled)

transformer_model = TransformerClassifier(
    len(vocab), embed_dim=64, num_heads=4, num_layers=2, num_classes=2
)
print(f"TransformerClassifier 파라미터: {sum(p.numel() for p in transformer_model.parameters()):,}")


# ============================================
# 8. 학습 함수
# ============================================
print("\n[8] 학습 파이프라인")
print("-" * 40)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()

        # 기울기 클리핑 (RNN에 중요)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100. * correct / total


def train_model(model, train_loader, test_loader, epochs=10, lr=1e-3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}: Train Loss={train_loss:.4f}, Acc={train_acc:.1f}% | "
                  f"Test Loss={test_loss:.4f}, Acc={test_acc:.1f}%")

    return history


# ============================================
# 9. 모델 비교 학습
# ============================================
print("\n[9] 모델 비교 학습")
print("-" * 40)

# 모델 재생성 (학습 전 상태)
models = {
    'Simple': SimpleClassifier(len(vocab), embed_dim=64, num_classes=2),
    'LSTM': LSTMClassifier(len(vocab), embed_dim=64, hidden_dim=128, num_classes=2),
    'Transformer': TransformerClassifier(len(vocab), embed_dim=64, num_heads=4,
                                          num_layers=2, num_classes=2)
}

results = {}
for name, model in models.items():
    print(f"\n--- {name} 학습 ---")
    history = train_model(model, train_loader, test_loader, epochs=15)
    results[name] = history
    print(f"{name} 최종 테스트 정확도: {history['test_acc'][-1]:.1f}%")


# ============================================
# 10. 결과 시각화
# ============================================
print("\n[10] 결과 시각화")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 정확도
for name, history in results.items():
    axes[0].plot(history['test_acc'], label=f"{name} (final={history['test_acc'][-1]:.1f}%)")
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Test Accuracy (%)')
axes[0].set_title('Model Comparison - Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 손실
for name, history in results.items():
    axes[1].plot(history['test_loss'], label=name)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Test Loss')
axes[1].set_title('Model Comparison - Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('text_classification_comparison.png', dpi=100)
plt.close()
print("그래프 저장: text_classification_comparison.png")


# ============================================
# 11. 추론 함수
# ============================================
print("\n[11] 추론 테스트")
print("-" * 40)

def predict_sentiment(model, text, vocab, tokenizer, device):
    """텍스트 감성 예측"""
    model.eval()
    encoded = vocab.encode(text, tokenizer, max_len=50)
    tensor = torch.tensor(encoded).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()

    sentiment = 'Positive' if pred == 1 else 'Negative'
    confidence = prob[0, pred].item()

    return sentiment, confidence

# 테스트 문장
test_sentences = [
    "This product is amazing and I love it",
    "Terrible quality waste of money",
    "It's okay nothing special",
    "Best purchase ever highly recommended",
    "Very disappointed will not buy again",
]

# LSTM 모델로 예측
lstm_model = models['LSTM']
print("\nLSTM 모델 예측:")
for sentence in test_sentences:
    sentiment, conf = predict_sentiment(lstm_model, sentence, vocab, simple_tokenizer, device)
    print(f"  [{sentiment:8s}] ({conf*100:5.1f}%) {sentence}")


# ============================================
# 12. Attention 시각화
# ============================================
print("\n[12] Attention 가중치 분석")
print("-" * 40)

class AttentionLSTM(nn.Module):
    """Attention이 있는 LSTM"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, return_attention=False):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)

        # Attention
        attn_weights = torch.softmax(self.attention(output).squeeze(-1), dim=1)
        context = (output * attn_weights.unsqueeze(-1)).sum(dim=1)

        logits = self.fc(context)

        if return_attention:
            return logits, attn_weights
        return logits

attn_model = AttentionLSTM(len(vocab), embed_dim=64, hidden_dim=128, num_classes=2)
print(f"AttentionLSTM 파라미터: {sum(p.numel() for p in attn_model.parameters()):,}")

# 학습
print("\nAttentionLSTM 학습:")
attn_model = attn_model.to(device)
history = train_model(attn_model, train_loader, test_loader, epochs=10)


# Attention 시각화
def visualize_attention(model, text, vocab, tokenizer, device):
    model.eval()
    tokens = tokenizer(text)
    encoded = vocab.encode(text, tokenizer, max_len=len(tokens))
    tensor = torch.tensor(encoded).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attn = model(tensor, return_attention=True)
        pred = logits.argmax(dim=1).item()
        prob = F.softmax(logits, dim=1)[0, pred].item()

    attn = attn[0].cpu().numpy()[:len(tokens)]
    sentiment = 'Positive' if pred == 1 else 'Negative'

    return tokens, attn, sentiment, prob

# 시각화
sample_text = "This movie is absolutely amazing and wonderful"
tokens, attn, sentiment, prob = visualize_attention(attn_model, sample_text, vocab, simple_tokenizer, device)

print(f"\n문장: {sample_text}")
print(f"예측: {sentiment} ({prob*100:.1f}%)")
print("\nAttention 가중치:")
for token, weight in zip(tokens, attn):
    bar = '█' * int(weight * 50)
    print(f"  {token:12s} {weight:.3f} {bar}")


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("텍스트 분류 정리")
print("=" * 60)

summary = """
텍스트 분류 파이프라인:
    1. 토큰화: 텍스트 → 단어 리스트
    2. 어휘 구축: 단어 → 인덱스 매핑
    3. 인코딩: 텍스트 → 텐서
    4. 모델: 임베딩 → 인코더 → 분류

모델 비교:
    - Simple (임베딩 평균): 빠름, 간단
    - LSTM: 순서 정보 활용, 안정적
    - Transformer: 병렬화, 긴 시퀀스

핵심 코드:
    # 어휘 구축
    vocab = Vocabulary(min_freq=2)
    vocab.build(texts, tokenizer)

    # LSTM 분류기
    lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
    _, (h_n, _) = lstm(embedded)
    combined = torch.cat([h_n[-2], h_n[-1]], dim=1)

    # Transformer 분류기
    encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    output = encoder(embedded, src_key_padding_mask=padding_mask)

학습 팁:
    - 기울기 클리핑 (RNN에 필수)
    - Dropout (과적합 방지)
    - 적절한 패딩 처리

다음 단계:
    - HuggingFace Transformers
    - BERT/GPT 파인튜닝
    - 대규모 데이터셋 (IMDb)
"""
print(summary)
print("=" * 60)
