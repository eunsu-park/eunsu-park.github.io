"""
01. NLP 기초 - 토큰화 예제

텍스트 전처리와 토큰화 기법 실습
"""

import re
from collections import Counter

print("=" * 60)
print("NLP 기초: 토큰화")
print("=" * 60)


# ============================================
# 1. 기본 전처리
# ============================================
print("\n[1] 기본 전처리")
print("-" * 40)

def preprocess(text):
    """기본 텍스트 전처리"""
    # 소문자 변환
    text = text.lower()
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    # 다중 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()
    return text

sample = "Hello, World! This is NLP   processing."
cleaned = preprocess(sample)
print(f"원본: {sample}")
print(f"전처리: {cleaned}")


# ============================================
# 2. 단어 토큰화
# ============================================
print("\n[2] 단어 토큰화")
print("-" * 40)

def simple_tokenize(text):
    """공백 기반 토큰화"""
    return text.lower().split()

text = "I love natural language processing"
tokens = simple_tokenize(text)
print(f"텍스트: {text}")
print(f"토큰: {tokens}")

# NLTK 토큰화 (설치 필요: pip install nltk)
try:
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import word_tokenize

    text2 = "I don't like it. It's not good!"
    nltk_tokens = word_tokenize(text2)
    print(f"\nNLTK 토큰화: {nltk_tokens}")
except ImportError:
    print("\nNLTK 미설치 (pip install nltk)")


# ============================================
# 3. 어휘 사전 구축
# ============================================
print("\n[3] 어휘 사전 구축")
print("-" * 40)

class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}
        self.word_freq = Counter()
        self.min_freq = min_freq

    def build(self, texts):
        """텍스트 리스트로 어휘 구축"""
        for text in texts:
            tokens = simple_tokenize(text)
            self.word_freq.update(tokens)

        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, text):
        """텍스트를 인덱스로 변환"""
        tokens = simple_tokenize(text)
        return [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]

    def decode(self, indices):
        """인덱스를 토큰으로 변환"""
        return [self.idx2word.get(i, '<unk>') for i in indices]

    def __len__(self):
        return len(self.word2idx)

# 어휘 구축
texts = [
    "I love machine learning",
    "Machine learning is amazing",
    "Deep learning is a subset of machine learning",
    "I love deep learning"
]

vocab = Vocabulary(min_freq=1)
vocab.build(texts)

print(f"어휘 크기: {len(vocab)}")
print(f"상위 빈도 단어: {vocab.word_freq.most_common(5)}")

# 인코딩/디코딩
test_text = "I love learning"
encoded = vocab.encode(test_text)
decoded = vocab.decode(encoded)
print(f"\n원본: {test_text}")
print(f"인코딩: {encoded}")
print(f"디코딩: {decoded}")


# ============================================
# 4. 패딩
# ============================================
print("\n[4] 패딩")
print("-" * 40)

def pad_sequences(sequences, max_len=None, pad_value=0):
    """시퀀스 패딩"""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        if len(seq) > max_len:
            padded.append(seq[:max_len])
        else:
            padded.append(seq + [pad_value] * (max_len - len(seq)))
    return padded

sequences = [
    vocab.encode("I love learning"),
    vocab.encode("Machine learning is great"),
    vocab.encode("Deep")
]

print("원본 시퀀스:")
for seq in sequences:
    print(f"  {seq}")

padded = pad_sequences(sequences, max_len=5)
print("\n패딩 후:")
for seq in padded:
    print(f"  {seq}")


# ============================================
# 5. HuggingFace 토크나이저 (설치 필요)
# ============================================
print("\n[5] HuggingFace 토크나이저")
print("-" * 40)

try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = "Hello, how are you?"
    encoded = tokenizer(text, return_tensors='pt')

    print(f"텍스트: {text}")
    print(f"토큰: {tokenizer.tokenize(text)}")
    print(f"input_ids: {encoded['input_ids'].tolist()}")
    print(f"attention_mask: {encoded['attention_mask'].tolist()}")

    # 배치 인코딩
    texts = ["Hello world", "How are you?", "I'm fine"]
    batch_encoded = tokenizer(texts, padding=True, return_tensors='pt')
    print(f"\n배치 인코딩 shape: {batch_encoded['input_ids'].shape}")

except ImportError:
    print("transformers 미설치 (pip install transformers)")


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("토큰화 정리")
print("=" * 60)

summary = """
토큰화 파이프라인:
    텍스트 → 전처리 → 토큰화 → 어휘 매핑 → 패딩 → 텐서

주요 기법:
    - 단어 토큰화: 공백/구두점 기준 분리
    - 서브워드 토큰화: BPE, WordPiece, SentencePiece
    - 어휘 사전: word2idx, idx2word 매핑

HuggingFace 사용:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded = tokenizer(text, padding=True, return_tensors='pt')
"""
print(summary)
