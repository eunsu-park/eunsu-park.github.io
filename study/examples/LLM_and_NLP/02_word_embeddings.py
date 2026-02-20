"""
02. Word2Vec과 GloVe - 단어 임베딩 예제

단어 임베딩 학습과 활용
"""

import numpy as np

print("=" * 60)
print("단어 임베딩")
print("=" * 60)


# ============================================
# 1. 코사인 유사도
# ============================================
print("\n[1] 코사인 유사도")
print("-" * 40)

def cosine_similarity(v1, v2):
    """두 벡터의 코사인 유사도"""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / norm if norm > 0 else 0

# 예시 벡터
vec_king = np.array([0.5, 0.3, 0.8, 0.1])
vec_queen = np.array([0.5, 0.4, 0.7, 0.2])
vec_apple = np.array([-0.2, 0.9, 0.1, 0.5])

print(f"king-queen 유사도: {cosine_similarity(vec_king, vec_queen):.4f}")
print(f"king-apple 유사도: {cosine_similarity(vec_king, vec_apple):.4f}")


# ============================================
# 2. 간단한 임베딩 레이어 (PyTorch)
# ============================================
print("\n[2] PyTorch 임베딩 레이어")
print("-" * 40)

try:
    import torch
    import torch.nn as nn

    # 임베딩 레이어
    vocab_size = 100
    embed_dim = 64
    embedding = nn.Embedding(vocab_size, embed_dim)

    # 입력: 단어 인덱스
    input_ids = torch.tensor([1, 5, 10, 20])
    embedded = embedding(input_ids)

    print(f"입력 shape: {input_ids.shape}")
    print(f"출력 shape: {embedded.shape}")
    print(f"임베딩 가중치 shape: {embedding.weight.shape}")

except ImportError:
    print("PyTorch 미설치")


# ============================================
# 3. Gensim Word2Vec
# ============================================
print("\n[3] Gensim Word2Vec")
print("-" * 40)

try:
    from gensim.models import Word2Vec

    # 샘플 코퍼스
    sentences = [
        ["i", "love", "machine", "learning"],
        ["machine", "learning", "is", "fun"],
        ["deep", "learning", "is", "great"],
        ["i", "love", "deep", "learning"],
        ["neural", "networks", "are", "powerful"],
        ["deep", "neural", "networks", "learn", "features"],
    ]

    # Word2Vec 학습
    model = Word2Vec(
        sentences,
        vector_size=50,    # 임베딩 차원
        window=3,          # 컨텍스트 윈도우
        min_count=1,       # 최소 빈도
        sg=1,              # Skip-gram (0=CBOW)
        epochs=100
    )

    # 유사 단어
    print("'learning' 유사 단어:")
    similar = model.wv.most_similar("learning", topn=3)
    for word, score in similar:
        print(f"  {word}: {score:.4f}")

    # 벡터 가져오기
    vec = model.wv["learning"]
    print(f"\n'learning' 벡터 shape: {vec.shape}")

    # 저장/로드
    model.save("word2vec_demo.model")
    loaded = Word2Vec.load("word2vec_demo.model")
    print("모델 저장/로드 완료")

    # 정리
    import os
    os.remove("word2vec_demo.model")

except ImportError:
    print("gensim 미설치 (pip install gensim)")


# ============================================
# 4. 사전학습 임베딩 사용
# ============================================
print("\n[4] 사전학습 임베딩 적용")
print("-" * 40)

try:
    import torch
    import torch.nn as nn

    # 가상의 사전학습 임베딩 (실제로는 GloVe 등 로드)
    pretrained_embeddings = torch.randn(1000, 100)  # vocab_size=1000, dim=100

    # 임베딩 레이어에 적용
    embedding = nn.Embedding.from_pretrained(
        pretrained_embeddings,
        freeze=False,  # True면 학습 안 함
        padding_idx=0
    )

    print(f"사전학습 임베딩 shape: {pretrained_embeddings.shape}")
    print(f"freeze=False: 파인튜닝 가능")

    # 분류 모델에 적용
    class TextClassifier(nn.Module):
        def __init__(self, pretrained_emb, num_classes):
            super().__init__()
            self.embedding = nn.Embedding.from_pretrained(pretrained_emb, freeze=False)
            self.fc = nn.Linear(pretrained_emb.shape[1], num_classes)

        def forward(self, x):
            embedded = self.embedding(x)  # (batch, seq, embed)
            pooled = embedded.mean(dim=1)  # 평균 풀링
            return self.fc(pooled)

    model = TextClassifier(pretrained_embeddings, num_classes=2)
    print(f"분류 모델 생성 완료")

except ImportError:
    print("PyTorch 미설치")


# ============================================
# 5. 단어 유추 (Word Analogy)
# ============================================
print("\n[5] 단어 유추")
print("-" * 40)

def word_analogy(word_a, word_b, word_c, embeddings, word2idx, idx2word, topk=3):
    """
    a : b = c : ?
    예: king : queen = man : woman
    """
    # 벡터 가져오기
    vec_a = embeddings[word2idx[word_a]]
    vec_b = embeddings[word2idx[word_b]]
    vec_c = embeddings[word2idx[word_c]]

    # 유추 벡터: b - a + c
    target = vec_b - vec_a + vec_c

    # 유사도 계산
    similarities = np.dot(embeddings, target) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target)
    )

    # 상위 k개 (a, b, c 제외)
    exclude = {word2idx[word_a], word2idx[word_b], word2idx[word_c]}
    results = []
    for idx in np.argsort(similarities)[::-1]:
        if idx not in exclude:
            results.append((idx2word[idx], similarities[idx]))
        if len(results) >= topk:
            break

    return results

# 예시 (가상 데이터)
vocab = ["king", "queen", "man", "woman", "prince", "princess", "boy", "girl"]
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}

# 가상 임베딩 (실제로는 학습된 임베딩 사용)
np.random.seed(42)
embeddings = np.random.randn(len(vocab), 50)
# 의미적 관계 시뮬레이션
embeddings[word2idx["queen"]] = embeddings[word2idx["king"]] + np.array([0.1] * 50)
embeddings[word2idx["woman"]] = embeddings[word2idx["man"]] + np.array([0.1] * 50)

result = word_analogy("king", "queen", "man", embeddings, word2idx, idx2word)
print(f"king : queen = man : ?")
for word, score in result:
    print(f"  {word}: {score:.4f}")


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("단어 임베딩 정리")
print("=" * 60)

summary = """
핵심 개념:
    - 분산 표현: 단어를 밀집 벡터로 표현
    - Word2Vec: Skip-gram, CBOW
    - GloVe: 동시 출현 통계 기반

사용법:
    # Gensim Word2Vec
    model = Word2Vec(sentences, vector_size=100, window=5)
    similar = model.wv.most_similar("word", topn=5)

    # PyTorch 임베딩
    embedding = nn.Embedding.from_pretrained(vectors, freeze=False)

단어 연산:
    king - queen + man ≈ woman
"""
print(summary)
