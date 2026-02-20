"""
11. 벡터 데이터베이스 예제

Chroma, FAISS를 사용한 벡터 검색
"""

import numpy as np

print("=" * 60)
print("벡터 데이터베이스")
print("=" * 60)


# ============================================
# 1. 기본 벡터 검색 (NumPy)
# ============================================
print("\n[1] NumPy 벡터 검색")
print("-" * 40)

def cosine_similarity(query, vectors):
    """코사인 유사도 계산"""
    query_norm = query / np.linalg.norm(query)
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.dot(vectors_norm, query_norm)

# 샘플 데이터
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Deep learning is a subset of ML",
    "JavaScript is for web development",
    "Data science involves statistics"
]

# 가상 임베딩
np.random.seed(42)
embeddings = np.random.randn(len(documents), 128)

# 검색
query_embedding = np.random.randn(128)
similarities = cosine_similarity(query_embedding, embeddings)

# 상위 결과
top_k = 3
top_indices = np.argsort(similarities)[-top_k:][::-1]

print("검색 결과:")
for idx in top_indices:
    print(f"  [{similarities[idx]:.4f}] {documents[idx]}")


# ============================================
# 2. Chroma DB
# ============================================
print("\n[2] Chroma DB")
print("-" * 40)

try:
    import chromadb

    # 클라이언트 (메모리)
    client = chromadb.Client()

    # 컬렉션 생성
    collection = client.create_collection(
        name="demo_collection",
        metadata={"description": "Demo collection"}
    )

    # 문서 추가
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=[{"source": "demo"} for _ in documents]
    )

    print(f"컬렉션 생성: {collection.name}")
    print(f"문서 수: {collection.count()}")

    # 검색
    results = collection.query(
        query_texts=["What is Python?"],
        n_results=3
    )

    print("\nChroma 검색 결과:")
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        print(f"  [{dist:.4f}] {doc}")

    # 메타데이터 필터링
    filtered = collection.query(
        query_texts=["programming"],
        n_results=2,
        where={"source": "demo"}
    )
    print(f"\n필터링 결과: {len(filtered['documents'][0])}개")

except ImportError:
    print("chromadb 미설치 (pip install chromadb)")


# ============================================
# 3. FAISS
# ============================================
print("\n[3] FAISS")
print("-" * 40)

try:
    import faiss

    # 인덱스 생성
    dimension = 128
    index = faiss.IndexFlatL2(dimension)  # L2 거리

    # 벡터 추가
    vectors = np.random.randn(1000, dimension).astype('float32')
    index.add(vectors)

    print(f"인덱스 생성: {index.ntotal} 벡터")

    # 검색
    query = np.random.randn(1, dimension).astype('float32')
    distances, indices = index.search(query, k=5)

    print(f"검색 결과 (상위 5개):")
    print(f"  인덱스: {indices[0]}")
    print(f"  거리: {distances[0]}")

    # IVF 인덱스 (대규모용)
    nlist = 10  # 클러스터 수
    quantizer = faiss.IndexFlatL2(dimension)
    ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

    # 학습 및 추가
    ivf_index.train(vectors)
    ivf_index.add(vectors)
    ivf_index.nprobe = 3  # 검색할 클러스터 수

    print(f"\nIVF 인덱스: {ivf_index.ntotal} 벡터, {nlist} 클러스터")

    # 저장/로드
    faiss.write_index(index, "demo_index.faiss")
    loaded_index = faiss.read_index("demo_index.faiss")
    print(f"인덱스 저장/로드 완료")

    import os
    os.remove("demo_index.faiss")

except ImportError:
    print("faiss 미설치 (pip install faiss-cpu)")


# ============================================
# 4. Sentence Transformers + Chroma
# ============================================
print("\n[4] Sentence Transformers + Chroma")
print("-" * 40)

try:
    import chromadb
    from chromadb.utils import embedding_functions

    # Sentence Transformer 임베딩 함수
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # 클라이언트
    client = chromadb.Client()

    # 컬렉션 (임베딩 함수 지정)
    collection = client.create_collection(
        name="semantic_search",
        embedding_function=embedding_fn
    )

    # 문서 추가 (임베딩 자동 생성)
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )

    # 시맨틱 검색
    results = collection.query(
        query_texts=["How to learn programming?"],
        n_results=3
    )

    print("시맨틱 검색 결과:")
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        print(f"  [{dist:.4f}] {doc}")

except ImportError as e:
    print(f"필요 패키지 미설치: {e}")


# ============================================
# 5. LangChain + Chroma
# ============================================
print("\n[5] LangChain + Chroma (코드)")
print("-" * 40)

langchain_chroma = '''
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 임베딩
embeddings = OpenAIEmbeddings()

# 벡터 스토어 생성
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 검색
docs = vectorstore.similarity_search("What is Python?", k=3)

# Retriever로 변환
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("programming languages")

# 메타데이터와 함께 생성
from langchain.schema import Document

docs_with_meta = [
    Document(page_content=text, metadata={"source": f"doc_{i}"})
    for i, text in enumerate(texts)
]

vectorstore = Chroma.from_documents(
    documents=docs_with_meta,
    embedding=embeddings
)
'''
print(langchain_chroma)


# ============================================
# 6. 인덱스 타입 비교
# ============================================
print("\n[6] FAISS 인덱스 타입 비교")
print("-" * 40)

index_comparison = """
| 인덱스 타입 | 정확도 | 속도 | 메모리 | 사용 시점 |
|------------|--------|------|--------|----------|
| IndexFlatL2| 100%   | 느림 | 높음   | 소규모 (<100K) |
| IndexIVF   | 95%+   | 빠름 | 중간   | 중규모 |
| IndexHNSW  | 98%+   | 매우빠름| 높음 | 대규모, 실시간 |
| IndexPQ    | 90%+   | 빠름 | 낮음   | 메모리 제한 |
"""
print(index_comparison)

faiss_indexes = '''
import faiss

# Flat (정확)
index = faiss.IndexFlatL2(dim)

# IVF (클러스터링)
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist=100)
index.train(vectors)

# HNSW (그래프 기반)
index = faiss.IndexHNSWFlat(dim, 32)

# PQ (압축)
index = faiss.IndexPQ(dim, m=8, nbits=8)
index.train(vectors)
'''
print(faiss_indexes)


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("벡터 DB 정리")
print("=" * 60)

summary = """
선택 가이드:
    - 개발/프로토타입: Chroma
    - 대규모 로컬: FAISS
    - 프로덕션 관리형: Pinecone

핵심 코드:
    # Chroma
    collection = client.create_collection("name")
    collection.add(documents=texts, ids=ids)
    results = collection.query(query_texts=["query"], n_results=5)

    # FAISS
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    distances, indices = index.search(query, k=5)

    # LangChain
    vectorstore = Chroma.from_texts(texts, embeddings)
    retriever = vectorstore.as_retriever()
"""
print(summary)
