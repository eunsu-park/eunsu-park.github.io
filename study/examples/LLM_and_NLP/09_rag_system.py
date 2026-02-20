"""
09. RAG (Retrieval-Augmented Generation) 예제

문서 검색 + LLM 생성 결합
"""

import numpy as np

print("=" * 60)
print("RAG 시스템")
print("=" * 60)


# ============================================
# 1. 간단한 RAG 구현 (NumPy만 사용)
# ============================================
print("\n[1] 간단한 RAG (NumPy)")
print("-" * 40)

class SimpleVectorStore:
    """간단한 벡터 저장소"""
    def __init__(self):
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents, embeddings):
        self.documents = documents
        self.embeddings = np.array(embeddings)

    def search(self, query_embedding, top_k=3):
        """코사인 유사도로 검색"""
        query = np.array(query_embedding)

        # 코사인 유사도
        similarities = np.dot(self.embeddings, query) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query)
        )

        # 상위 k개
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_indices]


# 예시 문서
documents = [
    "Python is a high-level programming language known for its readability.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing deals with text and speech.",
    "Computer vision enables machines to interpret images."
]

# 가상 임베딩 (실제로는 모델 사용)
np.random.seed(42)
embeddings = np.random.randn(len(documents), 128)

# 벡터 저장소
store = SimpleVectorStore()
store.add_documents(documents, embeddings)

# 검색
query_embedding = np.random.randn(128)
results = store.search(query_embedding, top_k=2)

print("검색 결과:")
for doc, score in results:
    print(f"  [{score:.4f}] {doc[:50]}...")


# ============================================
# 2. Sentence Transformers + RAG
# ============================================
print("\n[2] Sentence Transformers RAG")
print("-" * 40)

try:
    from sentence_transformers import SentenceTransformer

    # 임베딩 모델
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 문서 임베딩
    doc_embeddings = model.encode(documents)
    print(f"문서 임베딩 shape: {doc_embeddings.shape}")

    # 쿼리
    query = "What is machine learning?"
    query_embedding = model.encode(query)

    # 검색
    store = SimpleVectorStore()
    store.add_documents(documents, doc_embeddings)
    results = store.search(query_embedding, top_k=2)

    print(f"\n쿼리: {query}")
    print("검색 결과:")
    for doc, score in results:
        print(f"  [{score:.4f}] {doc}")

except ImportError:
    print("sentence-transformers 미설치")


# ============================================
# 3. 청킹 (Chunking)
# ============================================
print("\n[3] 텍스트 청킹")
print("-" * 40)

def chunk_text(text, chunk_size=100, overlap=20):
    """오버랩이 있는 청킹"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

long_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines,
as opposed to natural intelligence displayed by animals including humans.
AI research has been defined as the field of study of intelligent agents,
which refers to any system that perceives its environment and takes actions
that maximize its chance of achieving its goals. The term "artificial intelligence"
had previously been used to describe machines that mimic and display "human"
cognitive skills that are associated with the human mind, such as "learning" and
"problem-solving". This definition has since been rejected by major AI researchers
who now describe AI in terms of rationality and acting rationally.
"""

chunks = chunk_text(long_text, chunk_size=150, overlap=30)
print(f"원본 길이: {len(long_text)} chars")
print(f"청크 수: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"  청크 {i+1}: {chunk[:50]}...")


# ============================================
# 4. 완전한 RAG 파이프라인
# ============================================
print("\n[4] 완전한 RAG 파이프라인")
print("-" * 40)

class RAGPipeline:
    """RAG 파이프라인"""

    def __init__(self, embedding_model=None):
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.embedding_model = embedding_model

    def add_documents(self, documents, chunk_size=200, overlap=50):
        """문서 추가 및 청킹"""
        self.documents = documents

        # 청킹
        for doc in documents:
            doc_chunks = chunk_text(doc, chunk_size, overlap)
            self.chunks.extend(doc_chunks)

        # 임베딩
        if self.embedding_model:
            self.embeddings = self.embedding_model.encode(self.chunks)
        else:
            # 가상 임베딩
            self.embeddings = np.random.randn(len(self.chunks), 128)

        print(f"문서 {len(documents)}개 → 청크 {len(self.chunks)}개")

    def retrieve(self, query, top_k=3):
        """관련 청크 검색"""
        if self.embedding_model:
            query_emb = self.embedding_model.encode(query)
        else:
            query_emb = np.random.randn(128)

        # 코사인 유사도
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices]

    def generate(self, query, context):
        """프롬프트 구성 (실제로는 LLM 호출)"""
        prompt = f"""Answer based on the context:

Context:
{context}

Question: {query}

Answer:"""
        return prompt

    def query(self, question, top_k=3):
        """RAG 쿼리"""
        # 검색
        relevant_chunks = self.retrieve(question, top_k)
        context = "\n\n".join(relevant_chunks)

        # 프롬프트 생성
        prompt = self.generate(question, context)

        return {
            "question": question,
            "context": context,
            "prompt": prompt
        }


# RAG 파이프라인 테스트
rag = RAGPipeline()
rag.add_documents([long_text])

result = rag.query("What is artificial intelligence?", top_k=2)
print(f"\n질문: {result['question']}")
print(f"컨텍스트 길이: {len(result['context'])} chars")
print(f"프롬프트 미리보기:\n{result['prompt'][:200]}...")


# ============================================
# 5. OpenAI RAG (API 필요)
# ============================================
print("\n[5] OpenAI RAG 예제 (코드만)")
print("-" * 40)

openai_rag_code = '''
from openai import OpenAI
from sentence_transformers import SentenceTransformer

class OpenAIRAG:
    def __init__(self):
        self.client = OpenAI()
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents):
        self.documents = documents
        self.embeddings = self.embed_model.encode(documents)

    def search(self, query, top_k=3):
        query_emb = self.embed_model.encode(query)
        similarities = cosine_similarity([query_emb], self.embeddings)[0]
        top_idx = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_idx]

    def query(self, question, top_k=3):
        # 검색
        relevant = self.search(question, top_k)
        context = "\\n\\n".join(relevant)

        # LLM 호출
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer based on the context."},
                {"role": "user", "content": f"Context:\\n{context}\\n\\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content
'''
print(openai_rag_code)


# ============================================
# 정리
# ============================================
print("\n" + "=" * 60)
print("RAG 정리")
print("=" * 60)

summary = """
RAG 파이프라인:
    1. 문서 → 청킹 → 임베딩 → 벡터 DB 저장
    2. 쿼리 → 임베딩 → 유사 문서 검색
    3. 쿼리 + 문서 → LLM → 답변

핵심 코드:
    # 임베딩
    embeddings = model.encode(documents)

    # 검색
    similarities = cosine_similarity([query_emb], embeddings)
    top_docs = documents[top_indices]

    # 생성
    prompt = f"Context: {context}\\nQuestion: {query}"
    response = llm.generate(prompt)
"""
print(summary)
