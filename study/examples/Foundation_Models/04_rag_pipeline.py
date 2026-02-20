"""
Foundation Models - RAG Pipeline Implementation

Implements a simple Retrieval-Augmented Generation (RAG) pipeline.
Demonstrates document retrieval using TF-IDF and simple embeddings.
Shows prompt composition with retrieved context.

No external LLM API calls - focuses on retrieval and prompt engineering.
"""

import re
import math
from collections import Counter, defaultdict
import numpy as np


class TFIDFRetriever:
    """Simple TF-IDF based document retriever."""

    def __init__(self):
        self.documents = []
        self.vocab = {}
        self.idf = {}
        self.doc_vectors = []

    def tokenize(self, text):
        """Simple tokenization: lowercase and split."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def compute_tf(self, tokens):
        """Compute term frequency for a document."""
        tf = Counter(tokens)
        total = len(tokens)

        # Normalize
        for term in tf:
            tf[term] = tf[term] / total

        return tf

    def build_index(self, documents):
        """
        Build TF-IDF index from documents.

        Args:
            documents: List of document strings
        """
        self.documents = documents
        n_docs = len(documents)

        # Tokenize all documents
        tokenized_docs = [self.tokenize(doc) for doc in documents]

        # Build vocabulary
        all_terms = set()
        for tokens in tokenized_docs:
            all_terms.update(tokens)

        self.vocab = {term: idx for idx, term in enumerate(sorted(all_terms))}

        # Compute IDF
        df = Counter()
        for tokens in tokenized_docs:
            unique_terms = set(tokens)
            df.update(unique_terms)

        for term in self.vocab:
            # IDF = log(N / df(t))
            self.idf[term] = math.log(n_docs / (df[term] + 1))

        # Compute TF-IDF vectors for all documents
        self.doc_vectors = []
        for tokens in tokenized_docs:
            tf = self.compute_tf(tokens)
            vector = np.zeros(len(self.vocab))

            for term, freq in tf.items():
                if term in self.vocab:
                    idx = self.vocab[term]
                    vector[idx] = freq * self.idf[term]

            self.doc_vectors.append(vector)

        print(f"Indexed {n_docs} documents with vocabulary size {len(self.vocab)}")

    def retrieve(self, query, top_k=3):
        """
        Retrieve top-k most relevant documents for query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of (doc_idx, score, document) tuples
        """
        # Compute query vector
        tokens = self.tokenize(query)
        tf = self.compute_tf(tokens)

        query_vector = np.zeros(len(self.vocab))
        for term, freq in tf.items():
            if term in self.vocab:
                idx = self.vocab[term]
                query_vector[idx] = freq * self.idf[term]

        # Compute cosine similarity with all documents
        scores = []
        for doc_idx, doc_vector in enumerate(self.doc_vectors):
            # Cosine similarity
            dot_product = np.dot(query_vector, doc_vector)
            query_norm = np.linalg.norm(query_vector)
            doc_norm = np.linalg.norm(doc_vector)

            if query_norm > 0 and doc_norm > 0:
                similarity = dot_product / (query_norm * doc_norm)
            else:
                similarity = 0.0

            scores.append((doc_idx, similarity, self.documents[doc_idx]))

        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class SimpleEmbeddingRetriever:
    """Simple embedding-based retriever using random projections."""

    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        self.documents = []
        self.vocab = {}
        self.word_embeddings = {}
        self.doc_embeddings = []

    def tokenize(self, text):
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def build_index(self, documents):
        """Build simple embedding index."""
        self.documents = documents

        # Build vocabulary
        all_tokens = []
        for doc in documents:
            all_tokens.extend(self.tokenize(doc))

        unique_tokens = set(all_tokens)
        self.vocab = {token: idx for idx, token in enumerate(sorted(unique_tokens))}

        # Create random word embeddings (in practice, use pretrained)
        np.random.seed(42)
        for token in self.vocab:
            self.word_embeddings[token] = np.random.randn(self.embedding_dim)
            # Normalize
            self.word_embeddings[token] /= np.linalg.norm(self.word_embeddings[token])

        # Create document embeddings (average of word embeddings)
        self.doc_embeddings = []
        for doc in documents:
            tokens = self.tokenize(doc)
            if tokens:
                embeddings = [self.word_embeddings[t] for t in tokens if t in self.word_embeddings]
                doc_emb = np.mean(embeddings, axis=0) if embeddings else np.zeros(self.embedding_dim)
            else:
                doc_emb = np.zeros(self.embedding_dim)

            self.doc_embeddings.append(doc_emb)

        print(f"Indexed {len(documents)} documents with {self.embedding_dim}-dim embeddings")

    def retrieve(self, query, top_k=3):
        """Retrieve top-k documents by embedding similarity."""
        tokens = self.tokenize(query)

        # Compute query embedding
        embeddings = [self.word_embeddings[t] for t in tokens if t in self.word_embeddings]
        if embeddings:
            query_emb = np.mean(embeddings, axis=0)
        else:
            query_emb = np.zeros(self.embedding_dim)

        # Compute similarities
        scores = []
        for doc_idx, doc_emb in enumerate(self.doc_embeddings):
            # Cosine similarity
            similarity = np.dot(query_emb, doc_emb)
            scores.append((doc_idx, similarity, self.documents[doc_idx]))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class RAGPipeline:
    """Complete RAG pipeline with retrieval and prompt composition."""

    def __init__(self, retriever):
        self.retriever = retriever

    def generate_prompt(self, query, context_docs, system_prompt=None):
        """
        Compose RAG prompt with retrieved context.

        Args:
            query: User query
            context_docs: Retrieved documents
            system_prompt: Optional system instruction

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        # System prompt
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")

        # Context
        prompt_parts.append("Context:\n")
        for idx, (doc_idx, score, doc) in enumerate(context_docs):
            prompt_parts.append(f"[{idx+1}] {doc}\n")

        # Query
        prompt_parts.append(f"\nQuestion: {query}\n")
        prompt_parts.append("\nAnswer based on the context above:")

        return ''.join(prompt_parts)

    def query(self, query, top_k=3, system_prompt=None):
        """
        Full RAG query pipeline.

        Args:
            query: User question
            top_k: Number of documents to retrieve
            system_prompt: Optional system instruction

        Returns:
            Dictionary with retrieved docs and formatted prompt
        """
        # Retrieve relevant documents
        retrieved = self.retriever.retrieve(query, top_k=top_k)

        # Generate prompt
        prompt = self.generate_prompt(query, retrieved, system_prompt)

        return {
            'query': query,
            'retrieved_docs': retrieved,
            'prompt': prompt,
        }


# ============================================================
# Demonstrations
# ============================================================

def get_sample_documents():
    """Get sample knowledge base documents."""
    return [
        "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
        "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes organized in layers.",
        "Deep learning uses neural networks with many layers to learn hierarchical representations of data.",
        "Natural language processing (NLP) is a field of AI focused on enabling computers to understand and generate human language.",
        "Transformers are a type of neural network architecture introduced in 2017 that use self-attention mechanisms.",
        "Large language models like GPT are trained on vast amounts of text data to generate human-like text.",
        "Transfer learning involves taking a pretrained model and fine-tuning it for a specific task.",
        "Computer vision is a field of AI that enables computers to understand and interpret visual information from images and videos.",
        "Reinforcement learning is a type of machine learning where agents learn to make decisions by interacting with an environment.",
    ]


def demo_tfidf_retrieval():
    """Demonstrate TF-IDF based retrieval."""
    print("=" * 60)
    print("DEMO 1: TF-IDF Retrieval")
    print("=" * 60)

    documents = get_sample_documents()

    retriever = TFIDFRetriever()
    retriever.build_index(documents)

    # Test queries
    queries = [
        "What is Python?",
        "How do neural networks work?",
        "What are transformers?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        results = retriever.retrieve(query, top_k=3)

        for rank, (doc_idx, score, doc) in enumerate(results, 1):
            print(f"{rank}. [Score: {score:.4f}] {doc[:80]}...")


def demo_embedding_retrieval():
    """Demonstrate embedding-based retrieval."""
    print("\n" + "=" * 60)
    print("DEMO 2: Embedding-Based Retrieval")
    print("=" * 60)

    documents = get_sample_documents()

    retriever = SimpleEmbeddingRetriever(embedding_dim=64)
    retriever.build_index(documents)

    queries = [
        "programming languages",
        "AI and machine learning",
        "language models",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        results = retriever.retrieve(query, top_k=3)

        for rank, (doc_idx, score, doc) in enumerate(results, 1):
            print(f"{rank}. [Score: {score:.4f}] {doc[:80]}...")


def demo_rag_pipeline():
    """Demonstrate complete RAG pipeline."""
    print("\n" + "=" * 60)
    print("DEMO 3: Complete RAG Pipeline")
    print("=" * 60)

    documents = get_sample_documents()

    # Use TF-IDF retriever
    retriever = TFIDFRetriever()
    retriever.build_index(documents)

    # Create RAG pipeline
    rag = RAGPipeline(retriever)

    # Test query
    query = "What is the relationship between deep learning and neural networks?"

    system_prompt = "You are a helpful AI assistant. Answer questions based only on the provided context."

    result = rag.query(query, top_k=3, system_prompt=system_prompt)

    print(f"\nQuery: {result['query']}")
    print("\n" + "=" * 60)
    print("Retrieved Documents:")
    print("=" * 60)

    for rank, (doc_idx, score, doc) in enumerate(result['retrieved_docs'], 1):
        print(f"\n{rank}. [Score: {score:.4f}]")
        print(f"   {doc}")

    print("\n" + "=" * 60)
    print("Generated Prompt:")
    print("=" * 60)
    print(result['prompt'])


def demo_retrieval_comparison():
    """Compare TF-IDF vs embedding retrieval."""
    print("\n" + "=" * 60)
    print("DEMO 4: Retrieval Method Comparison")
    print("=" * 60)

    documents = get_sample_documents()

    # Build both retrievers
    tfidf = TFIDFRetriever()
    tfidf.build_index(documents)

    embedding = SimpleEmbeddingRetriever(embedding_dim=128)
    embedding.build_index(documents)

    query = "What is artificial intelligence?"

    print(f"\nQuery: {query}\n")

    # TF-IDF results
    print("TF-IDF Retrieval:")
    print("-" * 60)
    tfidf_results = tfidf.retrieve(query, top_k=3)
    for rank, (doc_idx, score, doc) in enumerate(tfidf_results, 1):
        print(f"{rank}. [Score: {score:.4f}] Doc {doc_idx}")

    # Embedding results
    print("\nEmbedding Retrieval:")
    print("-" * 60)
    emb_results = embedding.retrieve(query, top_k=3)
    for rank, (doc_idx, score, doc) in enumerate(emb_results, 1):
        print(f"{rank}. [Score: {score:.4f}] Doc {doc_idx}")


def demo_prompt_engineering():
    """Demonstrate different prompt strategies in RAG."""
    print("\n" + "=" * 60)
    print("DEMO 5: RAG Prompt Engineering")
    print("=" * 60)

    documents = get_sample_documents()
    retriever = TFIDFRetriever()
    retriever.build_index(documents)
    rag = RAGPipeline(retriever)

    query = "How does transfer learning work?"

    # Different system prompts
    prompts = {
        "Basic": "Answer the question based on the context.",
        "Detailed": "Provide a detailed answer using only information from the context. If the context doesn't contain the answer, say so.",
        "Concise": "Give a brief, one-sentence answer based on the context.",
    }

    for style, system_prompt in prompts.items():
        print(f"\n{style} Style:")
        print("-" * 60)
        result = rag.query(query, top_k=2, system_prompt=system_prompt)
        print(result['prompt'][:300] + "...\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Foundation Models: RAG Pipeline")
    print("=" * 60)

    demo_tfidf_retrieval()
    demo_embedding_retrieval()
    demo_rag_pipeline()
    demo_retrieval_comparison()
    demo_prompt_engineering()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. RAG combines retrieval with generation for grounded answers")
    print("2. TF-IDF: sparse retrieval based on term importance")
    print("3. Embeddings: dense retrieval based on semantic similarity")
    print("4. Prompt engineering: format context and query effectively")
    print("5. Retrieval quality directly impacts generation quality")
    print("=" * 60)
