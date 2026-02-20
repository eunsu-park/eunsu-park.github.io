"""
Foundation Models - BPE Tokenizer Implementation

Implements Byte Pair Encoding (BPE) from scratch.
Demonstrates vocabulary building, merge rules, encoding/decoding.
Compares with character-level tokenization.

No external dependencies except collections.
"""

import re
from collections import Counter, defaultdict


class CharacterTokenizer:
    """Simple character-level tokenizer for comparison."""

    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}

    def build_vocab(self, text):
        """Build vocabulary from unique characters."""
        unique_chars = sorted(set(text))
        self.vocab = {char: idx for idx, char in enumerate(unique_chars)}
        self.inv_vocab = {idx: char for char, idx in self.vocab.items()}

        return len(self.vocab)

    def encode(self, text):
        """Encode text to list of token IDs."""
        return [self.vocab[char] for char in text if char in self.vocab]

    def decode(self, tokens):
        """Decode token IDs back to text."""
        return ''.join([self.inv_vocab[tok] for tok in tokens])


class BPETokenizer:
    """Byte Pair Encoding tokenizer implementation."""

    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size
        self.merges = []  # List of merge operations
        self.vocab = {}   # Token to ID mapping
        self.inv_vocab = {}  # ID to token mapping

    def get_stats(self, words):
        """
        Count frequency of adjacent pairs in word list.

        Args:
            words: Dictionary of {word: frequency}

        Returns:
            Counter of pair frequencies
        """
        pairs = Counter()

        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq

        return pairs

    def merge_pair(self, pair, words):
        """
        Merge all occurrences of pair in words.

        Args:
            pair: Tuple of (token1, token2) to merge
            words: Dictionary of {word: frequency}

        Returns:
            New words dictionary with merged pairs
        """
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        # Compile pattern for efficiency
        pattern = re.escape(bigram)

        for word, freq in words.items():
            # Replace pair with merged token
            new_word = re.sub(pattern, replacement, word)
            new_words[new_word] = freq

        return new_words

    def build_vocab(self, text, verbose=False):
        """
        Build BPE vocabulary from text.

        Args:
            text: Training text
            verbose: Print merge operations

        Returns:
            Final vocabulary size
        """
        # Initialize with character-level tokens
        # Each word is space-separated characters + end marker
        words = defaultdict(int)

        for word in text.split():
            # Add space between characters and end-of-word marker
            word_chars = ' '.join(list(word)) + ' </w>'
            words[word_chars] += 1

        # Get initial vocabulary (unique characters)
        initial_vocab = set()
        for word in words.keys():
            initial_vocab.update(word.split())

        print(f"Initial vocabulary size (characters): {len(initial_vocab)}")
        print(f"Target vocabulary size: {self.vocab_size}")

        # Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(initial_vocab)
        print(f"Number of merges to perform: {num_merges}\n")

        for i in range(num_merges):
            # Get pair statistics
            pairs = self.get_stats(words)

            if not pairs:
                print(f"No more pairs to merge at iteration {i}")
                break

            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)
            freq = pairs[best_pair]

            # Merge the pair
            words = self.merge_pair(best_pair, words)
            self.merges.append(best_pair)

            if verbose and (i < 10 or i % 50 == 0):
                print(f"Merge {i+1}: {best_pair[0]} + {best_pair[1]} "
                      f"= {''.join(best_pair)} (freq={freq})")

        # Build final vocabulary from current state
        final_vocab = set()
        for word in words.keys():
            final_vocab.update(word.split())

        # Create token-to-ID mapping
        self.vocab = {token: idx for idx, token in enumerate(sorted(final_vocab))}
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}

        print(f"\nFinal vocabulary size: {len(self.vocab)}")
        print(f"Total merges performed: {len(self.merges)}")

        return len(self.vocab)

    def encode_word(self, word):
        """
        Encode a single word using learned merges.

        Args:
            word: String to encode

        Returns:
            List of tokens
        """
        # Start with character-level
        tokens = list(word) + ['</w>']

        # Apply merges in order
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == merge:
                    # Merge the pair
                    tokens = tokens[:i] + [''.join(merge)] + tokens[i + 2:]
                else:
                    i += 1

        return tokens

    def encode(self, text):
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        words = text.split()
        token_ids = []

        for word in words:
            tokens = self.encode_word(word)
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # Unknown token - use character fallback
                    for char in token:
                        if char in self.vocab:
                            token_ids.append(self.vocab[char])

        return token_ids

    def decode(self, token_ids):
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        tokens = [self.inv_vocab[tid] for tid in token_ids if tid in self.inv_vocab]
        text = ''.join(tokens)

        # Remove end-of-word markers and add spaces
        text = text.replace('</w>', ' ')

        return text.strip()

    def get_token_stats(self):
        """Get statistics about learned tokens."""
        token_lengths = Counter()

        for token in self.vocab.keys():
            # Don't count special markers
            if token != '</w>':
                token_lengths[len(token)] += 1

        return token_lengths


# ============================================================
# Demonstrations
# ============================================================

def demo_character_tokenizer():
    """Demonstrate simple character-level tokenization."""
    print("=" * 60)
    print("DEMO 1: Character-Level Tokenizer")
    print("=" * 60)

    text = "Hello world! Machine learning is amazing."

    tokenizer = CharacterTokenizer()
    vocab_size = tokenizer.build_vocab(text)

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Vocabulary: {sorted(tokenizer.vocab.keys())[:20]}")

    # Encode
    encoded = tokenizer.encode(text)
    print(f"\nOriginal text: {text}")
    print(f"Encoded ({len(encoded)} tokens): {encoded[:30]}")

    # Decode
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    print(f"Matches original: {decoded == text}")


def demo_bpe_basic():
    """Demonstrate basic BPE tokenization."""
    print("\n" + "=" * 60)
    print("DEMO 2: BPE Tokenizer - Basic")
    print("=" * 60)

    # Simple training corpus
    text = "low lower lowest higher high highest new newer newest"

    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.build_vocab(text, verbose=True)

    # Show learned merges
    print("\n" + "-" * 60)
    print("First 10 learned merges:")
    print("-" * 60)
    for i, (a, b) in enumerate(tokenizer.merges[:10]):
        print(f"{i+1}. {a} + {b} → {''.join([a, b])}")


def demo_bpe_encoding():
    """Demonstrate BPE encoding and decoding."""
    print("\n" + "=" * 60)
    print("DEMO 3: BPE Encoding/Decoding")
    print("=" * 60)

    # Training corpus
    corpus = """
    the quick brown fox jumps over the lazy dog
    the dog runs fast and the fox runs faster
    machine learning models learn from data
    deep learning uses neural networks
    """

    tokenizer = BPETokenizer(vocab_size=150)
    tokenizer.build_vocab(corpus, verbose=False)

    # Test encoding
    test_sentences = [
        "the fox runs",
        "machine learning",
        "deep neural networks",
        "the quick dog",
    ]

    print("\n" + "-" * 60)
    print("Encoding examples:")
    print("-" * 60)

    for sentence in test_sentences:
        tokens = []
        for word in sentence.split():
            word_tokens = tokenizer.encode_word(word)
            tokens.extend(word_tokens)

        token_ids = tokenizer.encode(sentence)
        decoded = tokenizer.decode(token_ids)

        print(f"\nSentence: {sentence}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs ({len(token_ids)}): {token_ids}")
        print(f"Decoded: {decoded}")


def demo_compression_comparison():
    """Compare compression between character and BPE tokenization."""
    print("\n" + "=" * 60)
    print("DEMO 4: Compression Comparison")
    print("=" * 60)

    corpus = """
    Natural language processing enables computers to understand human language.
    Machine learning algorithms can learn patterns from data automatically.
    Deep learning models use neural networks with multiple layers.
    Transformers have revolutionized natural language understanding.
    Large language models can generate coherent and contextual text.
    """ * 5  # Repeat for more data

    # Character tokenizer
    char_tok = CharacterTokenizer()
    char_tok.build_vocab(corpus)
    char_encoded = char_tok.encode(corpus)

    # BPE tokenizer
    bpe_tok = BPETokenizer(vocab_size=200)
    bpe_tok.build_vocab(corpus, verbose=False)
    bpe_encoded = bpe_tok.encode(corpus)

    print("\n" + "-" * 60)
    print("Comparison:")
    print("-" * 60)
    print(f"Original text length: {len(corpus)} characters")
    print(f"\nCharacter tokenizer:")
    print(f"  Vocabulary size: {len(char_tok.vocab)}")
    print(f"  Encoded length: {len(char_encoded)} tokens")
    print(f"  Compression ratio: {len(corpus)/len(char_encoded):.2f}x")

    print(f"\nBPE tokenizer:")
    print(f"  Vocabulary size: {len(bpe_tok.vocab)}")
    print(f"  Encoded length: {len(bpe_encoded)} tokens")
    print(f"  Compression ratio: {len(corpus)/len(bpe_encoded):.2f}x")

    reduction = (1 - len(bpe_encoded) / len(char_encoded)) * 100
    print(f"\nBPE reduces tokens by {reduction:.1f}% vs character-level")


def demo_token_statistics():
    """Analyze learned token statistics."""
    print("\n" + "=" * 60)
    print("DEMO 5: Token Statistics")
    print("=" * 60)

    corpus = """
    Large language models are trained on massive amounts of text data.
    These models learn statistical patterns and relationships in language.
    Tokenization is a crucial preprocessing step for language models.
    BPE allows models to handle unknown words through subword units.
    Common words are represented as single tokens for efficiency.
    Rare words are broken into multiple subword tokens.
    """ * 10

    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.build_vocab(corpus, verbose=False)

    # Get token length statistics
    token_lengths = tokenizer.get_token_stats()

    print("\n" + "-" * 60)
    print("Token length distribution:")
    print("-" * 60)
    for length in sorted(token_lengths.keys()):
        count = token_lengths[length]
        bar = '█' * (count // 5)
        print(f"Length {length}: {count:3d} tokens {bar}")

    # Show example tokens by length
    print("\n" + "-" * 60)
    print("Example tokens by length:")
    print("-" * 60)

    tokens_by_length = defaultdict(list)
    for token in tokenizer.vocab.keys():
        if token != '</w>':
            tokens_by_length[len(token)].append(token)

    for length in sorted(tokens_by_length.keys())[:8]:
        examples = tokens_by_length[length][:10]
        print(f"Length {length}: {examples}")


def demo_merge_frequency():
    """Analyze merge operation frequencies."""
    print("\n" + "=" * 60)
    print("DEMO 6: Most Important Merges")
    print("=" * 60)

    corpus = "the the the and and or if then else while for " * 20

    tokenizer = BPETokenizer(vocab_size=80)
    tokenizer.build_vocab(corpus, verbose=False)

    print("\n" + "-" * 60)
    print("Top 20 merges (in order):")
    print("-" * 60)

    for i, (a, b) in enumerate(tokenizer.merges[:20]):
        merged = ''.join([a, b])
        print(f"{i+1:2d}. '{a}' + '{b}' → '{merged}'")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Foundation Models: BPE Tokenizer")
    print("=" * 60)

    demo_character_tokenizer()
    demo_bpe_basic()
    demo_bpe_encoding()
    demo_compression_comparison()
    demo_token_statistics()
    demo_merge_frequency()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. BPE builds vocabulary by iteratively merging frequent pairs")
    print("2. Balances vocabulary size with sequence length")
    print("3. Handles unknown words through subword decomposition")
    print("4. Common words → single tokens, rare words → multiple tokens")
    print("5. Reduces sequence length by 30-50% vs character-level")
    print("=" * 60)
