"""
Foundation Models - Evaluation Metrics

Implements common evaluation metrics for language models.
Demonstrates BLEU, ROUGE, perplexity, exact match, and F1 score.
Shows how to evaluate model outputs on different tasks.

No external dependencies except numpy.
"""

import re
import math
from collections import Counter, defaultdict
import numpy as np


def tokenize(text):
    """Simple word tokenization."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


def ngrams(tokens, n):
    """Generate n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference, candidate, max_n=4, weights=None):
    """
    Compute BLEU score.

    BLEU = BP × exp(sum(w_n × log(p_n)))

    where p_n is n-gram precision and BP is brevity penalty.

    Args:
        reference: Reference text (string or list of tokens)
        candidate: Candidate text (string or list of tokens)
        max_n: Maximum n-gram order
        weights: Weights for each n-gram order (default: uniform)

    Returns:
        BLEU score (0-1)
    """
    if isinstance(reference, str):
        reference = tokenize(reference)
    if isinstance(candidate, str):
        candidate = tokenize(candidate)

    if weights is None:
        weights = [1.0 / max_n] * max_n

    # Compute n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(ngrams(reference, n))
        cand_ngrams = Counter(ngrams(candidate, n))

        # Clipped count: min(count, ref_count)
        clipped_count = 0
        total_count = 0

        for ng in cand_ngrams:
            clipped_count += min(cand_ngrams[ng], ref_ngrams[ng])
            total_count += cand_ngrams[ng]

        if total_count > 0:
            precision = clipped_count / total_count
        else:
            precision = 0

        precisions.append(precision)

    # Brevity penalty
    ref_len = len(reference)
    cand_len = len(candidate)

    if cand_len > ref_len:
        bp = 1
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0

    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        log_precision = sum(w * math.log(p) for w, p in zip(weights, precisions))
        bleu = bp * math.exp(log_precision)
    else:
        bleu = 0

    return bleu


def rouge_l_score(reference, candidate):
    """
    Compute ROUGE-L score based on longest common subsequence.

    ROUGE-L = F1 score of LCS

    Args:
        reference: Reference text
        candidate: Candidate text

    Returns:
        Dictionary with precision, recall, and f1
    """
    if isinstance(reference, str):
        reference = tokenize(reference)
    if isinstance(candidate, str):
        candidate = tokenize(candidate)

    # Compute LCS length using dynamic programming
    m, n = len(reference), len(candidate)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i-1] == candidate[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_length = dp[m][n]

    # Compute precision, recall, F1
    if n > 0:
        precision = lcs_length / n
    else:
        precision = 0

    if m > 0:
        recall = lcs_length / m
    else:
        recall = 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'lcs_length': lcs_length
    }


def exact_match(reference, candidate, normalize=True):
    """
    Compute exact match score.

    Args:
        reference: Reference answer
        candidate: Predicted answer
        normalize: Whether to normalize (lowercase, strip)

    Returns:
        1 if exact match, 0 otherwise
    """
    if normalize:
        reference = reference.lower().strip()
        candidate = candidate.lower().strip()

    return 1 if reference == candidate else 0


def f1_token_score(reference, candidate):
    """
    Compute token-level F1 score (for span-based QA).

    Args:
        reference: Reference text
        candidate: Predicted text

    Returns:
        F1 score
    """
    ref_tokens = set(tokenize(reference))
    cand_tokens = set(tokenize(candidate))

    if not cand_tokens:
        return 0.0

    common = ref_tokens & cand_tokens

    if not common:
        return 0.0

    precision = len(common) / len(cand_tokens)
    recall = len(common) / len(ref_tokens)

    f1 = 2 * precision * recall / (precision + recall)

    return f1


def perplexity(log_probs):
    """
    Compute perplexity from log probabilities.

    PPL = exp(-1/N × sum(log P(w_i)))

    Args:
        log_probs: List of log probabilities for each token

    Returns:
        Perplexity value
    """
    if not log_probs:
        return float('inf')

    avg_log_prob = sum(log_probs) / len(log_probs)
    return math.exp(-avg_log_prob)


def classification_metrics(y_true, y_pred):
    """
    Compute classification metrics: accuracy, precision, recall, F1.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    assert len(y_true) == len(y_pred), "Length mismatch"

    # True/False Positives/Negatives (assuming binary)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    # Metrics
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    }


# ============================================================
# Demonstrations
# ============================================================

def demo_bleu():
    """Demonstrate BLEU score computation."""
    print("=" * 60)
    print("DEMO 1: BLEU Score")
    print("=" * 60)

    reference = "The cat is sitting on the mat"
    candidates = [
        "The cat is sitting on the mat",  # Perfect match
        "The cat sits on the mat",         # Close
        "A cat is on the mat",             # Moderate
        "There is a cat",                  # Poor
    ]

    print(f"\nReference: {reference}\n")

    for cand in candidates:
        score = bleu_score(reference, cand, max_n=4)
        print(f"Candidate: {cand}")
        print(f"BLEU score: {score:.4f}\n")


def demo_rouge():
    """Demonstrate ROUGE-L score computation."""
    print("\n" + "=" * 60)
    print("DEMO 2: ROUGE-L Score")
    print("=" * 60)

    reference = "The quick brown fox jumps over the lazy dog"
    candidates = [
        "The quick brown fox jumps over the lazy dog",
        "The brown fox jumps over the dog",
        "A quick fox jumped over a dog",
        "The cat sleeps",
    ]

    print(f"\nReference: {reference}\n")

    for cand in candidates:
        scores = rouge_l_score(reference, cand)
        print(f"Candidate: {cand}")
        print(f"ROUGE-L: Precision={scores['precision']:.3f}, "
              f"Recall={scores['recall']:.3f}, F1={scores['f1']:.3f}\n")


def demo_exact_match():
    """Demonstrate exact match evaluation."""
    print("\n" + "=" * 60)
    print("DEMO 3: Exact Match")
    print("=" * 60)

    qa_pairs = [
        ("Paris", "Paris"),
        ("Paris", "paris"),
        ("Paris", "Paris, France"),
        ("1776", "1776"),
        ("1776", "1776.0"),
    ]

    print("\nQuestion Answering Evaluation:\n")

    for ref, pred in qa_pairs:
        em = exact_match(ref, pred, normalize=True)
        print(f"Reference: '{ref}' | Prediction: '{pred}' | EM: {em}")


def demo_f1_token():
    """Demonstrate token F1 score."""
    print("\n" + "=" * 60)
    print("DEMO 4: Token-level F1 Score")
    print("=" * 60)

    qa_pairs = [
        ("Barack Obama", "Barack Obama"),
        ("Barack Obama", "Barack Hussein Obama"),
        ("Barack Obama", "Obama"),
        ("New York City", "New York"),
    ]

    print("\nSpan-based QA Evaluation:\n")

    for ref, pred in qa_pairs:
        f1 = f1_token_score(ref, pred)
        em = exact_match(ref, pred, normalize=True)
        print(f"Reference: '{ref}'")
        print(f"Prediction: '{pred}'")
        print(f"EM: {em} | F1: {f1:.3f}\n")


def demo_perplexity():
    """Demonstrate perplexity computation."""
    print("\n" + "=" * 60)
    print("DEMO 5: Perplexity")
    print("=" * 60)

    # Simulate log probabilities for different model qualities
    # Good model: high probabilities (less negative log probs)
    good_model = [-0.1, -0.2, -0.15, -0.3, -0.1, -0.2]

    # Medium model
    medium_model = [-1.0, -1.5, -1.2, -0.8, -1.3, -1.1]

    # Poor model: low probabilities (very negative log probs)
    poor_model = [-3.0, -4.0, -3.5, -4.2, -3.8, -3.9]

    print("\nPerplexity for different model qualities:")
    print("-" * 60)

    for name, log_probs in [("Good", good_model), ("Medium", medium_model), ("Poor", poor_model)]:
        ppl = perplexity(log_probs)
        avg_prob = math.exp(sum(log_probs) / len(log_probs))

        print(f"{name} model:")
        print(f"  Avg log prob: {sum(log_probs)/len(log_probs):.3f}")
        print(f"  Avg prob: {avg_prob:.3f}")
        print(f"  Perplexity: {ppl:.2f}\n")


def demo_classification():
    """Demonstrate classification metrics."""
    print("\n" + "=" * 60)
    print("DEMO 6: Classification Metrics")
    print("=" * 60)

    # Simulate sentiment classification
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]

    metrics = classification_metrics(y_true, y_pred)

    print("\nBinary Classification Results:")
    print("-" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 Score:  {metrics['f1']:.3f}")

    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['tp']}  FP: {metrics['fp']}")
    print(f"  FN: {metrics['fn']}  TN: {metrics['tn']}")


def demo_summarization_eval():
    """Evaluate summarization task."""
    print("\n" + "=" * 60)
    print("DEMO 7: Summarization Evaluation")
    print("=" * 60)

    reference = "Machine learning is a subset of artificial intelligence. It enables systems to learn from data."

    summaries = [
        "Machine learning is part of AI and allows systems to learn from data.",
        "ML is a type of AI that learns from data.",
        "Artificial intelligence includes machine learning.",
    ]

    print(f"\nReference: {reference}\n")

    for i, summary in enumerate(summaries, 1):
        bleu = bleu_score(reference, summary, max_n=2)
        rouge = rouge_l_score(reference, summary)

        print(f"Summary {i}: {summary}")
        print(f"  BLEU-2: {bleu:.3f}")
        print(f"  ROUGE-L F1: {rouge['f1']:.3f}\n")


def demo_translation_eval():
    """Evaluate machine translation."""
    print("\n" + "=" * 60)
    print("DEMO 8: Translation Evaluation")
    print("=" * 60)

    # Example: French to English
    reference = "The cat is on the table"

    translations = [
        "The cat is on the table",        # Perfect
        "The cat sits on the table",      # Good
        "A cat is on a table",            # Medium
        "Cat table on",                   # Poor
    ]

    print(f"\nReference: {reference}\n")

    for i, trans in enumerate(translations, 1):
        bleu = bleu_score(reference, trans, max_n=4)
        rouge = rouge_l_score(reference, trans)

        print(f"Translation {i}: {trans}")
        print(f"  BLEU-4: {bleu:.3f}")
        print(f"  ROUGE-L F1: {rouge['f1']:.3f}\n")


def demo_multi_reference():
    """Evaluate with multiple references."""
    print("\n" + "=" * 60)
    print("DEMO 9: Multi-Reference Evaluation")
    print("=" * 60)

    references = [
        "It is raining heavily",
        "The rain is very strong",
        "Heavy rainfall is occurring",
    ]

    candidate = "It is raining a lot"

    print(f"\nCandidate: {candidate}\n")
    print("References:")
    for i, ref in enumerate(references, 1):
        print(f"  {i}. {ref}")

    print("\nScores against each reference:")
    print("-" * 60)

    bleu_scores = []
    rouge_scores = []

    for i, ref in enumerate(references, 1):
        bleu = bleu_score(ref, candidate, max_n=2)
        rouge = rouge_l_score(ref, candidate)

        bleu_scores.append(bleu)
        rouge_scores.append(rouge['f1'])

        print(f"Reference {i}: BLEU={bleu:.3f}, ROUGE-L F1={rouge['f1']:.3f}")

    # Best score across references
    print(f"\nBest scores:")
    print(f"  BLEU: {max(bleu_scores):.3f}")
    print(f"  ROUGE-L F1: {max(rouge_scores):.3f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Foundation Models: Evaluation Metrics")
    print("=" * 60)

    demo_bleu()
    demo_rouge()
    demo_exact_match()
    demo_f1_token()
    demo_perplexity()
    demo_classification()
    demo_summarization_eval()
    demo_translation_eval()
    demo_multi_reference()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. BLEU: n-gram overlap (translation, generation)")
    print("2. ROUGE-L: Longest common subsequence (summarization)")
    print("3. Exact Match: Binary correctness (QA)")
    print("4. Token F1: Partial credit for overlap (span QA)")
    print("5. Perplexity: Model uncertainty (lower is better)")
    print("6. Classification: Accuracy, precision, recall, F1")
    print("7. Multi-reference: Take max/avg across references")
    print("8. Choose metrics appropriate for task")
    print("=" * 60)
