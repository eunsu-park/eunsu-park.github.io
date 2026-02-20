"""
Information Theory for Machine Learning

This script demonstrates key information theory concepts:
1. Entropy - measure of uncertainty
2. Cross-entropy and KL divergence
3. Mutual information
4. Connection to ML loss functions
5. ELBO (Evidence Lower Bound) visualization

Author: Math for AI Examples
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple


def entropy_demo():
    """
    Demonstrate entropy as a measure of uncertainty.

    Entropy: H(X) = -Σ p(x) log p(x)

    Properties:
    - H(X) >= 0 (non-negative)
    - H(X) = 0 when X is deterministic
    - H(X) is maximum when X is uniform
    """
    print("\n" + "="*60)
    print("1. Entropy - Measure of Uncertainty")
    print("="*60)

    def compute_entropy(probs: np.ndarray) -> float:
        """Compute Shannon entropy (base 2)."""
        # Filter out zero probabilities to avoid log(0)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    # Example 1: Binary random variable
    print("\n--- Binary Random Variable ---")
    p_values = np.linspace(0.01, 0.99, 100)
    entropies = []

    for p in p_values:
        probs = np.array([p, 1-p])
        H = compute_entropy(probs)
        entropies.append(H)

    print(f"Entropy when p=0.5 (max): {max(entropies):.4f} bits")
    print(f"Entropy when p→0 or p→1 (min): ~0 bits")

    # Example 2: Different distributions
    print("\n--- Comparing Different Distributions ---")

    # Uniform distribution (max entropy)
    n = 8
    uniform = np.ones(n) / n
    H_uniform = compute_entropy(uniform)
    print(f"Uniform distribution (n={n}): H = {H_uniform:.4f} bits")

    # Peaked distribution (low entropy)
    peaked = np.array([0.7, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02])
    H_peaked = compute_entropy(peaked)
    print(f"Peaked distribution: H = {H_peaked:.4f} bits")

    # Deterministic (zero entropy)
    deterministic = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    H_deterministic = compute_entropy(deterministic)
    print(f"Deterministic: H = {H_deterministic:.4f} bits")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Binary entropy curve
    ax1.plot(p_values, entropies, linewidth=3, color='blue')
    ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Maximum at p=0.5')
    ax1.set_xlabel('p (probability of outcome 1)', fontsize=12)
    ax1.set_ylabel('Entropy (bits)', fontsize=12)
    ax1.set_title('Binary Entropy Function', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: Different distributions
    x = np.arange(n)
    width = 0.25

    ax2.bar(x - width, uniform, width, label=f'Uniform (H={H_uniform:.2f})',
            alpha=0.7, color='blue')
    ax2.bar(x, peaked, width, label=f'Peaked (H={H_peaked:.2f})',
            alpha=0.7, color='orange')
    ax2.bar(x + width, deterministic, width, label=f'Deterministic (H={H_deterministic:.2f})',
            alpha=0.7, color='green')

    ax2.set_xlabel('Outcome', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title('Entropy of Different Distributions', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('entropy.png', dpi=150, bbox_inches='tight')
    print("\nSaved entropy visualization to 'entropy.png'")


def cross_entropy_kl_divergence():
    """
    Demonstrate cross-entropy and KL divergence.

    Cross-Entropy: H(P,Q) = -Σ p(x) log q(x)
    KL Divergence: D_KL(P||Q) = Σ p(x) log(p(x)/q(x)) = H(P,Q) - H(P)

    KL divergence measures how much Q differs from P.
    """
    print("\n" + "="*60)
    print("2. Cross-Entropy and KL Divergence")
    print("="*60)

    def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
        """Compute cross-entropy H(P,Q)."""
        return -np.sum(p * np.log(q + 1e-10))

    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence D_KL(P||Q)."""
        return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

    # True distribution P
    p = np.array([0.1, 0.2, 0.3, 0.25, 0.1, 0.05])

    # Various approximate distributions Q
    q1 = np.array([0.15, 0.2, 0.25, 0.25, 0.1, 0.05])  # Close to P
    q2 = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])      # Moderately different
    q3 = np.array([0.05, 0.05, 0.1, 0.1, 0.3, 0.4])    # Very different
    q_uniform = np.ones(6) / 6                          # Uniform

    distributions = [
        ('P (true)', p),
        ('Q1 (close)', q1),
        ('Q2 (moderate)', q2),
        ('Q3 (far)', q3),
        ('Uniform', q_uniform)
    ]

    H_p = -np.sum(p * np.log(p))

    print(f"\nTrue distribution P: {p}")
    print(f"Entropy H(P) = {H_p:.4f}")
    print("\n" + "-"*60)

    results = []
    for name, q in distributions[1:]:  # Skip P itself
        ce = cross_entropy(p, q)
        kl = kl_divergence(p, q)
        results.append((name, q, ce, kl))

        print(f"\n{name}: {q}")
        print(f"  Cross-Entropy H(P,Q) = {ce:.4f}")
        print(f"  KL Divergence D_KL(P||Q) = {kl:.4f}")
        print(f"  Relationship: H(P,Q) = H(P) + D_KL(P||Q)")
        print(f"              {ce:.4f} = {H_p:.4f} + {kl:.4f}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Distributions
    x = np.arange(len(p))
    width = 0.15

    ax1.bar(x - 2*width, p, width, label='P (true)', alpha=0.8, color='blue')
    for i, (name, q, _, _) in enumerate(results):
        ax1.bar(x + (i-1)*width, q, width, label=name, alpha=0.7)

    ax1.set_xlabel('Outcome', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('True vs Approximate Distributions', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Cross-entropy and KL divergence
    names = [r[0] for r in results]
    ces = [r[2] for r in results]
    kls = [r[3] for r in results]

    x_pos = np.arange(len(names))
    ax2.bar(x_pos - 0.2, ces, 0.4, label='Cross-Entropy H(P,Q)', alpha=0.8, color='orange')
    ax2.bar(x_pos + 0.2, kls, 0.4, label='KL Divergence D_KL(P||Q)', alpha=0.8, color='red')
    ax2.axhline(H_p, color='blue', linestyle='--', linewidth=2, label='H(P)')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylabel('Value (nats)', fontsize=12)
    ax2.set_title('Cross-Entropy and KL Divergence', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('cross_entropy_kl.png', dpi=150, bbox_inches='tight')
    print("\nSaved cross-entropy/KL divergence plot to 'cross_entropy_kl.png'")


def mutual_information_demo():
    """
    Demonstrate mutual information.

    Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y)
                                = H(X) - H(X|Y)
                                = D_KL(P(X,Y) || P(X)P(Y))

    Measures how much knowing Y reduces uncertainty about X.
    """
    print("\n" + "="*60)
    print("3. Mutual Information")
    print("="*60)

    def entropy_joint(pxy: np.ndarray) -> float:
        """Compute joint entropy H(X,Y)."""
        return -np.sum(pxy * np.log(pxy + 1e-10))

    def mutual_information(pxy: np.ndarray) -> float:
        """Compute mutual information I(X;Y)."""
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        px_py = px[:, np.newaxis] * py[np.newaxis, :]

        # I(X;Y) = D_KL(P(X,Y) || P(X)P(Y))
        return np.sum(pxy * np.log((pxy + 1e-10) / (px_py + 1e-10)))

    # Example 1: Independent variables (I = 0)
    print("\n--- Independent Variables ---")
    px = np.array([0.5, 0.5])
    py = np.array([0.3, 0.7])
    pxy_indep = px[:, np.newaxis] * py[np.newaxis, :]

    mi_indep = mutual_information(pxy_indep)
    print("P(X,Y) = P(X)P(Y)  (independent)")
    print(f"Mutual Information I(X;Y) = {mi_indep:.6f} ≈ 0")

    # Example 2: Perfectly correlated (I = H(X) = H(Y))
    print("\n--- Perfectly Correlated Variables ---")
    pxy_perfect = np.array([[0.3, 0.0], [0.0, 0.7]])

    mi_perfect = mutual_information(pxy_perfect)
    h_x = entropy_joint(pxy_perfect.sum(axis=1, keepdims=True).T)
    print("X = Y  (perfect correlation)")
    print(f"Mutual Information I(X;Y) = {mi_perfect:.4f}")
    print(f"H(X) = {h_x:.4f}  (I(X;Y) = H(X) when perfectly correlated)")

    # Example 3: Partial dependence
    print("\n--- Partially Dependent Variables ---")
    pxy_partial = np.array([[0.25, 0.05], [0.15, 0.55]])

    mi_partial = mutual_information(pxy_partial)
    print(f"Mutual Information I(X;Y) = {mi_partial:.4f}")
    print("(Between 0 and H(X), indicating partial dependence)")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    scenarios = [
        ('Independent\nI(X;Y) ≈ 0', pxy_indep, mi_indep),
        ('Partially Dependent\nI(X;Y) = {:.3f}'.format(mi_partial), pxy_partial, mi_partial),
        ('Perfect Correlation\nI(X;Y) = {:.3f}'.format(mi_perfect), pxy_perfect, mi_perfect)
    ]

    for ax, (title, pxy, mi) in zip(axes, scenarios):
        im = ax.imshow(pxy, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.7)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel('Y', fontsize=11)
        ax.set_ylabel('X', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Annotate cells with probabilities
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{pxy[i,j]:.2f}', ha='center', va='center',
                       color='white' if pxy[i,j] > 0.35 else 'black', fontsize=13)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('mutual_information.png', dpi=150, bbox_inches='tight')
    print("\nSaved mutual information plot to 'mutual_information.png'")


def ml_loss_functions():
    """
    Connect information theory to ML loss functions.

    Binary Classification:
    - Cross-entropy loss = -[y log(ŷ) + (1-y) log(1-ŷ)]
    - Minimizing cross-entropy = Maximizing likelihood

    Multi-class Classification:
    - Cross-entropy loss = -Σ y_c log(ŷ_c)
    - Same as negative log-likelihood
    """
    print("\n" + "="*60)
    print("4. Connection to ML Loss Functions")
    print("="*60)

    # Binary classification example
    print("\n--- Binary Classification ---")

    y_true = np.array([1, 0, 1, 1, 0])  # True labels
    y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])  # Predicted probabilities

    # Binary cross-entropy loss
    bce_loss = -np.mean(y_true * np.log(y_pred + 1e-10) +
                        (1 - y_true) * np.log(1 - y_pred + 1e-10))

    print(f"True labels: {y_true}")
    print(f"Predictions: {y_pred}")
    print(f"Binary Cross-Entropy Loss: {bce_loss:.4f}")

    # Multi-class classification example
    print("\n--- Multi-class Classification ---")

    # 3 samples, 4 classes
    y_true_mc = np.array([
        [1, 0, 0, 0],  # Class 0
        [0, 1, 0, 0],  # Class 1
        [0, 0, 1, 0]   # Class 2
    ])

    y_pred_mc = np.array([
        [0.7, 0.2, 0.05, 0.05],
        [0.1, 0.6, 0.2, 0.1],
        [0.1, 0.1, 0.7, 0.1]
    ])

    # Categorical cross-entropy loss
    cce_loss = -np.mean(np.sum(y_true_mc * np.log(y_pred_mc + 1e-10), axis=1))

    print(f"Categorical Cross-Entropy Loss: {cce_loss:.4f}")
    print("\nInterpretation:")
    print("  Lower loss = predictions closer to true distribution")
    print("  Minimizing cross-entropy = Maximizing likelihood")
    print("  Cross-entropy > entropy of true distribution")

    # Visualization: Effect of prediction confidence on loss
    probs = np.linspace(0.01, 0.99, 100)
    loss_correct = -np.log(probs)  # When prediction matches true label
    loss_incorrect = -np.log(1 - probs)  # When prediction doesn't match

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(probs, loss_correct, linewidth=3, label='Correct prediction (y=1, ŷ=p)',
            color='green')
    ax.plot(probs, loss_incorrect, linewidth=3, label='Incorrect prediction (y=0, ŷ=p)',
            color='red')

    ax.set_xlabel('Predicted Probability (ŷ)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Binary Cross-Entropy Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig('ml_loss_functions.png', dpi=150, bbox_inches='tight')
    print("\nSaved ML loss functions plot to 'ml_loss_functions.png'")


def elbo_visualization():
    """
    Demonstrate ELBO (Evidence Lower Bound) for variational inference.

    ELBO: log p(x) >= E_q[log p(x,z)] - E_q[log q(z)]
                    = E_q[log p(x|z)] - D_KL(q(z)||p(z))

    Maximizing ELBO = Minimizing KL(q||p) while maximizing likelihood.
    """
    print("\n" + "="*60)
    print("5. ELBO (Evidence Lower Bound)")
    print("="*60)

    print("\nVariational Inference Setup:")
    print("  True posterior: p(z|x) - intractable")
    print("  Approximate: q(z) ≈ p(z|x)")
    print("\nELBO Decomposition:")
    print("  log p(x) = ELBO + D_KL(q(z)||p(z|x))")
    print("  ELBO = E_q[log p(x|z)] - D_KL(q(z)||p(z))")
    print("\nMaximizing ELBO:")
    print("  1. Minimizes KL divergence to true posterior")
    print("  2. Maximizes expected log-likelihood")

    # Simulate ELBO during training
    n_iterations = 200

    # ELBO components (simulated)
    np.random.seed(42)
    reconstruction_loss = 100 * np.exp(-np.linspace(0, 3, n_iterations))
    reconstruction_loss += np.random.randn(n_iterations) * 2

    kl_divergence = 50 * (1 - np.exp(-np.linspace(0, 2, n_iterations)))
    kl_divergence += np.random.randn(n_iterations) * 1.5

    elbo = -(reconstruction_loss + kl_divergence)  # Negative because we're minimizing loss

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: ELBO components
    iterations = np.arange(n_iterations)

    ax1.plot(iterations, reconstruction_loss, linewidth=2, label='Reconstruction Loss (negative log p(x|z))',
             color='red')
    ax1.plot(iterations, kl_divergence, linewidth=2, label='KL Divergence D_KL(q||p)',
             color='blue')
    ax1.plot(iterations, reconstruction_loss + kl_divergence, linewidth=2.5,
             label='Total Loss (negative ELBO)', color='black', linestyle='--')

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('ELBO Components During Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: ELBO (to be maximized)
    ax2.plot(iterations, elbo, linewidth=3, color='green')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('ELBO (to maximize)', fontsize=12)
    ax2.set_title('Evidence Lower Bound (ELBO)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Annotate
    ax2.text(150, elbo[150], 'ELBO increases →\nbetter approximation',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('elbo_visualization.png', dpi=150, bbox_inches='tight')
    print("\nSaved ELBO visualization to 'elbo_visualization.png'")

    print(f"\nFinal values:")
    print(f"  Reconstruction Loss: {reconstruction_loss[-1]:.2f}")
    print(f"  KL Divergence: {kl_divergence[-1]:.2f}")
    print(f"  ELBO: {elbo[-1]:.2f}")


if __name__ == "__main__":
    print("="*60)
    print("Information Theory for Machine Learning")
    print("="*60)

    # Run demonstrations
    entropy_demo()
    cross_entropy_kl_divergence()
    mutual_information_demo()
    ml_loss_functions()
    elbo_visualization()

    print("\n" + "="*60)
    print("Key Takeaways:")
    print("="*60)
    print("1. Entropy: Measures uncertainty/information content")
    print("2. Cross-Entropy: Used as loss function in classification")
    print("3. KL Divergence: Measures difference between distributions")
    print("4. Mutual Information: Measures dependence between variables")
    print("5. ELBO: Fundamental to variational inference (VAE, etc.)")
    print("6. Minimizing cross-entropy = Maximizing likelihood")
