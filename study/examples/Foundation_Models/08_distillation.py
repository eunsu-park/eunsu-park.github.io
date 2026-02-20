"""
Foundation Models - Knowledge Distillation

Implements knowledge distillation from scratch using PyTorch.
Demonstrates student-teacher training with soft labels and temperature.
Shows how to compress large models into smaller ones.

Requires: PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TeacherModel(nn.Module):
    """Large teacher model (to be distilled)."""

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class StudentModel(nn.Module):
    """Small student model (distilled from teacher)."""

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7):
    """
    Compute distillation loss.

    L = α * L_soft + (1-α) * L_hard

    where:
    - L_soft: KL divergence between softened student and teacher outputs
    - L_hard: Cross-entropy with true labels
    - T: Temperature for softening

    Args:
        student_logits: Student model outputs (before softmax)
        teacher_logits: Teacher model outputs (before softmax)
        labels: True labels
        temperature: Temperature for soft targets
        alpha: Weight for soft loss (0-1)

    Returns:
        Total distillation loss
    """
    # Soft targets (with temperature)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)

    # KL divergence loss (scaled by T^2)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

    # Hard targets (standard cross-entropy)
    hard_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    return alpha * soft_loss + (1 - alpha) * hard_loss


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_teacher(model, X_train, y_train, epochs=100, batch_size=32, lr=0.001):
    """Train teacher model on data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    num_samples = X_train.shape[0]

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(num_samples)[:batch_size]

        X_batch = X_train[indices]
        y_batch = y_train[indices]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_train).argmax(dim=1)
                acc = (preds == y_train).float().mean()
            print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")


def train_student_standard(model, X_train, y_train, epochs=100, batch_size=32, lr=0.001):
    """Train student model with standard supervised learning (no distillation)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    num_samples = X_train.shape[0]

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(num_samples)[:batch_size]

        X_batch = X_train[indices]
        y_batch = y_train[indices]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_train).argmax(dim=1)
                acc = (preds == y_train).float().mean()
            print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")


def train_student_distillation(student, teacher, X_train, y_train, epochs=100,
                                batch_size=32, lr=0.001, temperature=3.0, alpha=0.7):
    """Train student model with knowledge distillation."""
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    num_samples = X_train.shape[0]
    teacher.eval()

    for epoch in range(epochs):
        student.train()
        indices = torch.randperm(num_samples)[:batch_size]

        X_batch = X_train[indices]
        y_batch = y_train[indices]

        optimizer.zero_grad()

        # Get student outputs
        student_logits = student(X_batch)

        # Get teacher outputs (no gradients)
        with torch.no_grad():
            teacher_logits = teacher(X_batch)

        # Distillation loss
        loss = distillation_loss(student_logits, teacher_logits, y_batch,
                                  temperature=temperature, alpha=alpha)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            student.eval()
            with torch.no_grad():
                preds = student(X_train).argmax(dim=1)
                acc = (preds == y_train).float().mean()
            print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")


# ============================================================
# Demonstrations
# ============================================================

def demo_temperature_effect():
    """Demonstrate effect of temperature on softmax."""
    print("=" * 60)
    print("DEMO 1: Temperature Effect on Softmax")
    print("=" * 60)

    # Logits with clear winner
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.2]])

    temperatures = [1.0, 2.0, 5.0, 10.0]

    print(f"\nLogits: {logits[0].tolist()}\n")

    for T in temperatures:
        probs = F.softmax(logits / T, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        print(f"Temperature {T}:")
        print(f"  Probabilities: {probs[0].tolist()}")
        print(f"  Entropy: {entropy:.4f}\n")

    print("Higher temperature → softer (more uniform) distribution")
    print("This reveals more of the teacher's knowledge")


def demo_soft_vs_hard_labels():
    """Compare soft and hard labels."""
    print("\n" + "=" * 60)
    print("DEMO 2: Soft vs Hard Labels")
    print("=" * 60)

    # Simulate teacher predictions
    teacher_logits = torch.tensor([
        [3.0, 1.5, 0.8, 0.5],  # High confidence
        [2.0, 1.8, 1.5, 1.0],  # Lower confidence
    ])

    hard_labels = torch.tensor([0, 0])

    print("\nTeacher logits:")
    print(teacher_logits)

    print("\nHard labels (one-hot):")
    for label in hard_labels:
        one_hot = torch.zeros(4)
        one_hot[label] = 1
        print(f"  {one_hot.tolist()}")

    print("\nSoft labels (T=3):")
    soft_labels = F.softmax(teacher_logits / 3.0, dim=1)
    for i, soft in enumerate(soft_labels):
        print(f"  {soft.tolist()}")

    print("\nSoft labels encode similarity between classes!")


def demo_basic_distillation():
    """Demonstrate basic knowledge distillation."""
    print("\n" + "=" * 60)
    print("DEMO 3: Basic Knowledge Distillation")
    print("=" * 60)

    # Generate synthetic data
    torch.manual_seed(42)
    np.random.seed(42)

    input_dim = 50
    num_classes = 5
    num_samples = 500

    X_train = torch.randn(num_samples, input_dim)
    y_train = torch.randint(0, num_classes, (num_samples,))

    # Create models
    teacher = TeacherModel(input_dim, hidden_dim=256, num_classes=num_classes)
    student = StudentModel(input_dim, hidden_dim=64, num_classes=num_classes)

    print(f"\nTeacher parameters: {count_parameters(teacher):,}")
    print(f"Student parameters: {count_parameters(student):,}")
    print(f"Compression ratio: {count_parameters(teacher) / count_parameters(student):.2f}x\n")

    # Train teacher
    print("Training teacher model...")
    print("-" * 60)
    train_teacher(teacher, X_train, y_train, epochs=100, batch_size=64)

    # Evaluate teacher
    teacher.eval()
    with torch.no_grad():
        teacher_preds = teacher(X_train).argmax(dim=1)
        teacher_acc = (teacher_preds == y_train).float().mean()
    print(f"\nTeacher final accuracy: {teacher_acc:.4f}")


def demo_student_comparison():
    """Compare student trained with and without distillation."""
    print("\n" + "=" * 60)
    print("DEMO 4: Student Training Comparison")
    print("=" * 60)

    # Generate data
    torch.manual_seed(42)
    input_dim = 50
    num_classes = 5
    num_samples = 500

    X_train = torch.randn(num_samples, input_dim)
    y_train = torch.randint(0, num_classes, (num_samples,))

    # Train teacher
    teacher = TeacherModel(input_dim, hidden_dim=256, num_classes=num_classes)
    print("Training teacher...")
    train_teacher(teacher, X_train, y_train, epochs=100, batch_size=64, lr=0.001)

    teacher.eval()
    with torch.no_grad():
        teacher_acc = (teacher(X_train).argmax(dim=1) == y_train).float().mean()
    print(f"Teacher accuracy: {teacher_acc:.4f}\n")

    # Student 1: Standard training
    print("-" * 60)
    print("Student 1: Standard training (no distillation)")
    print("-" * 60)
    student1 = StudentModel(input_dim, hidden_dim=64, num_classes=num_classes)
    train_student_standard(student1, X_train, y_train, epochs=100, batch_size=64, lr=0.001)

    student1.eval()
    with torch.no_grad():
        student1_acc = (student1(X_train).argmax(dim=1) == y_train).float().mean()

    # Student 2: Distillation
    print("\n" + "-" * 60)
    print("Student 2: Knowledge distillation (T=3, α=0.7)")
    print("-" * 60)
    student2 = StudentModel(input_dim, hidden_dim=64, num_classes=num_classes)
    train_student_distillation(student2, teacher, X_train, y_train,
                                epochs=100, batch_size=64, lr=0.001,
                                temperature=3.0, alpha=0.7)

    student2.eval()
    with torch.no_grad():
        student2_acc = (student2(X_train).argmax(dim=1) == y_train).float().mean()

    # Compare
    print("\n" + "=" * 60)
    print("Comparison:")
    print("=" * 60)
    print(f"Teacher accuracy:              {teacher_acc:.4f}")
    print(f"Student (standard):            {student1_acc:.4f}")
    print(f"Student (distillation):        {student2_acc:.4f}")
    print(f"Improvement from distillation: {(student2_acc - student1_acc):.4f}")


def demo_hyperparameter_tuning():
    """Study effect of temperature and alpha."""
    print("\n" + "=" * 60)
    print("DEMO 5: Hyperparameter Tuning")
    print("=" * 60)

    # Generate data
    torch.manual_seed(42)
    input_dim = 40
    num_classes = 4
    num_samples = 400

    X_train = torch.randn(num_samples, input_dim)
    y_train = torch.randint(0, num_classes, (num_samples,))

    # Train teacher
    teacher = TeacherModel(input_dim, hidden_dim=200, num_classes=num_classes)
    train_teacher(teacher, X_train, y_train, epochs=80, batch_size=64, lr=0.001)

    print("\n" + "-" * 60)
    print("Testing different temperatures (α=0.7):")
    print("-" * 60)

    for T in [1.0, 2.0, 4.0, 8.0]:
        student = StudentModel(input_dim, hidden_dim=50, num_classes=num_classes)
        train_student_distillation(student, teacher, X_train, y_train,
                                    epochs=80, batch_size=64, lr=0.001,
                                    temperature=T, alpha=0.7)

        student.eval()
        with torch.no_grad():
            acc = (student(X_train).argmax(dim=1) == y_train).float().mean()
        print(f"T={T}: Final accuracy = {acc:.4f}")

    print("\n" + "-" * 60)
    print("Testing different alpha values (T=3.0):")
    print("-" * 60)

    for alpha in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
        student = StudentModel(input_dim, hidden_dim=50, num_classes=num_classes)
        train_student_distillation(student, teacher, X_train, y_train,
                                    epochs=80, batch_size=64, lr=0.001,
                                    temperature=3.0, alpha=alpha)

        student.eval()
        with torch.no_grad():
            acc = (student(X_train).argmax(dim=1) == y_train).float().mean()
        print(f"α={alpha}: Final accuracy = {acc:.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Foundation Models: Knowledge Distillation")
    print("=" * 60)

    demo_temperature_effect()
    demo_soft_vs_hard_labels()
    demo_basic_distillation()
    demo_student_comparison()
    demo_hyperparameter_tuning()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. Distillation: Compress large model → small model")
    print("2. Soft labels: Encode class similarities, not just winner")
    print("3. Temperature: Controls softness of distribution")
    print("4. Loss: α × L_soft + (1-α) × L_hard")
    print("5. Typical: T=3-5, α=0.5-0.9")
    print("6. Student learns from teacher's mistakes and uncertainties")
    print("7. Can achieve similar performance with much smaller model")
    print("=" * 60)
