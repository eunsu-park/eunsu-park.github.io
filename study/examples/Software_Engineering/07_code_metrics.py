"""
Code Quality Metrics Calculator

Parses Python source code using the `ast` module and computes:

1. Lines of Code (LOC)
   - Total lines
   - Blank lines
   - Comment lines (# ...)
   - Logical lines (non-blank, non-comment)

2. Cyclomatic Complexity (McCabe)
   Complexity = 1 + number of decision points
   Decision points counted: if, elif, for, while, and, or, except, with,
   ternary expressions (IfExp), assert, comprehension conditions.
   Interpretation:
     1–10   Low risk, simple
     11–20  Moderate complexity, some risk
     21–50  High complexity, hard to test
     >50    Very high risk, should refactor

3. Halstead Metrics
   n1 = distinct operators, n2 = distinct operands
   N1 = total operators,  N2 = total operands
   Vocabulary (n) = n1 + n2
   Length   (N) = N1 + N2
   Volume   (V) = N * log2(n)
   Difficulty (D) = (n1/2) * (N2/n2)
   Effort   (E) = D * V
   Time to implement (T, seconds) = E / 18

Operators counted: augmented assignments, binary ops, unary ops,
boolean ops, comparison ops, subscript, attribute access, calls.
Operands counted: names, constants, string literals.

Run:
    python 07_code_metrics.py
"""

import ast
import math
import textwrap
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Lines of Code
# ---------------------------------------------------------------------------

@dataclass
class LOCMetrics:
    total: int = 0
    blank: int = 0
    comment: int = 0
    logical: int = 0

    @classmethod
    def from_source(cls, source: str) -> "LOCMetrics":
        m = cls()
        for line in source.splitlines():
            m.total += 1
            stripped = line.strip()
            if not stripped:
                m.blank += 1
            elif stripped.startswith("#"):
                m.comment += 1
            else:
                m.logical += 1
        return m


# ---------------------------------------------------------------------------
# Cyclomatic Complexity
# ---------------------------------------------------------------------------

class CyclomaticVisitor(ast.NodeVisitor):
    """Count decision points that increase cyclomatic complexity."""

    def __init__(self) -> None:
        self.complexity: int = 1  # base complexity

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        # Ternary: x if cond else y
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        # Each context manager item adds a potential branch
        self.complexity += len(node.items) - 1 if len(node.items) > 1 else 0
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # Each 'and'/'or' adds a branch (n-1 for n values)
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        # Each 'if' in a comprehension adds a branch
        self.complexity += len(node.ifs)
        self.generic_visit(node)


def cyclomatic_complexity(tree: ast.AST) -> int:
    visitor = CyclomaticVisitor()
    visitor.visit(tree)
    return visitor.complexity


def complexity_rating(cc: int) -> str:
    if cc <= 10:
        return "Low (simple, easy to test)"
    if cc <= 20:
        return "Moderate (some risk)"
    if cc <= 50:
        return "High (hard to test, consider refactoring)"
    return "Very High (must refactor)"


# ---------------------------------------------------------------------------
# Halstead Metrics
# ---------------------------------------------------------------------------

OPERATOR_TYPES = (
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
    ast.MatMult,
    ast.UAdd, ast.USub, ast.Not, ast.Invert,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
)


class HalsteadVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.operators: list[str] = []
        self.operands: list[str] = []

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.operators.append(type(node.op).__name__)
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        self.operators.append(type(node.op).__name__)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.operators.append(type(node.op).__name__)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            self.operators.append(type(op).__name__)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.operators.append(f"{type(node.op).__name__}Assign")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.operators.append("Assign")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.operators.append("Call()")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.operators.append(".")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.operators.append("[]")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.operands.append(node.id)

    def visit_Constant(self, node: ast.Constant) -> None:
        self.operands.append(repr(node.value))


@dataclass
class HalsteadMetrics:
    n1: int      # distinct operators
    n2: int      # distinct operands
    N1: int      # total operators
    N2: int      # total operands
    vocabulary: int
    length: int
    volume: float
    difficulty: float
    effort: float
    time_seconds: float


def halstead_metrics(tree: ast.AST) -> HalsteadMetrics:
    visitor = HalsteadVisitor()
    visitor.visit(tree)

    ops = visitor.operators
    opds = visitor.operands

    n1 = len(set(ops))
    n2 = len(set(opds))
    N1 = len(ops)
    N2 = len(opds)

    vocab = n1 + n2
    length = N1 + N2
    volume = length * math.log2(vocab) if vocab > 1 else 0.0
    difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0.0
    effort = difficulty * volume
    time_s = effort / 18.0

    return HalsteadMetrics(
        n1=n1, n2=n2, N1=N1, N2=N2,
        vocabulary=vocab, length=length,
        volume=round(volume, 1),
        difficulty=round(difficulty, 2),
        effort=round(effort, 1),
        time_seconds=round(time_s, 1),
    )


# ---------------------------------------------------------------------------
# All-in-one analysis
# ---------------------------------------------------------------------------

def analyze(source: str, label: str = "source") -> None:
    tree = ast.parse(source)

    loc = LOCMetrics.from_source(source)
    cc = cyclomatic_complexity(tree)
    hal = halstead_metrics(tree)

    print(f"\n{'=' * 58}")
    print(f"  Metrics Report: {label}")
    print("=" * 58)

    print("\n  [ Lines of Code ]")
    print(f"    Total lines    : {loc.total}")
    print(f"    Blank lines    : {loc.blank}")
    print(f"    Comment lines  : {loc.comment}")
    print(f"    Logical lines  : {loc.logical}")

    print("\n  [ Cyclomatic Complexity ]")
    print(f"    Complexity (CC): {cc}")
    print(f"    Risk rating    : {complexity_rating(cc)}")

    print("\n  [ Halstead Metrics ]")
    print(f"    Distinct operators (n1): {hal.n1}")
    print(f"    Distinct operands  (n2): {hal.n2}")
    print(f"    Total operators    (N1): {hal.N1}")
    print(f"    Total operands     (N2): {hal.N2}")
    print(f"    Vocabulary  (n1+n2)    : {hal.vocabulary}")
    print(f"    Length      (N1+N2)    : {hal.length}")
    print(f"    Volume      (V)        : {hal.volume} bits")
    print(f"    Difficulty  (D)        : {hal.difficulty}")
    print(f"    Effort      (E)        : {hal.effort}")
    print(f"    Impl. time  (E/18)     : {hal.time_seconds} seconds")
    print()


# ---------------------------------------------------------------------------
# Sample functions to analyze
# ---------------------------------------------------------------------------

SAMPLE_SIMPLE = textwrap.dedent("""\
    def add(a, b):
        return a + b
""")

SAMPLE_MODERATE = textwrap.dedent("""\
    def classify_score(score):
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
""")

SAMPLE_COMPLEX = textwrap.dedent("""\
    def merge_sorted(a, b):
        result = []
        i, j = 0, 0
        while i < len(a) and j < len(b):
            if a[i] <= b[j]:
                result.append(a[i])
                i += 1
            else:
                result.append(b[j])
                j += 1
        result.extend(a[i:])
        result.extend(b[j:])
        return result

    def find_duplicates(items):
        seen = set()
        duplicates = []
        for item in items:
            if item in seen:
                if item not in duplicates:
                    duplicates.append(item)
            else:
                seen.add(item)
        return duplicates

    def safe_divide(x, y, default=0):
        try:
            return x / y if y != 0 else default
        except TypeError as e:
            raise ValueError(f"Invalid operands: {e}") from e
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nCode Quality Metrics Calculator")
    print("Using Python ast module for static analysis\n")

    print("  Sample 1 — Simple function (add):")
    print("  " + "-" * 30)
    for line in SAMPLE_SIMPLE.splitlines():
        print(f"    {line}")

    print("\n  Sample 2 — Moderate complexity (score classifier):")
    print("  " + "-" * 30)
    for line in SAMPLE_MODERATE.splitlines():
        print(f"    {line}")

    print("\n  Sample 3 — Higher complexity (merge sort + utilities):")
    print("  " + "-" * 30)
    for line in SAMPLE_COMPLEX.splitlines():
        print(f"    {line}")

    analyze(SAMPLE_SIMPLE,   label="Simple function (add)")
    analyze(SAMPLE_MODERATE, label="Moderate (score classifier)")
    analyze(SAMPLE_COMPLEX,  label="Higher complexity (merge + utils)")

    print("=" * 58)
    print("  Tip: aim for CC <= 10 and Halstead Volume < 1000 per")
    print("  function to keep code readable and maintainable.")
    print("=" * 58 + "\n")
