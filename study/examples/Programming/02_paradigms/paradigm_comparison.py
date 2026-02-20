"""
Programming Paradigm Comparison

Problem: Given a list of numbers, filter out negatives, square the positives,
and return the sum of squares.

This demonstrates the same problem solved using:
1. Imperative Programming (step-by-step instructions)
2. Object-Oriented Programming (data + behavior encapsulation)
3. Functional Programming (pure functions + composition)
"""

from typing import List
from functools import reduce


# =============================================================================
# 1. IMPERATIVE STYLE: Focus on HOW to do it (step-by-step)
# =============================================================================

def imperative_approach(numbers: List[int]) -> int:
    """
    Imperative style uses explicit loops and mutable state.
    We tell the computer exactly WHAT to do at each step.
    """
    # Step 1: Create mutable accumulator
    result = 0

    # Step 2: Iterate through each number
    for num in numbers:
        # Step 3: Check condition
        if num > 0:
            # Step 4: Transform (square)
            squared = num * num
            # Step 5: Accumulate
            result += squared

    return result


# =============================================================================
# 2. OBJECT-ORIENTED STYLE: Focus on objects with data + behavior
# =============================================================================

class NumberProcessor:
    """
    OOP encapsulates data and operations together.
    State is managed within the object.
    """

    def __init__(self, numbers: List[int]):
        self._numbers = numbers
        self._filtered = []
        self._squared = []
        self._sum = 0

    def filter_positives(self) -> 'NumberProcessor':
        """Remove negative numbers (method chaining support)"""
        self._filtered = [n for n in self._numbers if n > 0]
        return self

    def square_values(self) -> 'NumberProcessor':
        """Square each value"""
        self._squared = [n * n for n in self._filtered]
        return self

    def calculate_sum(self) -> 'NumberProcessor':
        """Calculate sum of all values"""
        self._sum = sum(self._squared)
        return self

    def get_result(self) -> int:
        """Return final result"""
        return self._sum


def oop_approach(numbers: List[int]) -> int:
    """
    OOP style uses objects to encapsulate state and behavior.
    Supports method chaining for fluent interface.
    """
    processor = NumberProcessor(numbers)
    return (processor
            .filter_positives()
            .square_values()
            .calculate_sum()
            .get_result())


# =============================================================================
# 3. FUNCTIONAL STYLE: Focus on WHAT to compute (composition of functions)
# =============================================================================

def is_positive(n: int) -> bool:
    """Pure function: same input always returns same output"""
    return n > 0


def square(n: int) -> int:
    """Pure function: no side effects"""
    return n * n


def add(a: int, b: int) -> int:
    """Pure function for reduction"""
    return a + b


def functional_approach(numbers: List[int]) -> int:
    """
    Functional style uses pure functions and function composition.
    No mutable state, no side effects.
    Data flows through transformations.
    """
    return reduce(
        add,
        map(square, filter(is_positive, numbers)),
        0  # initial value
    )


def functional_approach_comprehension(numbers: List[int]) -> int:
    """
    Alternative functional style using list comprehension
    (more Pythonic, still functionally pure)
    """
    return sum(n * n for n in numbers if n > 0)


# =============================================================================
# COMPARISON & DEMONSTRATION
# =============================================================================

def compare_paradigms():
    """Compare all three paradigms with the same input"""

    test_data = [-5, 3, -2, 8, 0, 4, -1, 6]

    print("=" * 70)
    print("PROGRAMMING PARADIGM COMPARISON")
    print("=" * 70)
    print(f"\nInput: {test_data}")
    print(f"Task: Filter positives → Square → Sum\n")

    # Test all approaches
    imperative_result = imperative_approach(test_data)
    oop_result = oop_approach(test_data)
    functional_result = functional_approach(test_data)
    functional_comp_result = functional_approach_comprehension(test_data)

    print("RESULTS:")
    print(f"  Imperative:        {imperative_result}")
    print(f"  Object-Oriented:   {oop_result}")
    print(f"  Functional:        {functional_result}")
    print(f"  Functional (comp): {functional_comp_result}")

    # Verify all give same result
    assert imperative_result == oop_result == functional_result == functional_comp_result
    print("\n✓ All paradigms produce identical results!\n")

    # Show step-by-step breakdown
    print("STEP-BY-STEP BREAKDOWN:")
    positives = [n for n in test_data if n > 0]
    print(f"  1. Filter positives: {positives}")
    squares = [n * n for n in positives]
    print(f"  2. Square values:    {squares}")
    total = sum(squares)
    print(f"  3. Sum:              {total}")

    print("\n" + "=" * 70)
    print("PARADIGM CHARACTERISTICS")
    print("=" * 70)

    print("\nIMPERATIVE:")
    print("  ✓ Explicit control flow (loops, conditionals)")
    print("  ✓ Mutable state (variables change)")
    print("  ✓ Step-by-step instructions")
    print("  ✓ Easy to debug (can inspect each step)")
    print("  ✗ More verbose")
    print("  ✗ Harder to parallelize")

    print("\nOBJECT-ORIENTED:")
    print("  ✓ Encapsulation (data + behavior together)")
    print("  ✓ Reusable objects")
    print("  ✓ Method chaining (fluent interface)")
    print("  ✓ Good for complex state management")
    print("  ✗ Can be over-engineered for simple tasks")
    print("  ✗ Mutable state can lead to bugs")

    print("\nFUNCTIONAL:")
    print("  ✓ Pure functions (no side effects)")
    print("  ✓ Immutable data")
    print("  ✓ Function composition")
    print("  ✓ Easy to test and parallelize")
    print("  ✓ Declarative (what, not how)")
    print("  ✗ Can be less intuitive initially")
    print("  ✗ May use more memory (creates new data)")

    print("\n" + "=" * 70)
    print("WHEN TO USE EACH PARADIGM")
    print("=" * 70)

    print("\nIMPERATIVE:")
    print("  • Performance-critical code")
    print("  • Systems programming")
    print("  • When you need fine-grained control")

    print("\nOBJECT-ORIENTED:")
    print("  • Complex state management")
    print("  • Modeling real-world entities")
    print("  • Large codebases with multiple developers")
    print("  • GUI applications")

    print("\nFUNCTIONAL:")
    print("  • Data transformation pipelines")
    print("  • Concurrent/parallel processing")
    print("  • Mathematical computations")
    print("  • When immutability is important")


def demonstrate_hybrid_approach():
    """
    Modern programming often uses a HYBRID approach,
    combining the best of all paradigms.
    """
    print("\n" + "=" * 70)
    print("HYBRID APPROACH (Real-world Python)")
    print("=" * 70)

    class DataPipeline:
        """OOP structure with functional transformations"""

        def __init__(self, data: List[int]):
            self.data = data

        def process(self) -> int:
            """Functional operations within OOP structure"""
            return sum(n * n for n in self.data if n > 0)

    test_data = [-5, 3, -2, 8, 0, 4]
    pipeline = DataPipeline(test_data)
    result = pipeline.process()

    print(f"\nInput: {test_data}")
    print(f"Result: {result}")
    print("\nThis combines:")
    print("  • OOP: Class structure for organization")
    print("  • Functional: Pure transformation with comprehension")
    print("  • Imperative: Hidden in Python's built-in functions")
    print("\nMost modern code uses this pragmatic mix!")


if __name__ == "__main__":
    compare_paradigms()
    demonstrate_hybrid_approach()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAY")
    print("=" * 70)
    print("""
Paradigms are TOOLS, not religions.
Choose based on the problem:
  • Imperative for control and performance
  • OOP for modeling and state management
  • Functional for transformations and safety
  • Hybrid for real-world pragmatism
""")
