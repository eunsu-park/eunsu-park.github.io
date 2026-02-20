"""
Python Generators and Iterators

Demonstrates:
- Generator functions with yield
- Generator expressions
- send() and bidirectional generators
- yield from delegation
- itertools module
- Infinite generators
- Generator pipelines
"""

import itertools
from typing import Iterator, Any


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Basic Generator Functions
# =============================================================================

section("Basic Generator Functions")


def simple_generator():
    """Simple generator yielding values."""
    print("    Yielding 1")
    yield 1
    print("    Yielding 2")
    yield 2
    print("    Yielding 3")
    yield 3
    print("    Generator exhausted")


gen = simple_generator()
print(f"Generator object: {gen}")
print(f"Type: {type(gen)}")

for value in gen:
    print(f"  Received: {value}")


def countdown(n: int) -> Iterator[int]:
    """Countdown from n to 1."""
    while n > 0:
        yield n
        n -= 1


print("\nCountdown from 5:")
for i in countdown(5):
    print(f"  {i}")


# =============================================================================
# Generator Expressions
# =============================================================================

section("Generator Expressions")

# List comprehension - creates entire list in memory
squares_list = [x**2 for x in range(10)]
print(f"List comprehension: {squares_list}")

# Generator expression - lazy evaluation
squares_gen = (x**2 for x in range(10))
print(f"Generator expression: {squares_gen}")
print(f"First 5 squares: {list(itertools.islice(squares_gen, 5))}")


# Memory efficiency example
def memory_comparison():
    """Compare memory usage of list vs generator."""
    import sys

    n = 1000
    list_comp = [x**2 for x in range(n)]
    gen_exp = (x**2 for x in range(n))

    list_size = sys.getsizeof(list_comp)
    gen_size = sys.getsizeof(gen_exp)

    print(f"  List ({n} items): {list_size} bytes")
    print(f"  Generator: {gen_size} bytes")
    print(f"  Memory saved: {list_size - gen_size} bytes")


memory_comparison()


# =============================================================================
# Fibonacci Generator
# =============================================================================

section("Fibonacci Generator")


def fibonacci() -> Iterator[int]:
    """Infinite Fibonacci sequence."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


print("First 10 Fibonacci numbers:")
for i, fib in enumerate(fibonacci()):
    if i >= 10:
        break
    print(f"  F({i}) = {fib}")


# =============================================================================
# Generator with send()
# =============================================================================

section("Bidirectional Generators with send()")


def accumulator() -> Iterator[int]:
    """Generator that accumulates sent values."""
    total = 0
    while True:
        value = yield total
        if value is not None:
            total += value
            print(f"    Received {value}, total now {total}")


acc = accumulator()
next(acc)  # Prime the generator
print(f"  send(10) -> {acc.send(10)}")
print(f"  send(20) -> {acc.send(20)}")
print(f"  send(5) -> {acc.send(5)}")


def running_average() -> Iterator[float]:
    """Calculate running average of sent values."""
    total = 0.0
    count = 0
    while True:
        value = yield total / count if count > 0 else 0.0
        if value is not None:
            total += value
            count += 1


avg = running_average()
next(avg)  # Prime
print(f"\n  Running average:")
for val in [10, 20, 30, 40]:
    result = avg.send(val)
    print(f"    After {val}: {result:.2f}")


# =============================================================================
# yield from - Generator Delegation
# =============================================================================

section("yield from - Generator Delegation")


def generator_a():
    """First generator."""
    yield 1
    yield 2


def generator_b():
    """Second generator."""
    yield 3
    yield 4


def combined_manual():
    """Manual delegation."""
    for value in generator_a():
        yield value
    for value in generator_b():
        yield value


def combined_yield_from():
    """Delegation with yield from."""
    yield from generator_a()
    yield from generator_b()


print("Manual delegation:")
print(f"  {list(combined_manual())}")

print("yield from delegation:")
print(f"  {list(combined_yield_from())}")


# =============================================================================
# itertools - Combinatoric Generators
# =============================================================================

section("itertools - Combinatoric Generators")

# chain - concatenate iterables
print("itertools.chain([1,2], [3,4], [5,6]):")
print(f"  {list(itertools.chain([1, 2], [3, 4], [5, 6]))}")

# islice - slice an iterator
print("\nitertools.islice(range(10), 2, 8, 2):")
print(f"  {list(itertools.islice(range(10), 2, 8, 2))}")

# groupby - group consecutive items
data = [1, 1, 2, 2, 2, 3, 1, 1]
print(f"\nitertools.groupby({data}):")
for key, group in itertools.groupby(data):
    print(f"  {key}: {list(group)}")

# product - cartesian product
print("\nitertools.product('AB', '12'):")
for item in itertools.product('AB', '12'):
    print(f"  {item}")

# combinations
print("\nitertools.combinations('ABCD', 2):")
for combo in itertools.combinations('ABCD', 2):
    print(f"  {combo}")


# =============================================================================
# Infinite Generators
# =============================================================================

section("Infinite Generators")


def cycle_colors() -> Iterator[str]:
    """Infinite cycle of colors."""
    colors = ['red', 'green', 'blue']
    while True:
        yield from colors


print("First 7 colors from infinite cycle:")
for i, color in enumerate(cycle_colors()):
    if i >= 7:
        break
    print(f"  {i}: {color}")


# Using itertools.cycle
print("\nitertools.cycle (first 5):")
for i, color in enumerate(itertools.cycle(['red', 'green', 'blue'])):
    if i >= 5:
        break
    print(f"  {color}")

# itertools.count
print("\nitertools.count(start=10, step=5) (first 6):")
for i, num in enumerate(itertools.count(start=10, step=5)):
    if i >= 6:
        break
    print(f"  {num}")


# =============================================================================
# Generator Pipelines
# =============================================================================

section("Generator Pipelines")


def read_numbers():
    """Simulate reading numbers from source."""
    yield from range(1, 21)


def filter_even(numbers: Iterator[int]) -> Iterator[int]:
    """Filter even numbers."""
    for n in numbers:
        if n % 2 == 0:
            yield n


def square(numbers: Iterator[int]) -> Iterator[int]:
    """Square each number."""
    for n in numbers:
        yield n ** 2


def take(n: int, iterable: Iterator[Any]) -> Iterator[Any]:
    """Take first n items."""
    for i, item in enumerate(iterable):
        if i >= n:
            break
        yield item


# Build pipeline
pipeline = take(5, square(filter_even(read_numbers())))
result = list(pipeline)
print(f"Pipeline: read -> filter_even -> square -> take(5)")
print(f"Result: {result}")


# =============================================================================
# File Processing with Generators
# =============================================================================

section("File Processing with Generators")


def generate_log_lines():
    """Simulate log file lines."""
    logs = [
        "INFO: Application started",
        "DEBUG: Loading configuration",
        "ERROR: Connection failed",
        "INFO: Retrying connection",
        "ERROR: Timeout exceeded",
        "INFO: Application stopped"
    ]
    yield from logs


def filter_errors(lines: Iterator[str]) -> Iterator[str]:
    """Filter ERROR lines."""
    for line in lines:
        if 'ERROR' in line:
            yield line


def extract_message(lines: Iterator[str]) -> Iterator[str]:
    """Extract message after colon."""
    for line in lines:
        if ':' in line:
            yield line.split(':', 1)[1].strip()


print("Error messages from logs:")
error_pipeline = extract_message(filter_errors(generate_log_lines()))
for msg in error_pipeline:
    print(f"  - {msg}")


# =============================================================================
# Generator State Preservation
# =============================================================================

section("Generator State Preservation")


def stateful_generator():
    """Generator maintains state between yields."""
    state = {"count": 0}

    while True:
        state["count"] += 1
        received = yield f"Call #{state['count']}"
        if received:
            print(f"    Received: {received}")


gen = stateful_generator()
print(next(gen))
print(gen.send("hello"))
print(next(gen))
print(gen.send("world"))


# =============================================================================
# Generator Best Practices
# =============================================================================

section("Generator Best Practices")


def process_large_dataset(limit: int = 10) -> Iterator[dict]:
    """
    Process large dataset lazily.

    Best practices:
    - Yield one item at a time
    - Don't accumulate all results
    - Let caller control iteration
    """
    for i in range(limit):
        # Simulate expensive processing
        record = {
            'id': i,
            'value': i ** 2,
            'processed': True
        }
        yield record


print("Processing records lazily:")
for record in process_large_dataset(5):
    print(f"  {record}")


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Generator patterns covered:
1. Basic generators - yield keyword
2. Generator expressions - memory-efficient iteration
3. send() - bidirectional communication
4. yield from - delegation to sub-generators
5. itertools - powerful combinatoric generators
   - chain, islice, groupby, product, combinations
6. Infinite generators - cycle, count
7. Generator pipelines - composable data processing
8. State preservation - generators maintain local state

Generators provide:
- Memory efficiency (lazy evaluation)
- Infinite sequences
- Pipeline processing
- Stateful iteration
- Clean, readable code
""")
