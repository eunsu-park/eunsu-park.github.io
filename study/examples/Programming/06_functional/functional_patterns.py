"""
Functional Programming Patterns in Python

Key concepts:
1. Pure Functions - no side effects, same input = same output
2. Higher-Order Functions - functions that take/return functions
3. Closures - functions that capture variables from outer scope
4. Function Composition - combining simple functions to build complex ones
5. Currying - transforming multi-argument function to chain of single-argument
6. Map/Filter/Reduce - fundamental data transformation operations
"""

from typing import Callable, List, TypeVar, Any
from functools import reduce, partial
import operator

T = TypeVar('T')
U = TypeVar('U')


# =============================================================================
# 1. PURE FUNCTIONS
# No side effects, deterministic output
# =============================================================================

print("=" * 70)
print("1. PURE FUNCTIONS")
print("=" * 70)


# ❌ IMPURE: Has side effects, depends on external state
counter = 0


def impure_increment(x: int) -> int:
    """Impure: modifies global state"""
    global counter
    counter += 1  # Side effect!
    return x + counter  # Depends on external state!


# ✅ PURE: No side effects, deterministic
def pure_add(x: int, y: int) -> int:
    """Pure: same inputs always give same output"""
    return x + y


def pure_square(x: float) -> float:
    """Pure: mathematical function"""
    return x * x


def pure_capitalize_words(text: str) -> str:
    """Pure: transforms input without side effects"""
    return ' '.join(word.capitalize() for word in text.split())


# =============================================================================
# 2. HIGHER-ORDER FUNCTIONS
# Functions that take or return other functions
# =============================================================================

print("\n" + "=" * 70)
print("2. HIGHER-ORDER FUNCTIONS")
print("=" * 70)


# Function that takes another function as argument
def apply_twice(func: Callable[[T], T], value: T) -> T:
    """Apply function twice to a value"""
    return func(func(value))


def apply_n_times(n: int) -> Callable[[Callable[[T], T], T], T]:
    """Returns a function that applies func n times"""

    def applier(func: Callable[[T], T], value: T) -> T:
        result = value
        for _ in range(n):
            result = func(result)
        return result

    return applier


# Function that returns another function
def make_multiplier(factor: int) -> Callable[[int], int]:
    """Returns a function that multiplies by factor"""

    def multiplier(x: int) -> int:
        return x * factor

    return multiplier


def make_adder(n: int) -> Callable[[int], int]:
    """Returns a function that adds n"""

    def adder(x: int) -> int:
        return x + n

    return adder


# =============================================================================
# 3. CLOSURES
# Functions that capture variables from their enclosing scope
# =============================================================================

print("\n" + "=" * 70)
print("3. CLOSURES")
print("=" * 70)


def make_counter(start: int = 0) -> Callable[[], int]:
    """Returns a counter function with private state"""
    count = start  # Captured by closure

    def counter() -> int:
        nonlocal count  # Allows modification of captured variable
        count += 1
        return count

    return counter


def make_greeting(greeting: str) -> Callable[[str], str]:
    """Returns a greeting function that remembers the greeting"""

    def greet(name: str) -> str:
        return f"{greeting}, {name}!"

    return greet


def make_accumulator(initial: float = 0.0) -> Callable[[float], float]:
    """Returns a function that accumulates values"""
    total = initial

    def accumulate(value: float) -> float:
        nonlocal total
        total += value
        return total

    return accumulate


# =============================================================================
# 4. FUNCTION COMPOSITION
# Building complex functions from simple ones
# =============================================================================

print("\n" + "=" * 70)
print("4. FUNCTION COMPOSITION")
print("=" * 70)


def compose(*functions: Callable) -> Callable:
    """
    Compose functions right to left: compose(f, g, h)(x) = f(g(h(x)))
    """

    def inner(arg):
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result

    return inner


def pipe(*functions: Callable) -> Callable:
    """
    Compose functions left to right: pipe(f, g, h)(x) = h(g(f(x)))
    More intuitive for data pipelines
    """

    def inner(arg):
        result = arg
        for func in functions:
            result = func(result)
        return result

    return inner


# Example transformation functions
def remove_spaces(text: str) -> str:
    return text.replace(" ", "")


def to_uppercase(text: str) -> str:
    return text.upper()


def add_exclamation(text: str) -> str:
    return text + "!"


def double(x: int) -> int:
    return x * 2


def increment(x: int) -> int:
    return x + 1


# =============================================================================
# 5. CURRYING
# Transform multi-argument function to chain of single-argument functions
# =============================================================================

print("\n" + "=" * 70)
print("5. CURRYING")
print("=" * 70)


# Regular function
def add_three_numbers(a: int, b: int, c: int) -> int:
    return a + b + c


# Curried version
def curried_add(a: int) -> Callable[[int], Callable[[int], int]]:
    def add_b(b: int) -> Callable[[int], int]:
        def add_c(c: int) -> int:
            return a + b + c

        return add_c

    return add_b


# Generic curry function
def curry(func: Callable) -> Callable:
    """
    Automatically curry a function
    (simplified version for demonstration)
    """

    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more_args: curried(*(args + more_args))

    return curried


# Using functools.partial (Pythonic currying)
def multiply(a: int, b: int, c: int) -> int:
    return a * b * c


# Create specialized functions using partial
double_then_multiply = partial(multiply, 2)
triple_then_multiply = partial(multiply, 3)


# =============================================================================
# 6. MAP / FILTER / REDUCE
# Fundamental functional operations
# =============================================================================

print("\n" + "=" * 70)
print("6. MAP / FILTER / REDUCE")
print("=" * 70)


def demonstrate_map():
    """Map: Transform each element"""
    numbers = [1, 2, 3, 4, 5]

    # Using built-in map
    squared = list(map(lambda x: x ** 2, numbers))

    # List comprehension (more Pythonic)
    squared_comp = [x ** 2 for x in numbers]

    return squared, squared_comp


def demonstrate_filter():
    """Filter: Keep elements matching predicate"""
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Using built-in filter
    evens = list(filter(lambda x: x % 2 == 0, numbers))

    # List comprehension (more Pythonic)
    evens_comp = [x for x in numbers if x % 2 == 0]

    return evens, evens_comp


def demonstrate_reduce():
    """Reduce: Combine all elements into single value"""
    numbers = [1, 2, 3, 4, 5]

    # Sum using reduce
    total = reduce(lambda acc, x: acc + x, numbers, 0)

    # Product using reduce
    product = reduce(lambda acc, x: acc * x, numbers, 1)

    # More Pythonic: use operator module
    total_op = reduce(operator.add, numbers, 0)
    product_op = reduce(operator.mul, numbers, 1)

    return total, product, total_op, product_op


# =============================================================================
# PRACTICAL EXAMPLES
# =============================================================================

def practical_pipeline_example():
    """Real-world data transformation pipeline"""

    # Data pipeline: process user data
    users = [
        {"name": "alice jones", "age": 25, "salary": 50000},
        {"name": "bob smith", "age": 17, "salary": 0},
        {"name": "charlie brown", "age": 35, "salary": 75000},
        {"name": "diana prince", "age": 30, "salary": 90000},
        {"name": "eve taylor", "age": 16, "salary": 0},
    ]

    # Functional pipeline
    pipeline = pipe(
        # Filter adults
        lambda users: filter(lambda u: u["age"] >= 18, users),
        # Extract names and capitalize
        lambda users: map(
            lambda u: {**u, "name": u["name"].title()},
            users
        ),
        # Calculate bonus (10% of salary)
        lambda users: map(
            lambda u: {**u, "bonus": u["salary"] * 0.1},
            users
        ),
        # Convert to list
        list,
    )

    return pipeline(users)


def practical_function_factory():
    """Function factory for validation"""

    def make_validator(
        min_value: float,
        max_value: float
    ) -> Callable[[float], bool]:
        """Create range validator"""

        def validate(value: float) -> bool:
            return min_value <= value <= max_value

        return validate

    # Create specialized validators
    is_percentage = make_validator(0, 100)
    is_temperature = make_validator(-273.15, float('inf'))
    is_grade = make_validator(0, 100)

    return is_percentage, is_temperature, is_grade


# =============================================================================
# DEMONSTRATION
# =============================================================================

def main():
    print("\n[PURE FUNCTIONS]")
    print("-" * 50)
    print(f"pure_add(5, 3) = {pure_add(5, 3)}")
    print(f"pure_square(4) = {pure_square(4)}")
    print(f"pure_capitalize_words('hello world') = {pure_capitalize_words('hello world')}")

    # Impure function produces different results
    print("\nImpure function (different results):")
    print(f"impure_increment(5) = {impure_increment(5)}")
    print(f"impure_increment(5) = {impure_increment(5)}")  # Different!

    print("\n[HIGHER-ORDER FUNCTIONS]")
    print("-" * 50)
    multiply_by_3 = make_multiplier(3)
    print(f"multiply_by_3(5) = {multiply_by_3(5)}")

    add_10 = make_adder(10)
    print(f"add_10(5) = {add_10(5)}")

    print(f"apply_twice(lambda x: x * 2, 5) = {apply_twice(lambda x: x * 2, 5)}")

    apply_3_times = apply_n_times(3)
    print(f"apply_3_times(double, 2) = {apply_3_times(double, 2)}")

    print("\n[CLOSURES]")
    print("-" * 50)
    counter1 = make_counter(0)
    counter2 = make_counter(100)

    print(f"counter1(): {counter1()}, {counter1()}, {counter1()}")
    print(f"counter2(): {counter2()}, {counter2()}")

    say_hello = make_greeting("Hello")
    say_bonjour = make_greeting("Bonjour")
    print(f"say_hello('Alice') = {say_hello('Alice')}")
    print(f"say_bonjour('Bob') = {say_bonjour('Bob')}")

    acc = make_accumulator()
    print(f"Accumulator: {acc(10)}, {acc(5)}, {acc(3)}")

    print("\n[FUNCTION COMPOSITION]")
    print("-" * 50)

    # Compose text transformations
    process_text = compose(add_exclamation, to_uppercase, remove_spaces)
    print(f"process_text('hello world') = {process_text('hello world')}")

    # Pipe is more intuitive
    process_text_pipe = pipe(remove_spaces, to_uppercase, add_exclamation)
    print(f"process_text_pipe('hello world') = {process_text_pipe('hello world')}")

    # Compose numeric operations
    process_number = pipe(increment, double, increment)
    print(f"process_number(5) = {process_number(5)}")  # (5+1)*2+1 = 13

    print("\n[CURRYING]")
    print("-" * 50)

    # Regular call
    print(f"add_three_numbers(1, 2, 3) = {add_three_numbers(1, 2, 3)}")

    # Curried call
    print(f"curried_add(1)(2)(3) = {curried_add(1)(2)(3)}")

    # Partial application
    add_5 = curried_add(5)
    add_5_and_10 = add_5(10)
    print(f"curried_add(5)(10)(3) = {add_5_and_10(3)}")

    # Using partial
    print(f"double_then_multiply(3, 4) = {double_then_multiply(3, 4)}")
    print(f"triple_then_multiply(3, 4) = {triple_then_multiply(3, 4)}")

    print("\n[MAP / FILTER / REDUCE]")
    print("-" * 50)

    squared, squared_comp = demonstrate_map()
    print(f"Map - Squared: {squared}")

    evens, evens_comp = demonstrate_filter()
    print(f"Filter - Evens: {evens}")

    total, product, total_op, product_op = demonstrate_reduce()
    print(f"Reduce - Sum: {total}, Product: {product}")

    print("\n[PRACTICAL PIPELINE]")
    print("-" * 50)
    processed_users = practical_pipeline_example()
    for user in processed_users:
        print(f"  {user}")

    print("\n[PRACTICAL FUNCTION FACTORY]")
    print("-" * 50)
    is_percentage, is_temperature, is_grade = practical_function_factory()

    print(f"is_percentage(50) = {is_percentage(50)}")
    print(f"is_percentage(150) = {is_percentage(150)}")
    print(f"is_temperature(-300) = {is_temperature(-300)}")
    print(f"is_grade(85) = {is_grade(85)}")


def print_summary():
    print("\n" + "=" * 70)
    print("FUNCTIONAL PROGRAMMING PATTERNS SUMMARY")
    print("=" * 70)

    print("""
1. PURE FUNCTIONS
   ✓ No side effects
   ✓ Deterministic (same input → same output)
   ✓ Easy to test and reason about
   ✓ Cacheable/memoizable

2. HIGHER-ORDER FUNCTIONS
   ✓ Functions as first-class citizens
   ✓ Can take functions as arguments
   ✓ Can return functions
   ✓ Enables abstraction and reusability

3. CLOSURES
   ✓ Functions that capture variables
   ✓ Private state without classes
   ✓ Function factories
   ✓ Data hiding and encapsulation

4. FUNCTION COMPOSITION
   ✓ Build complex from simple
   ✓ Compose/pipe for readability
   ✓ Reusable transformation chains
   ✓ Declarative style

5. CURRYING
   ✓ Transform multi-arg → single-arg chain
   ✓ Partial application
   ✓ Specialized functions from generic ones
   ✓ Better function reuse

6. MAP/FILTER/REDUCE
   ✓ Fundamental transformations
   ✓ Declarative data processing
   ✓ Composable operations
   ✓ Works with any iterable

BENEFITS:
  • More predictable code
  • Easier to test
  • Better concurrency (immutability)
  • Composability
  • Declarative style

PYTHON TIPS:
  • Use list comprehensions instead of map/filter when simple
  • functools module for partial, reduce
  • operator module for common operations
  • itertools for advanced iteration
  • toolz/fn.py libraries for more FP tools
""")


if __name__ == "__main__":
    main()
    print_summary()
