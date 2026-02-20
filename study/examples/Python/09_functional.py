"""
Functional Programming in Python

Demonstrates:
- functools module (partial, reduce, lru_cache)
- operator module
- map, filter, zip
- Lambda functions
- Function composition
- Immutability patterns
- Higher-order functions
"""

import functools
import operator
from typing import Callable, Iterable, Any, List


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Lambda Functions
# =============================================================================

section("Lambda Functions")

# Basic lambda
add = lambda x, y: x + y
print(f"add(3, 5) = {add(3, 5)}")

# Lambda with map
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(f"map(lambda x: x**2, {numbers}) = {squared}")

# Lambda with filter
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"filter(lambda x: x % 2 == 0, {numbers}) = {evens}")

# Lambda with sorted
words = ["apple", "pie", "zoo", "car"]
sorted_by_length = sorted(words, key=lambda x: len(x))
print(f"sorted by length: {sorted_by_length}")


# =============================================================================
# map, filter, zip
# =============================================================================

section("map, filter, zip")

# map - apply function to all items
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))
print(f"Doubled: {doubled}")

# Multiple iterables
a = [1, 2, 3]
b = [10, 20, 30]
summed = list(map(lambda x, y: x + y, a, b))
print(f"map(add, {a}, {b}) = {summed}")

# filter - keep items matching predicate
numbers = range(10)
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Even numbers: {evens}")

# zip - combine iterables
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
combined = list(zip(names, ages))
print(f"Zipped: {combined}")

# Unzip with zip(*...)
unzipped_names, unzipped_ages = zip(*combined)
print(f"Unzipped names: {list(unzipped_names)}")
print(f"Unzipped ages: {list(unzipped_ages)}")


# =============================================================================
# functools.partial
# =============================================================================

section("functools.partial")


def power(base: float, exponent: float) -> float:
    """Raise base to exponent."""
    return base ** exponent


# Create specialized functions
square = functools.partial(power, exponent=2)
cube = functools.partial(power, exponent=3)

print(f"square(5) = {square(5)}")
print(f"cube(3) = {cube(3)}")


def multiply(x: float, y: float, z: float) -> float:
    """Multiply three numbers."""
    return x * y * z


# Partial application
double = functools.partial(multiply, 2)
print(f"double(3, 4) = {double(3, 4)}")

triple_and_double = functools.partial(multiply, 2, 3)
print(f"triple_and_double(5) = {triple_and_double(5)}")


# =============================================================================
# functools.reduce
# =============================================================================

section("functools.reduce")

numbers = [1, 2, 3, 4, 5]

# Sum using reduce
total = functools.reduce(lambda x, y: x + y, numbers)
print(f"Sum of {numbers} = {total}")

# Product using reduce
product = functools.reduce(lambda x, y: x * y, numbers)
print(f"Product of {numbers} = {product}")

# With initial value
result = functools.reduce(lambda x, y: x + y, numbers, 100)
print(f"Sum with initial value 100: {result}")

# Find maximum
max_value = functools.reduce(lambda x, y: x if x > y else y, numbers)
print(f"Maximum: {max_value}")


# =============================================================================
# operator Module
# =============================================================================

section("operator Module")

# Arithmetic operators as functions
print(f"operator.add(3, 5) = {operator.add(3, 5)}")
print(f"operator.mul(4, 7) = {operator.mul(4, 7)}")
print(f"operator.pow(2, 10) = {operator.pow(2, 10)}")

# Comparison operators
print(f"operator.gt(5, 3) = {operator.gt(5, 3)}")
print(f"operator.eq(5, 5) = {operator.eq(5, 5)}")

# Item getters
data = [10, 20, 30, 40]
get_second = operator.itemgetter(1)
print(f"itemgetter(1) on {data} = {get_second(data)}")

# Multiple items
get_items = operator.itemgetter(0, 2)
print(f"itemgetter(0, 2) on {data} = {get_items(data)}")

# Attribute getter
from collections import namedtuple
Person = namedtuple('Person', ['name', 'age'])
people = [Person('Alice', 30), Person('Bob', 25), Person('Charlie', 35)]

get_name = operator.attrgetter('name')
names = list(map(get_name, people))
print(f"Names: {names}")

# Sort by attribute
sorted_people = sorted(people, key=operator.attrgetter('age'))
print(f"Sorted by age: {[f'{p.name}({p.age})' for p in sorted_people]}")


# =============================================================================
# functools.lru_cache
# =============================================================================

section("functools.lru_cache")


@functools.lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    """Fibonacci with memoization."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


print("Computing Fibonacci numbers with caching:")
for i in [10, 20, 10, 15]:  # 10 will use cache on second call
    result = fibonacci(i)
    print(f"  fib({i}) = {result}")

print(f"\nCache info: {fibonacci.cache_info()}")
fibonacci.cache_clear()
print("Cache cleared")


# =============================================================================
# Higher-Order Functions
# =============================================================================

section("Higher-Order Functions")


def apply_twice(func: Callable, x: Any) -> Any:
    """Apply function twice."""
    return func(func(x))


def add_five(x: int) -> int:
    return x + 5


result = apply_twice(add_five, 10)
print(f"apply_twice(add_five, 10) = {result}")


def make_multiplier(factor: float) -> Callable:
    """Return a function that multiplies by factor."""
    def multiplier(x: float) -> float:
        return x * factor
    return multiplier


times_three = make_multiplier(3)
times_ten = make_multiplier(10)

print(f"times_three(5) = {times_three(5)}")
print(f"times_ten(5) = {times_ten(5)}")


# =============================================================================
# Function Composition
# =============================================================================

section("Function Composition")


def compose(*functions):
    """Compose functions right to left."""
    def inner(arg):
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    return inner


def add_one(x: int) -> int:
    return x + 1


def multiply_by_two(x: int) -> int:
    return x * 2


def square(x: int) -> int:
    return x ** 2


# Compose: square(multiply_by_two(add_one(x)))
composed = compose(square, multiply_by_two, add_one)
result = composed(5)  # ((5 + 1) * 2) ** 2 = (6 * 2) ** 2 = 144
print(f"composed(5) = {result}")


def pipe(*functions):
    """Compose functions left to right."""
    def inner(arg):
        result = arg
        for func in functions:
            result = func(result)
        return result
    return inner


# Pipe: add_one -> multiply_by_two -> square
piped = pipe(add_one, multiply_by_two, square)
result = piped(5)  # Same as compose example
print(f"piped(5) = {result}")


# =============================================================================
# List Comprehensions vs Functional Style
# =============================================================================

section("List Comprehensions vs Functional Style")

numbers = range(1, 11)

# Comprehension style
result_comp = [x**2 for x in numbers if x % 2 == 0]
print(f"Comprehension: {result_comp}")

# Functional style
result_func = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, numbers)))
print(f"Functional: {result_func}")

# Nested comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened_comp = [item for row in matrix for item in row]
print(f"Flattened (comp): {flattened_comp}")

# Functional style
flattened_func = functools.reduce(operator.add, matrix)
print(f"Flattened (func): {flattened_func}")


# =============================================================================
# Immutability Patterns
# =============================================================================

section("Immutability Patterns")

# Tuples are immutable
point = (10, 20)
print(f"Original point: {point}")

# Create new tuple instead of modifying
moved_point = (point[0] + 5, point[1] + 3)
print(f"Moved point: {moved_point}")

# Frozen sets
frozen = frozenset([1, 2, 3, 4])
print(f"Frozen set: {frozen}")

# Named tuples for immutable records
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p1 = Point(10, 20)
print(f"NamedTuple: {p1}")

# "Update" by creating new instance
p2 = p1._replace(x=15)
print(f"Updated: {p2} (original: {p1})")


# =============================================================================
# Currying Pattern
# =============================================================================

section("Currying Pattern")


def curry(func: Callable) -> Callable:
    """Convert function to curried form."""
    @functools.wraps(func)
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return functools.partial(curried, *args, **kwargs)
    return curried


@curry
def add_three(a: int, b: int, c: int) -> int:
    """Add three numbers."""
    return a + b + c


# Can call with all args at once
print(f"add_three(1, 2, 3) = {add_three(1, 2, 3)}")

# Or curry it
add_1 = add_three(1)
add_1_2 = add_1(2)
result = add_1_2(3)
print(f"Curried: {result}")


# =============================================================================
# Pure Functions
# =============================================================================

section("Pure Functions")


# Pure function - no side effects, deterministic
def pure_sum(numbers: List[int]) -> int:
    """Pure function - always returns same output for same input."""
    return sum(numbers)


# Impure function - modifies external state
counter = 0


def impure_increment():
    """Impure function - has side effects."""
    global counter
    counter += 1
    return counter


data = [1, 2, 3, 4, 5]
print(f"pure_sum({data}) = {pure_sum(data)}")
print(f"pure_sum({data}) = {pure_sum(data)}")  # Same result

print(f"impure_increment() = {impure_increment()}")
print(f"impure_increment() = {impure_increment()}")  # Different result


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Functional programming patterns covered:
1. Lambda functions - anonymous functions
2. map, filter, zip - transform iterables
3. functools.partial - partial application
4. functools.reduce - accumulate values
5. operator module - operators as functions
6. functools.lru_cache - memoization
7. Higher-order functions - functions as arguments/return values
8. Function composition - combine functions
9. Immutability - prefer immutable data structures
10. Pure functions - no side effects

Benefits of functional programming:
- More predictable code (pure functions)
- Easier to test and reason about
- Better composability
- Natural parallelization (no shared state)

Python supports functional programming but isn't purely functional.
Use functional patterns where they improve code clarity.
""")
