"""
Python Performance Optimization

Demonstrates:
- timeit module
- cProfile profiling
- __slots__ for memory optimization
- List comprehensions vs loops
- String concatenation performance
- collections module optimizations
- Generator expressions
- Local variable lookups
"""

import timeit
import cProfile
import pstats
import io
import sys
from collections import defaultdict, Counter, deque, namedtuple


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# timeit - Measuring Execution Time
# =============================================================================

section("timeit - Measuring Execution Time")

# Simple timing
def slow_loop():
    result = []
    for i in range(1000):
        result.append(i ** 2)
    return result


def fast_comprehension():
    return [i ** 2 for i in range(1000)]


# Time with timeit.timeit
time1 = timeit.timeit(slow_loop, number=1000)
time2 = timeit.timeit(fast_comprehension, number=1000)

print(f"Loop: {time1:.4f} seconds")
print(f"Comprehension: {time2:.4f} seconds")
print(f"Speedup: {time1 / time2:.2f}x")


# Time with timeit.repeat
times = timeit.repeat('"-".join(str(i) for i in range(100))', number=1000, repeat=5)
print(f"\nString join (5 runs):")
print(f"  Min: {min(times):.4f}s")
print(f"  Max: {max(times):.4f}s")
print(f"  Avg: {sum(times)/len(times):.4f}s")


# =============================================================================
# List Comprehensions vs Loops
# =============================================================================

section("List Comprehensions vs Loops")


def with_loop(n):
    result = []
    for i in range(n):
        if i % 2 == 0:
            result.append(i ** 2)
    return result


def with_comprehension(n):
    return [i ** 2 for i in range(n) if i % 2 == 0]


n = 10000
t1 = timeit.timeit(lambda: with_loop(n), number=100)
t2 = timeit.timeit(lambda: with_comprehension(n), number=100)

print(f"Loop: {t1:.4f}s")
print(f"Comprehension: {t2:.4f}s")
print(f"Speedup: {t1/t2:.2f}x")


# =============================================================================
# String Concatenation
# =============================================================================

section("String Concatenation Performance")


def concat_with_plus(n):
    """Slow - creates new string each iteration."""
    result = ""
    for i in range(n):
        result += str(i)
    return result


def concat_with_join(n):
    """Fast - builds list then joins once."""
    parts = []
    for i in range(n):
        parts.append(str(i))
    return "".join(parts)


def concat_with_comprehension(n):
    """Fastest - comprehension + join."""
    return "".join(str(i) for i in range(n))


n = 1000
t1 = timeit.timeit(lambda: concat_with_plus(n), number=100)
t2 = timeit.timeit(lambda: concat_with_join(n), number=100)
t3 = timeit.timeit(lambda: concat_with_comprehension(n), number=100)

print(f"Plus operator: {t1:.4f}s")
print(f"Join with list: {t2:.4f}s")
print(f"Join with generator: {t3:.4f}s")
print(f"Plus vs Join speedup: {t1/t2:.2f}x")


# =============================================================================
# __slots__ for Memory Optimization
# =============================================================================

section("__slots__ for Memory Optimization")


class PointWithDict:
    """Regular class with __dict__."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


class PointWithSlots:
    """Class with __slots__ - no __dict__."""
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y


# Compare memory usage
p1 = PointWithDict(10, 20)
p2 = PointWithSlots(10, 20)

print(f"PointWithDict size: {sys.getsizeof(p1) + sys.getsizeof(p1.__dict__)} bytes")
print(f"PointWithSlots size: {sys.getsizeof(p2)} bytes")

# Time attribute access
t1 = timeit.timeit('p.x', globals={'p': p1}, number=1000000)
t2 = timeit.timeit('p.x', globals={'p': p2}, number=1000000)

print(f"\nAttribute access:")
print(f"  With __dict__: {t1:.4f}s")
print(f"  With __slots__: {t2:.4f}s")


# =============================================================================
# collections.defaultdict
# =============================================================================

section("collections.defaultdict")


def count_words_dict(words):
    """Count words with regular dict."""
    counts = {}
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts


def count_words_defaultdict(words):
    """Count words with defaultdict."""
    counts = defaultdict(int)
    for word in words:
        counts[word] += 1
    return counts


words = ["apple", "banana", "apple", "cherry", "banana", "apple"] * 1000

t1 = timeit.timeit(lambda: count_words_dict(words), number=100)
t2 = timeit.timeit(lambda: count_words_defaultdict(words), number=100)

print(f"Regular dict: {t1:.4f}s")
print(f"defaultdict: {t2:.4f}s")
print(f"Speedup: {t1/t2:.2f}x")


# =============================================================================
# collections.Counter
# =============================================================================

section("collections.Counter")


def count_manual(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def count_with_counter(items):
    return Counter(items)


items = list(range(100)) * 100

t1 = timeit.timeit(lambda: count_manual(items), number=100)
t2 = timeit.timeit(lambda: count_with_counter(items), number=100)

print(f"Manual counting: {t1:.4f}s")
print(f"Counter: {t2:.4f}s")
print(f"Speedup: {t1/t2:.2f}x")

# Counter features
counter = Counter(['a', 'b', 'c', 'a', 'b', 'a'])
print(f"\nCounter example: {counter}")
print(f"Most common: {counter.most_common(2)}")


# =============================================================================
# collections.deque
# =============================================================================

section("collections.deque vs list")


def append_left_list(n):
    """Slow - O(n) for each insert."""
    lst = []
    for i in range(n):
        lst.insert(0, i)
    return lst


def append_left_deque(n):
    """Fast - O(1) for each appendleft."""
    dq = deque()
    for i in range(n):
        dq.appendleft(i)
    return dq


n = 1000
t1 = timeit.timeit(lambda: append_left_list(n), number=10)
t2 = timeit.timeit(lambda: append_left_deque(n), number=10)

print(f"list.insert(0, x): {t1:.4f}s")
print(f"deque.appendleft(x): {t2:.4f}s")
print(f"Speedup: {t1/t2:.2f}x")


# =============================================================================
# Generator Expressions vs List Comprehensions
# =============================================================================

section("Generator vs List Comprehension")


def sum_list_comp(n):
    """Creates entire list in memory."""
    return sum([i ** 2 for i in range(n)])


def sum_generator(n):
    """Lazy evaluation - no intermediate list."""
    return sum(i ** 2 for i in range(n))


n = 100000
t1 = timeit.timeit(lambda: sum_list_comp(n), number=100)
t2 = timeit.timeit(lambda: sum_generator(n), number=100)

print(f"List comprehension: {t1:.4f}s")
print(f"Generator expression: {t2:.4f}s")

# Memory comparison
import sys
list_size = sys.getsizeof([i for i in range(10000)])
gen_size = sys.getsizeof(i for i in range(10000))
print(f"\nMemory (10k items):")
print(f"  List: {list_size} bytes")
print(f"  Generator: {gen_size} bytes")


# =============================================================================
# Local Variable Lookups
# =============================================================================

section("Local Variable Lookups")


def with_global_lookup():
    """Slower - global lookup each iteration."""
    result = 0
    for i in range(10000):
        result += len(str(i))
    return result


def with_local_lookup():
    """Faster - local variable."""
    _len = len
    _str = str
    result = 0
    for i in range(10000):
        result += _len(_str(i))
    return result


t1 = timeit.timeit(with_global_lookup, number=100)
t2 = timeit.timeit(with_local_lookup, number=100)

print(f"Global lookups: {t1:.4f}s")
print(f"Local lookups: {t2:.4f}s")
print(f"Speedup: {t1/t2:.2f}x")


# =============================================================================
# namedtuple vs dict
# =============================================================================

section("namedtuple vs dict")

Point = namedtuple('Point', ['x', 'y'])


def create_dicts(n):
    return [{'x': i, 'y': i*2} for i in range(n)]


def create_tuples(n):
    return [Point(i, i*2) for i in range(n)]


n = 10000
t1 = timeit.timeit(lambda: create_dicts(n), number=100)
t2 = timeit.timeit(lambda: create_tuples(n), number=100)

print(f"dict creation: {t1:.4f}s")
print(f"namedtuple creation: {t2:.4f}s")
print(f"Speedup: {t1/t2:.2f}x")

# Access time
d = {'x': 10, 'y': 20}
t = Point(10, 20)

t1 = timeit.timeit('d["x"]', globals={'d': d}, number=1000000)
t2 = timeit.timeit('t.x', globals={'t': t}, number=1000000)

print(f"\nAccess time:")
print(f"  dict['x']: {t1:.4f}s")
print(f"  tuple.x: {t2:.4f}s")


# =============================================================================
# cProfile - Profiling Code
# =============================================================================

section("cProfile - Profiling Code")


def fibonacci(n):
    """Inefficient recursive Fibonacci."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def test_function():
    """Function to profile."""
    results = []
    for i in range(15):
        results.append(fibonacci(i))
    return results


# Profile the function
profiler = cProfile.Profile()
profiler.enable()
test_function()
profiler.disable()

# Print stats
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(10)

print("Top 10 functions by cumulative time:")
print(s.getvalue()[:800])  # Print first 800 chars


# =============================================================================
# Set Membership
# =============================================================================

section("Set vs List Membership")


def search_in_list(items, targets):
    """O(n) lookup."""
    return [item in items for item in targets]


def search_in_set(items, targets):
    """O(1) lookup."""
    item_set = set(items)
    return [item in item_set for item in targets]


items = list(range(1000))
targets = list(range(500, 1500))

t1 = timeit.timeit(lambda: search_in_list(items, targets), number=100)
t2 = timeit.timeit(lambda: search_in_set(items, targets), number=100)

print(f"List membership: {t1:.4f}s")
print(f"Set membership: {t2:.4f}s")
print(f"Speedup: {t1/t2:.2f}x")


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Performance optimization techniques:
1. timeit - measure execution time accurately
2. List comprehensions - faster than loops
3. String join - faster than concatenation
4. __slots__ - reduce memory for many instances
5. collections.defaultdict - avoid key checks
6. collections.Counter - optimized counting
7. collections.deque - fast append/pop from both ends
8. Generator expressions - memory efficient
9. Local variable lookups - faster than global
10. namedtuple - faster and lighter than dict
11. cProfile - identify performance bottlenecks
12. Set membership - O(1) vs O(n) for lists

General principles:
- Measure before optimizing (profile first!)
- Use built-in functions and types
- Avoid premature optimization
- Choose right data structure for use case
- Use generators for large datasets
- Cache expensive computations
""")
