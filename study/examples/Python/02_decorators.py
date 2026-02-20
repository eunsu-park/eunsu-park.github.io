"""
Python Decorators

Demonstrates:
- Function decorators
- Decorators with arguments
- Class decorators
- functools.wraps
- Practical examples (timing, retry, memoize, logging)
- Stacking decorators
"""

import functools
import time
from typing import Callable, Any, TypeVar


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Basic Function Decorator
# =============================================================================

section("Basic Function Decorator")


def simple_decorator(func: Callable) -> Callable:
    """A simple decorator that prints before/after function execution."""
    def wrapper(*args, **kwargs):
        print(f"  [Before calling {func.__name__}]")
        result = func(*args, **kwargs)
        print(f"  [After calling {func.__name__}]")
        return result
    return wrapper


@simple_decorator
def say_hello(name: str) -> str:
    print(f"  Hello, {name}!")
    return f"Greeted {name}"


result = say_hello("Alice")
print(f"Result: {result}")


# =============================================================================
# functools.wraps - Preserving Metadata
# =============================================================================

section("functools.wraps - Preserving Metadata")


def without_wraps(func: Callable) -> Callable:
    """Decorator without functools.wraps."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def with_wraps(func: Callable) -> Callable:
    """Decorator with functools.wraps."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@without_wraps
def func_a():
    """Original docstring for func_a."""
    pass


@with_wraps
def func_b():
    """Original docstring for func_b."""
    pass


print(f"without_wraps: __name__={func_a.__name__}, __doc__={func_a.__doc__}")
print(f"with_wraps: __name__={func_b.__name__}, __doc__={func_b.__doc__}")


# =============================================================================
# Timing Decorator
# =============================================================================

section("Timing Decorator")


def timing_decorator(func: Callable) -> Callable:
    """Measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"  {func.__name__} took {(end - start) * 1000:.4f} ms")
        return result
    return wrapper


@timing_decorator
def slow_function(n: int) -> int:
    """Simulate slow computation."""
    time.sleep(0.1)
    return sum(range(n))


result = slow_function(1000)
print(f"Result: {result}")


# =============================================================================
# Decorator with Arguments
# =============================================================================

section("Decorator with Arguments")


def repeat(times: int):
    """Decorator that repeats function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for i in range(times):
                print(f"  Iteration {i + 1}:")
                result = func(*args, **kwargs)
                results.append(result)
            return results
        return wrapper
    return decorator


@repeat(times=3)
def greet(name: str) -> str:
    msg = f"    Hello, {name}!"
    print(msg)
    return msg


results = greet("Bob")
print(f"All results: {len(results)} greetings")


# =============================================================================
# Retry Decorator
# =============================================================================

section("Retry Decorator")


def retry(max_attempts: int = 3, delay: float = 0.1):
    """Retry decorator for handling transient failures."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    print(f"  Attempt {attempt}/{max_attempts}")
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"    Failed: {e}")
                    if attempt == max_attempts:
                        print(f"    Max attempts reached, giving up")
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator


# Simulated unreliable function
call_count = 0


@retry(max_attempts=4, delay=0.05)
def unreliable_function():
    """Fails first 2 times, succeeds on 3rd."""
    global call_count
    call_count += 1
    if call_count < 3:
        raise ValueError(f"Simulated failure #{call_count}")
    return "Success!"


try:
    result = unreliable_function()
    print(f"Final result: {result}")
except Exception as e:
    print(f"Final exception: {e}")


# =============================================================================
# Memoization Decorator
# =============================================================================

section("Memoization Decorator")


def memoize(func: Callable) -> Callable:
    """Cache function results."""
    cache = {}

    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            print(f"  Computing {func.__name__}{args}...")
            cache[args] = func(*args)
        else:
            print(f"  Returning cached result for {func.__name__}{args}")
        return cache[args]

    return wrapper


@memoize
def fibonacci(n: int) -> int:
    """Compute nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


print(f"fibonacci(5) = {fibonacci(5)}")
print(f"fibonacci(5) = {fibonacci(5)}")  # Second call uses cache


# Compare with functools.lru_cache
section("functools.lru_cache")


@functools.lru_cache(maxsize=128)
def fib_cached(n: int) -> int:
    """Fibonacci with built-in LRU cache."""
    if n <= 1:
        return n
    return fib_cached(n - 1) + fib_cached(n - 2)


print(f"fib_cached(10) = {fib_cached(10)}")
print(f"Cache info: {fib_cached.cache_info()}")


# =============================================================================
# Class Decorator
# =============================================================================

section("Class Decorator")


def singleton(cls):
    """Singleton decorator - only one instance allowed."""
    instances = {}

    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            print(f"  Creating new instance of {cls.__name__}")
            instances[cls] = cls(*args, **kwargs)
        else:
            print(f"  Returning existing instance of {cls.__name__}")
        return instances[cls]

    return get_instance


@singleton
class Database:
    """Singleton database connection."""

    def __init__(self, name: str):
        self.name = name
        print(f"  Database '{name}' initialized")


db1 = Database("production")
db2 = Database("production")
print(f"db1 is db2: {db1 is db2}")


# =============================================================================
# Stacking Decorators
# =============================================================================

section("Stacking Decorators")


def uppercase(func: Callable) -> Callable:
    """Convert result to uppercase."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper()
    return wrapper


def exclaim(func: Callable) -> Callable:
    """Add exclamation marks."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return f"{result}!!!"
    return wrapper


@exclaim
@uppercase
def get_message(name: str) -> str:
    """Get a greeting message."""
    return f"hello {name}"


# Applied bottom-up: uppercase first, then exclaim
print(f"get_message('world') = {get_message('world')}")


# =============================================================================
# Logging Decorator
# =============================================================================

section("Logging Decorator")


def log_calls(func: Callable) -> Callable:
    """Log function calls with arguments and results."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"  Calling {func.__name__}({signature})")
        result = func(*args, **kwargs)
        print(f"  {func.__name__} returned {result!r}")
        return result
    return wrapper


@log_calls
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


@log_calls
def greet_person(name: str, greeting: str = "Hello") -> str:
    """Greet a person."""
    return f"{greeting}, {name}!"


add(3, 5)
greet_person("Charlie", greeting="Hi")


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Decorator patterns covered:
1. Basic decorators - wrap function behavior
2. functools.wraps - preserve function metadata
3. Decorators with arguments - parametrize behavior
4. Timing decorator - performance monitoring
5. Retry decorator - error handling
6. Memoization - cache results
7. Class decorators - modify class behavior
8. Stacking decorators - compose multiple decorators
9. Logging decorator - debug and audit

Decorators provide clean separation of concerns and reusable
cross-cutting functionality.
""")
