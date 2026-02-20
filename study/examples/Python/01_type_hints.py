"""
Type Hints and Annotations in Python

Demonstrates:
- Basic type annotations
- Generic types (List, Dict, Tuple)
- Union and Optional
- TypeVar and Generic classes
- Protocol (structural subtyping)
- Runtime type checking
"""

from typing import (
    List, Dict, Tuple, Union, Optional, TypeVar, Generic, Protocol, Callable
)
from typing import get_type_hints, get_args, get_origin


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Basic Type Annotations
# =============================================================================

section("Basic Type Annotations")


def greet(name: str, age: int) -> str:
    """Function with basic type hints."""
    return f"Hello {name}, you are {age} years old"


result = greet("Alice", 30)
print(f"greet('Alice', 30) -> {result}")
print(f"Function annotations: {greet.__annotations__}")


# =============================================================================
# Collection Types
# =============================================================================

section("Collection Types")


def process_scores(scores: List[int]) -> Dict[str, float]:
    """Process a list of scores and return statistics."""
    return {
        "mean": sum(scores) / len(scores) if scores else 0,
        "max": max(scores) if scores else 0,
        "min": min(scores) if scores else 0
    }


scores = [85, 92, 78, 95, 88]
stats = process_scores(scores)
print(f"Scores: {scores}")
print(f"Statistics: {stats}")


def coordinates() -> Tuple[float, float, float]:
    """Return 3D coordinates."""
    return (1.5, 2.7, 3.9)


coords = coordinates()
print(f"Coordinates: {coords}")


# =============================================================================
# Union and Optional
# =============================================================================

section("Union and Optional")


def process_id(user_id: Union[int, str]) -> str:
    """Accept either int or str ID."""
    return f"Processing ID: {user_id} (type: {type(user_id).__name__})"


print(process_id(12345))
print(process_id("USER_789"))


def find_user(user_id: int) -> Optional[Dict[str, str]]:
    """Return user dict or None if not found."""
    users = {1: {"name": "Alice"}, 2: {"name": "Bob"}}
    return users.get(user_id)


user = find_user(1)
print(f"find_user(1): {user}")
user = find_user(99)
print(f"find_user(99): {user}")


# =============================================================================
# TypeVar and Generics
# =============================================================================

section("TypeVar and Generics")

T = TypeVar('T')


def first_element(items: List[T]) -> Optional[T]:
    """Return first element of list, preserving type."""
    return items[0] if items else None


int_list = [1, 2, 3]
str_list = ["a", "b", "c"]
print(f"first_element({int_list}) -> {first_element(int_list)}")
print(f"first_element({str_list}) -> {first_element(str_list)}")


class Stack(Generic[T]):
    """Generic stack implementation."""

    def __init__(self) -> None:
        self._items: List[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> Optional[T]:
        return self._items.pop() if self._items else None

    def __repr__(self) -> str:
        return f"Stack({self._items})"


int_stack: Stack[int] = Stack()
int_stack.push(10)
int_stack.push(20)
print(f"int_stack: {int_stack}")
print(f"Popped: {int_stack.pop()}")

str_stack: Stack[str] = Stack()
str_stack.push("hello")
str_stack.push("world")
print(f"str_stack: {str_stack}")


# =============================================================================
# Protocol (Structural Subtyping)
# =============================================================================

section("Protocol (Structural Subtyping)")


class Drawable(Protocol):
    """Protocol for drawable objects."""

    def draw(self) -> str:
        ...


class Circle:
    """Circle - implements Drawable protocol without inheritance."""

    def __init__(self, radius: float):
        self.radius = radius

    def draw(self) -> str:
        return f"Drawing circle with radius {self.radius}"


class Square:
    """Square - also implements Drawable protocol."""

    def __init__(self, side: float):
        self.side = side

    def draw(self) -> str:
        return f"Drawing square with side {self.side}"


def render(obj: Drawable) -> None:
    """Render any drawable object."""
    print(obj.draw())


circle = Circle(5.0)
square = Square(10.0)
render(circle)
render(square)


# =============================================================================
# Callable Types
# =============================================================================

section("Callable Types")


def apply_operation(x: int, y: int, operation: Callable[[int, int], int]) -> int:
    """Apply a binary operation to two integers."""
    return operation(x, y)


def add(a: int, b: int) -> int:
    return a + b


def multiply(a: int, b: int) -> int:
    return a * b


result1 = apply_operation(5, 3, add)
result2 = apply_operation(5, 3, multiply)
print(f"apply_operation(5, 3, add) -> {result1}")
print(f"apply_operation(5, 3, multiply) -> {result2}")


# =============================================================================
# Runtime Type Checking
# =============================================================================

section("Runtime Type Checking with isinstance")


def process_value(value: Union[int, str, List[int]]) -> str:
    """Process different types at runtime."""
    if isinstance(value, int):
        return f"Integer: {value * 2}"
    elif isinstance(value, str):
        return f"String: {value.upper()}"
    elif isinstance(value, list):
        return f"List sum: {sum(value)}"
    else:
        return "Unknown type"


print(process_value(42))
print(process_value("hello"))
print(process_value([1, 2, 3, 4, 5]))


# =============================================================================
# Type Introspection
# =============================================================================

section("Type Introspection")


def example_function(x: int, y: str = "default") -> bool:
    """Example function for introspection."""
    return True


hints = get_type_hints(example_function)
print(f"Type hints for example_function: {hints}")

# Introspect Union type
union_type = Union[int, str]
print(f"\nUnion[int, str]:")
print(f"  Origin: {get_origin(union_type)}")
print(f"  Args: {get_args(union_type)}")

# Introspect List type
list_type = List[int]
print(f"\nList[int]:")
print(f"  Origin: {get_origin(list_type)}")
print(f"  Args: {get_args(list_type)}")


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Type hints provide several benefits:
1. Better IDE support (autocomplete, refactoring)
2. Early error detection with static type checkers (mypy, pyright)
3. Self-documenting code
4. Runtime type checking with isinstance()
5. Generic programming with TypeVar and Generic

Note: Type hints are optional and not enforced at runtime
unless you explicitly check them.
""")
