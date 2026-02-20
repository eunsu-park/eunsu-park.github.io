"""
Python Pattern Matching (Python 3.10+)

Demonstrates:
- match/case statement basics
- Literal patterns
- Capture patterns
- Wildcard patterns
- OR patterns
- Guards (if clauses)
- Class patterns
- Sequence patterns
- Mapping patterns
- AS patterns
"""

from dataclasses import dataclass
from typing import Any


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Basic Pattern Matching
# =============================================================================

section("Basic Pattern Matching")


def describe_number(n: int) -> str:
    """Match literal values."""
    match n:
        case 0:
            return "zero"
        case 1:
            return "one"
        case 2:
            return "two"
        case _:
            return f"many ({n})"


print(f"describe_number(0): {describe_number(0)}")
print(f"describe_number(1): {describe_number(1)}")
print(f"describe_number(42): {describe_number(42)}")


# =============================================================================
# OR Patterns
# =============================================================================

section("OR Patterns")


def classify_status_code(code: int) -> str:
    """Match multiple values."""
    match code:
        case 200 | 201 | 202:
            return "Success"
        case 400 | 401 | 403 | 404:
            return "Client Error"
        case 500 | 502 | 503:
            return "Server Error"
        case _:
            return "Unknown"


print(f"Status 200: {classify_status_code(200)}")
print(f"Status 404: {classify_status_code(404)}")
print(f"Status 500: {classify_status_code(500)}")
print(f"Status 999: {classify_status_code(999)}")


# =============================================================================
# Capture Patterns
# =============================================================================

section("Capture Patterns")


def process_point(point: tuple) -> str:
    """Capture matched values."""
    match point:
        case (0, 0):
            return "Origin"
        case (0, y):
            return f"On Y-axis at y={y}"
        case (x, 0):
            return f"On X-axis at x={x}"
        case (x, y):
            return f"Point at ({x}, {y})"


print(f"(0, 0): {process_point((0, 0))}")
print(f"(0, 5): {process_point((0, 5))}")
print(f"(3, 0): {process_point((3, 0))}")
print(f"(3, 4): {process_point((3, 4))}")


# =============================================================================
# Guards (if clauses)
# =============================================================================

section("Guards (if clauses)")


def categorize_point(point: tuple) -> str:
    """Use guards to add conditions."""
    match point:
        case (x, y) if x == y:
            return f"On diagonal: ({x}, {y})"
        case (x, y) if x > 0 and y > 0:
            return f"Quadrant I: ({x}, {y})"
        case (x, y) if x < 0 and y > 0:
            return f"Quadrant II: ({x}, {y})"
        case (x, y) if x < 0 and y < 0:
            return f"Quadrant III: ({x}, {y})"
        case (x, y) if x > 0 and y < 0:
            return f"Quadrant IV: ({x}, {y})"
        case _:
            return "On axis or origin"


print(f"(5, 5): {categorize_point((5, 5))}")
print(f"(3, 4): {categorize_point((3, 4))}")
print(f"(-3, 4): {categorize_point((-3, 4))}")
print(f"(3, -4): {categorize_point((3, -4))}")


# =============================================================================
# Sequence Patterns
# =============================================================================

section("Sequence Patterns")


def analyze_list(data: list) -> str:
    """Match sequences."""
    match data:
        case []:
            return "Empty list"
        case [x]:
            return f"Single element: {x}"
        case [x, y]:
            return f"Two elements: {x}, {y}"
        case [x, y, z]:
            return f"Three elements: {x}, {y}, {z}"
        case [first, *middle, last]:
            return f"Multiple elements: first={first}, middle={middle}, last={last}"


print(f"[]: {analyze_list([])}")
print(f"[1]: {analyze_list([1])}")
print(f"[1, 2]: {analyze_list([1, 2])}")
print(f"[1, 2, 3]: {analyze_list([1, 2, 3])}")
print(f"[1, 2, 3, 4, 5]: {analyze_list([1, 2, 3, 4, 5])}")


# =============================================================================
# Wildcard and Rest Patterns
# =============================================================================

section("Wildcard and Rest Patterns")


def parse_command(cmd: list) -> str:
    """Parse command with rest pattern."""
    match cmd:
        case ["quit"]:
            return "Quitting..."
        case ["help", topic]:
            return f"Help for: {topic}"
        case ["create", resource, *options]:
            return f"Creating {resource} with options: {options}"
        case ["delete", resource]:
            return f"Deleting {resource}"
        case _:
            return "Unknown command"


print(f"['quit']: {parse_command(['quit'])}")
print(f"['help', 'commands']: {parse_command(['help', 'commands'])}")
print(f"['create', 'user', '--admin']: {parse_command(['create', 'user', '--admin'])}")
print(f"['create', 'file', 'a.txt', 'b.txt']: {parse_command(['create', 'file', 'a.txt', 'b.txt'])}")


# =============================================================================
# Mapping Patterns
# =============================================================================

section("Mapping Patterns")


def handle_event(event: dict) -> str:
    """Match dictionary patterns."""
    match event:
        case {"type": "click", "x": x, "y": y}:
            return f"Click at ({x}, {y})"
        case {"type": "keypress", "key": key}:
            return f"Key pressed: {key}"
        case {"type": "scroll", "direction": direction, **rest}:
            return f"Scroll {direction}, data: {rest}"
        case {"type": event_type}:
            return f"Event: {event_type} (no additional data)"
        case _:
            return "Unknown event"


print(handle_event({"type": "click", "x": 100, "y": 200}))
print(handle_event({"type": "keypress", "key": "Enter"}))
print(handle_event({"type": "scroll", "direction": "down", "delta": 10}))
print(handle_event({"type": "custom"}))


# =============================================================================
# Class Patterns
# =============================================================================

section("Class Patterns")


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Circle:
    center: Point
    radius: float


@dataclass
class Rectangle:
    top_left: Point
    width: float
    height: float


def describe_shape(shape) -> str:
    """Match dataclass instances."""
    match shape:
        case Point(x=0, y=0):
            return "Point at origin"
        case Point(x=x, y=y):
            return f"Point at ({x}, {y})"
        case Circle(center=Point(x=cx, y=cy), radius=r):
            return f"Circle centered at ({cx}, {cy}) with radius {r}"
        case Rectangle(top_left=Point(x=x, y=y), width=w, height=h):
            return f"Rectangle at ({x}, {y}), size {w}x{h}"
        case _:
            return "Unknown shape"


p = Point(5, 10)
c = Circle(Point(0, 0), 5)
r = Rectangle(Point(10, 20), 30, 40)

print(describe_shape(p))
print(describe_shape(c))
print(describe_shape(r))


# =============================================================================
# Nested Patterns
# =============================================================================

section("Nested Patterns")


def evaluate(expr: list) -> float:
    """Simple expression evaluator with nested patterns."""
    match expr:
        case ["+", a, b]:
            return evaluate(a) + evaluate(b)
        case ["-", a, b]:
            return evaluate(a) - evaluate(b)
        case ["*", a, b]:
            return evaluate(a) * evaluate(b)
        case ["/", a, b]:
            return evaluate(a) / evaluate(b)
        case int(n) | float(n):
            return n
        case _:
            raise ValueError(f"Invalid expression: {expr}")


expr1 = ["+", 10, 5]
expr2 = ["*", ["+", 3, 2], 4]
expr3 = ["/", ["-", 20, 5], 3]

print(f"{expr1} = {evaluate(expr1)}")
print(f"{expr2} = {evaluate(expr2)}")
print(f"{expr3} = {evaluate(expr3)}")


# =============================================================================
# AS Patterns
# =============================================================================

section("AS Patterns (Capture and Match)")


def process_data(data: Any) -> str:
    """Use AS patterns to capture matched values."""
    match data:
        case [x, y] as point:
            return f"2D point {point}: x={x}, y={y}"
        case [x, y, z] as point:
            return f"3D point {point}: x={x}, y={y}, z={z}"
        case {"name": name, "age": age} as person:
            return f"Person {person}: {name} is {age} years old"
        case _:
            return "Unknown data format"


print(process_data([1, 2]))
print(process_data([1, 2, 3]))
print(process_data({"name": "Alice", "age": 30}))


# =============================================================================
# Type Patterns
# =============================================================================

section("Type Patterns")


def handle_value(value: Any) -> str:
    """Match by type."""
    match value:
        case int(n) if n < 0:
            return f"Negative integer: {n}"
        case int(n):
            return f"Positive integer: {n}"
        case float(x):
            return f"Float: {x}"
        case str(s):
            return f"String: '{s}'"
        case list(items):
            return f"List with {len(items)} items"
        case dict(mapping):
            return f"Dict with {len(mapping)} keys"
        case _:
            return f"Other type: {type(value).__name__}"


print(handle_value(42))
print(handle_value(-10))
print(handle_value(3.14))
print(handle_value("hello"))
print(handle_value([1, 2, 3]))
print(handle_value({"a": 1, "b": 2}))


# =============================================================================
# Complex Example: JSON Processing
# =============================================================================

section("Complex Example: JSON Processing")


def process_json(data: dict) -> str:
    """Process JSON-like structure."""
    match data:
        case {
            "type": "user",
            "name": str(name),
            "email": str(email),
            "age": int(age)
        } if age >= 18:
            return f"Adult user: {name} ({email})"

        case {
            "type": "user",
            "name": str(name),
            "age": int(age)
        } if age < 18:
            return f"Minor user: {name}"

        case {
            "type": "post",
            "title": str(title),
            "author": str(author),
            "tags": list(tags)
        }:
            return f"Post '{title}' by {author}, tags: {tags}"

        case {
            "type": "comment",
            "text": str(text),
            "replies": list(replies)
        }:
            return f"Comment with {len(replies)} replies: '{text}'"

        case {"type": type_name, **rest}:
            return f"Unknown type '{type_name}' with data: {rest}"

        case _:
            return "Invalid JSON structure"


user1 = {"type": "user", "name": "Alice", "email": "alice@example.com", "age": 25}
user2 = {"type": "user", "name": "Bob", "age": 16}
post = {"type": "post", "title": "Python Tips", "author": "Charlie", "tags": ["python", "programming"]}
comment = {"type": "comment", "text": "Great post!", "replies": []}

print(process_json(user1))
print(process_json(user2))
print(process_json(post))
print(process_json(comment))


# =============================================================================
# State Machine Example
# =============================================================================

section("State Machine Example")


def transition(state: str, event: tuple) -> tuple[str, str]:
    """Simple state machine using pattern matching."""
    match (state, event):
        case ("idle", ("start",)):
            return ("running", "Started")

        case ("running", ("pause",)):
            return ("paused", "Paused")

        case ("paused", ("resume",)):
            return ("running", "Resumed")

        case ("running", ("stop",)):
            return ("idle", "Stopped")

        case ("paused", ("stop",)):
            return ("idle", "Stopped from pause")

        case (current_state, (event_name,)):
            return (current_state, f"Invalid event '{event_name}' in state '{current_state}'")


state = "idle"
events = [("start",), ("pause",), ("resume",), ("stop",)]

print(f"Initial state: {state}")
for event in events:
    state, message = transition(state, event)
    print(f"  Event {event[0]:6} -> state: {state:8} ({message})")


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Pattern matching features (Python 3.10+):
1. match/case - structural pattern matching
2. Literal patterns - exact value matching
3. Capture patterns - bind matched values to variables
4. Wildcard (_) - match anything, don't capture
5. OR patterns (|) - match multiple alternatives
6. Guards (if) - add conditions to patterns
7. Sequence patterns - match lists/tuples
8. Mapping patterns - match dictionaries
9. Class patterns - match dataclass/class instances
10. AS patterns - capture while matching
11. Type patterns - match by type
12. Nested patterns - combine patterns

Benefits:
- More readable than if/elif chains
- Exhaustiveness checking (with mypy)
- Destructuring built-in
- Declarative style
- Better for complex matching scenarios

Use cases:
- Command parsing
- Event handling
- JSON/API response processing
- State machines
- Expression evaluation
- Data validation

Pattern matching vs if/elif:
- Patterns: Better for structural matching, destructuring
- if/elif: Better for simple conditions, complex logic

Note: Pattern matching requires Python 3.10+
""")
