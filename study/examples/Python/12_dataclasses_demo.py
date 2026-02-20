"""
Python Dataclasses

Demonstrates:
- @dataclass decorator
- field() with default values and factories
- Post-initialization processing
- Frozen dataclasses (immutable)
- Inheritance with dataclasses
- Comparison with NamedTuple and attrs
- asdict() and astuple()
- __post_init__ hook
"""

from dataclasses import dataclass, field, asdict, astuple, FrozenInstanceError
from typing import List, Optional, ClassVar
from collections import namedtuple


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Basic Dataclass
# =============================================================================

section("Basic Dataclass")


@dataclass
class Point:
    """Simple 2D point."""
    x: float
    y: float


p1 = Point(10, 20)
p2 = Point(10, 20)
p3 = Point(30, 40)

print(f"p1: {p1}")
print(f"p1.x = {p1.x}, p1.y = {p1.y}")

# Automatic __eq__
print(f"\np1 == p2: {p1 == p2}")
print(f"p1 == p3: {p1 == p3}")

# Automatic __repr__
print(f"repr(p1): {repr(p1)}")


# =============================================================================
# Default Values
# =============================================================================

section("Default Values")


@dataclass
class Rectangle:
    """Rectangle with default dimensions."""
    width: float = 10.0
    height: float = 5.0
    color: str = "blue"


rect1 = Rectangle()
rect2 = Rectangle(width=20)
rect3 = Rectangle(30, 15, "red")

print(f"rect1: {rect1}")
print(f"rect2: {rect2}")
print(f"rect3: {rect3}")


# =============================================================================
# field() with default_factory
# =============================================================================

section("field() with default_factory")


@dataclass
class TodoList:
    """Todo list with mutable default."""
    owner: str
    items: List[str] = field(default_factory=list)  # Correct way for mutable defaults
    tags: List[str] = field(default_factory=lambda: ["general"])


todo1 = TodoList("Alice")
todo2 = TodoList("Bob")

todo1.items.append("Buy milk")
todo2.items.append("Call dentist")

print(f"todo1: {todo1}")
print(f"todo2: {todo2}")
print("Lists are separate (not shared)")


# =============================================================================
# field() Options
# =============================================================================

section("field() Options")


@dataclass
class Product:
    """Product with various field options."""
    name: str
    price: float
    quantity: int = 0

    # Excluded from __repr__
    _internal_id: str = field(default="", repr=False)

    # Excluded from __init__
    total_value: float = field(init=False, repr=True)

    # Excluded from comparison
    last_updated: str = field(default="", compare=False)

    # Class variable (not instance field)
    category: ClassVar[str] = "General"

    def __post_init__(self):
        """Calculate derived fields."""
        self.total_value = self.price * self.quantity


prod = Product("Widget", 19.99, 10, _internal_id="INTERNAL-123")
print(f"Product: {prod}")
print(f"total_value: {prod.total_value}")
print(f"Category (class var): {Product.category}")


# =============================================================================
# __post_init__ Hook
# =============================================================================

section("__post_init__ Hook")


@dataclass
class User:
    """User with validation and derived fields."""
    username: str
    email: str
    age: int
    full_name: str = field(init=False)

    def __post_init__(self):
        """Validate and initialize derived fields."""
        # Validation
        if self.age < 0:
            raise ValueError("Age cannot be negative")
        if "@" not in self.email:
            raise ValueError("Invalid email")

        # Derived field
        self.full_name = f"User: {self.username}"


user = User("alice", "alice@example.com", 30)
print(f"User: {user}")
print(f"full_name: {user.full_name}")

try:
    invalid_user = User("bob", "invalid-email", 25)
except ValueError as e:
    print(f"\nValidation error: {e}")


# =============================================================================
# Frozen Dataclasses (Immutable)
# =============================================================================

section("Frozen Dataclasses (Immutable)")


@dataclass(frozen=True)
class ImmutablePoint:
    """Immutable point - cannot modify after creation."""
    x: float
    y: float

    def move(self, dx: float, dy: float) -> 'ImmutablePoint':
        """Return new point with offset."""
        return ImmutablePoint(self.x + dx, self.y + dy)


p = ImmutablePoint(10, 20)
print(f"Original: {p}")

p_moved = p.move(5, 3)
print(f"Moved: {p_moved}")
print(f"Original unchanged: {p}")

try:
    p.x = 100  # This will fail
except FrozenInstanceError as e:
    print(f"\nCannot modify frozen instance: {e}")


# =============================================================================
# Comparison and Ordering
# =============================================================================

section("Comparison and Ordering")


@dataclass(order=True)
class Version:
    """Version with automatic ordering."""
    major: int
    minor: int
    patch: int


v1 = Version(1, 0, 0)
v2 = Version(1, 5, 2)
v3 = Version(2, 0, 0)

print(f"v1: {v1}")
print(f"v2: {v2}")
print(f"v3: {v3}")
print(f"\nv1 < v2: {v1 < v2}")
print(f"v2 < v3: {v2 < v3}")
print(f"sorted([v3, v1, v2]): {sorted([v3, v1, v2])}")


# =============================================================================
# Custom Ordering
# =============================================================================

section("Custom Ordering")


@dataclass(order=True)
class Person:
    """Person with custom sort order."""
    sort_index: int = field(init=False, repr=False)
    name: str
    age: int
    salary: float

    def __post_init__(self):
        # Sort by salary (primary), then age (secondary)
        self.sort_index = self.age  # Can customize as needed


people = [
    Person("Alice", 30, 75000),
    Person("Bob", 25, 60000),
    Person("Charlie", 35, 90000),
]

print("Original:")
for p in people:
    print(f"  {p.name}: age={p.age}, salary={p.salary}")

print("\nSorted by age (via sort_index):")
for p in sorted(people):
    print(f"  {p.name}: age={p.age}, salary={p.salary}")


# =============================================================================
# Inheritance
# =============================================================================

section("Inheritance with Dataclasses")


@dataclass
class Animal:
    """Base animal class."""
    name: str
    age: int


@dataclass
class Dog(Animal):
    """Dog extends Animal."""
    breed: str
    is_good_boy: bool = True


dog = Dog("Buddy", 5, "Golden Retriever")
print(f"Dog: {dog}")
print(f"Is good boy? {dog.is_good_boy}")


# =============================================================================
# asdict() and astuple()
# =============================================================================

section("asdict() and astuple()")


@dataclass
class Book:
    """Book with author information."""
    title: str
    author: str
    year: int
    pages: int


book = Book("Python Tricks", "Dan Bader", 2017, 301)

# Convert to dict
book_dict = asdict(book)
print(f"asdict(): {book_dict}")
print(f"Type: {type(book_dict)}")

# Convert to tuple
book_tuple = astuple(book)
print(f"\nastuple(): {book_tuple}")
print(f"Type: {type(book_tuple)}")


# =============================================================================
# Nested Dataclasses
# =============================================================================

section("Nested Dataclasses")


@dataclass
class Address:
    """Address information."""
    street: str
    city: str
    zip_code: str


@dataclass
class Company:
    """Company with address."""
    name: str
    address: Address
    employees: int


address = Address("123 Main St", "Springfield", "12345")
company = Company("Acme Inc", address, 50)

print(f"Company: {company}")

# asdict with nested dataclasses
company_dict = asdict(company)
print(f"\nasdict() (nested): {company_dict}")


# =============================================================================
# Comparison with NamedTuple
# =============================================================================

section("Comparison with NamedTuple")

# NamedTuple
PersonTuple = namedtuple('PersonTuple', ['name', 'age'])


@dataclass
class PersonDataclass:
    """Person as dataclass."""
    name: str
    age: int


pt = PersonTuple("Alice", 30)
pd = PersonDataclass("Alice", 30)

print("NamedTuple:")
print(f"  Creation: {pt}")
print(f"  Immutable: {True}")
print(f"  Default values: Limited (via defaults)")
print(f"  Methods: Can't add methods to instance")

print("\nDataclass:")
print(f"  Creation: {pd}")
print(f"  Immutable: {False} (unless frozen=True)")
print(f"  Default values: Full support with field()")
print(f"  Methods: Can add methods")


# =============================================================================
# Match Pattern (Python 3.10+)
# =============================================================================

section("Pattern Matching with Dataclasses")


@dataclass
class Circle:
    """Circle shape."""
    radius: float


@dataclass
class Rectangle:
    """Rectangle shape."""
    width: float
    height: float


def area(shape):
    """Calculate area using pattern matching."""
    match shape:
        case Circle(radius=r):
            return 3.14159 * r * r
        case Rectangle(width=w, height=h):
            return w * h
        case _:
            return 0


circle = Circle(5)
rect = Rectangle(4, 6)

print(f"Circle area: {area(circle)}")
print(f"Rectangle area: {area(rect)}")


# =============================================================================
# Comparison with attrs (Conceptual)
# =============================================================================

section("Dataclasses vs attrs")

print("""
Dataclasses (stdlib, Python 3.7+):
  @dataclass
  class Point:
      x: int
      y: int

  Pros:
  - Built into Python 3.7+
  - No external dependencies
  - Standard library support
  - Good IDE integration

  Cons:
  - Less feature-rich than attrs
  - No validators (need __post_init__)
  - No converters

attrs (third-party, more features):
  @attrs.define
  class Point:
      x: int
      y: int

  Pros:
  - More features (validators, converters)
  - Works with Python 2.7+
  - More mature
  - Slots support

  Cons:
  - External dependency
  - Additional package to maintain

When to use:
- Dataclasses: Default choice for Python 3.7+
- attrs: Need advanced features or Python 2.7 support
- NamedTuple: Need immutable, tuple-like behavior
- Regular class: Need complex custom behavior
""")


# =============================================================================
# Real-World Example
# =============================================================================

section("Real-World Example")


@dataclass
class BlogPost:
    """Blog post with metadata."""
    title: str
    content: str
    author: str
    tags: List[str] = field(default_factory=list)
    published: bool = False
    views: int = 0
    slug: str = field(init=False)

    def __post_init__(self):
        """Generate slug from title."""
        self.slug = self.title.lower().replace(" ", "-")

    def publish(self):
        """Mark post as published."""
        self.published = True

    def increment_views(self):
        """Increment view count."""
        self.views += 1

    def summary(self) -> str:
        """Get post summary."""
        return f"{self.title} by {self.author} ({self.views} views)"


post = BlogPost(
    title="Python Dataclasses Guide",
    content="Lorem ipsum...",
    author="Alice",
    tags=["python", "tutorial"]
)

print(f"Post: {post}")
print(f"Slug: {post.slug}")

post.publish()
post.increment_views()
post.increment_views()

print(f"Summary: {post.summary()}")
print(f"Published: {post.published}")


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Dataclass features:
1. @dataclass decorator - auto-generate methods
2. field() - customize field behavior
3. default_factory - mutable defaults
4. __post_init__ - post-initialization processing
5. frozen=True - immutable instances
6. order=True - comparison operators
7. asdict()/astuple() - conversion utilities
8. Inheritance - works naturally
9. Pattern matching - structural pattern matching support

Generated methods:
- __init__ - initialization
- __repr__ - string representation
- __eq__ - equality comparison
- __lt__, __le__, __gt__, __ge__ - ordering (if order=True)
- __hash__ - hashing (if frozen=True)

Use dataclasses when:
- Need simple data containers
- Want automatic __init__, __repr__, __eq__
- Need type hints
- Want clean, readable code
- Python 3.7+ is available

Avoid when:
- Need complex custom initialization
- Require validation beyond __post_init__
- Need converters/validators (use attrs instead)
""")
