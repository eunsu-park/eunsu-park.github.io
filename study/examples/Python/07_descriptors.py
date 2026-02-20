"""
Python Descriptors

Demonstrates:
- Descriptor protocol (__get__, __set__, __delete__)
- Data vs non-data descriptors
- property() implementation
- Validation descriptors
- Cached properties
- Type checking descriptors
"""

from typing import Any, Optional, Callable
import functools


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Basic Descriptor
# =============================================================================

section("Basic Descriptor Protocol")


class Descriptor:
    """Basic descriptor demonstrating __get__, __set__, __delete__."""

    def __init__(self, name: str):
        self.name = name

    def __get__(self, instance, owner):
        """Called when attribute is accessed."""
        if instance is None:
            # Accessed on class, not instance
            print(f"  __get__ called on class {owner.__name__}")
            return self

        print(f"  __get__: {self.name} from {instance}")
        return instance.__dict__.get(self.name, None)

    def __set__(self, instance, value):
        """Called when attribute is assigned."""
        print(f"  __set__: {self.name} = {value}")
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        """Called when attribute is deleted."""
        print(f"  __delete__: {self.name}")
        del instance.__dict__[self.name]


class MyClass:
    """Class using descriptor."""
    attr = Descriptor("attr")


obj = MyClass()
print("Setting attribute:")
obj.attr = 42

print("\nGetting attribute:")
value = obj.attr
print(f"  Value: {value}")

print("\nDeleting attribute:")
del obj.attr

print("\nAccessing on class:")
MyClass.attr


# =============================================================================
# Data vs Non-Data Descriptors
# =============================================================================

section("Data vs Non-Data Descriptors")


class DataDescriptor:
    """Data descriptor - has __get__ AND __set__."""

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name, "default")

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value


class NonDataDescriptor:
    """Non-data descriptor - only has __get__."""

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name, "default")


class TestClass:
    data_desc = DataDescriptor("data_desc")
    non_data_desc = NonDataDescriptor("non_data_desc")


obj = TestClass()

print("Data descriptor (has priority over instance dict):")
obj.__dict__["data_desc"] = "instance value"
print(f"  obj.data_desc = {obj.data_desc}")  # Still goes through descriptor

print("\nNon-data descriptor (instance dict has priority):")
obj.__dict__["non_data_desc"] = "instance value"
print(f"  obj.non_data_desc = {obj.non_data_desc}")  # Uses instance dict


# =============================================================================
# Validation Descriptor
# =============================================================================

section("Validation Descriptor")


class PositiveNumber:
    """Descriptor that validates positive numbers."""

    def __init__(self, name: str):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} must be a number")
        if value <= 0:
            raise ValueError(f"{self.name} must be positive")
        instance.__dict__[self.name] = value


class Product:
    """Product with validated price and quantity."""

    price = PositiveNumber("price")
    quantity = PositiveNumber("quantity")

    def __init__(self, name: str, price: float, quantity: int):
        self.name = name
        self.price = price
        self.quantity = quantity

    def total_value(self) -> float:
        return self.price * self.quantity


product = Product("Widget", 19.99, 100)
print(f"Product: {product.name}")
print(f"  Price: ${product.price}")
print(f"  Quantity: {product.quantity}")
print(f"  Total value: ${product.total_value()}")

print("\nTrying invalid values:")
try:
    product.price = -10
except ValueError as e:
    print(f"  Error: {e}")

try:
    product.quantity = "not a number"
except TypeError as e:
    print(f"  Error: {e}")


# =============================================================================
# Type Checking Descriptor
# =============================================================================

section("Type Checking Descriptor")


class TypedDescriptor:
    """Descriptor with type checking."""

    def __init__(self, name: str, expected_type: type):
        self.name = name
        self.expected_type = expected_type

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.name} must be {self.expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        instance.__dict__[self.name] = value


class Person:
    """Person with type-checked attributes."""

    name = TypedDescriptor("name", str)
    age = TypedDescriptor("age", int)
    salary = TypedDescriptor("salary", float)

    def __init__(self, name: str, age: int, salary: float):
        self.name = name
        self.age = age
        self.salary = salary


person = Person("Alice", 30, 75000.0)
print(f"Person: {person.name}, {person.age} years old, ${person.salary}")

try:
    person.age = "thirty"
except TypeError as e:
    print(f"\nType error: {e}")


# =============================================================================
# property() - Built-in Descriptor
# =============================================================================

section("property() - Built-in Descriptor")


class Temperature:
    """Temperature with Celsius/Fahrenheit conversion."""

    def __init__(self, celsius: float):
        self._celsius = celsius

    @property
    def celsius(self) -> float:
        """Get temperature in Celsius."""
        print("  Getting celsius")
        return self._celsius

    @celsius.setter
    def celsius(self, value: float):
        """Set temperature in Celsius."""
        print(f"  Setting celsius to {value}")
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        """Get temperature in Fahrenheit."""
        print("  Computing fahrenheit")
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value: float):
        """Set temperature via Fahrenheit."""
        print(f"  Setting fahrenheit to {value}")
        self._celsius = (value - 32) * 5/9


temp = Temperature(25)
print(f"Celsius: {temp.celsius}°C")
print(f"Fahrenheit: {temp.fahrenheit}°F")

print("\nSetting via Fahrenheit:")
temp.fahrenheit = 100
print(f"Celsius: {temp.celsius}°C")


# =============================================================================
# Cached Property
# =============================================================================

section("Cached Property Descriptor")


class CachedProperty:
    """Descriptor that caches computed property value."""

    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__

    def __get__(self, instance, owner):
        if instance is None:
            return self

        # Check if value is cached
        cache_name = f"_cached_{self.name}"
        if not hasattr(instance, cache_name):
            print(f"  Computing {self.name}...")
            value = self.func(instance)
            setattr(instance, cache_name, value)
        else:
            print(f"  Using cached {self.name}")

        return getattr(instance, cache_name)


class DataProcessor:
    """Processor with expensive computations."""

    def __init__(self, data: list):
        self.data = data

    @CachedProperty
    def average(self) -> float:
        """Compute average (expensive operation)."""
        import time
        time.sleep(0.1)  # Simulate expensive computation
        return sum(self.data) / len(self.data)

    @CachedProperty
    def total(self) -> float:
        """Compute total (expensive operation)."""
        import time
        time.sleep(0.1)  # Simulate expensive computation
        return sum(self.data)


processor = DataProcessor([1, 2, 3, 4, 5])

print("First access (computed):")
print(f"  Average: {processor.average}")

print("\nSecond access (cached):")
print(f"  Average: {processor.average}")

print("\nAccessing total:")
print(f"  Total: {processor.total}")


# =============================================================================
# Using functools.cached_property
# =============================================================================

section("functools.cached_property")


class WebPage:
    """Web page with cached properties."""

    def __init__(self, url: str):
        self.url = url

    @functools.cached_property
    def content(self) -> str:
        """Fetch page content (cached)."""
        print(f"  Fetching {self.url}...")
        import time
        time.sleep(0.1)
        return f"Content from {self.url}"


page = WebPage("https://example.com")
print(f"Content (first): {page.content[:30]}...")
print(f"Content (cached): {page.content[:30]}...")


# =============================================================================
# Read-Only Descriptor
# =============================================================================

section("Read-Only Descriptor")


class ReadOnly:
    """Read-only descriptor."""

    def __init__(self, value):
        self.value = value

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        raise AttributeError("Cannot modify read-only attribute")


class Config:
    """Configuration with read-only values."""

    VERSION = ReadOnly("1.0.0")
    MAX_CONNECTIONS = ReadOnly(100)


config = Config()
print(f"Config.VERSION: {config.VERSION}")
print(f"Config.MAX_CONNECTIONS: {config.MAX_CONNECTIONS}")

try:
    config.VERSION = "2.0.0"
except AttributeError as e:
    print(f"\nError: {e}")


# =============================================================================
# Lazy Descriptor
# =============================================================================

section("Lazy Descriptor")


class LazyProperty:
    """Descriptor that lazily initializes value."""

    def __init__(self, init_func: Callable):
        self.init_func = init_func
        self.name = init_func.__name__

    def __get__(self, instance, owner):
        if instance is None:
            return self

        attr_name = f"_lazy_{self.name}"
        if not hasattr(instance, attr_name):
            print(f"  Lazy-initializing {self.name}")
            value = self.init_func(instance)
            setattr(instance, attr_name, value)

        return getattr(instance, attr_name)


class Application:
    """Application with lazy-loaded components."""

    @LazyProperty
    def database(self):
        """Initialize database connection."""
        print("    Connecting to database...")
        return "DatabaseConnection"

    @LazyProperty
    def cache(self):
        """Initialize cache."""
        print("    Connecting to cache...")
        return "CacheConnection"


app = Application()
print("Application created (no connections yet)")

print("\nAccessing database:")
print(f"  {app.database}")

print("\nAccessing database again:")
print(f"  {app.database}")


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Descriptor patterns covered:
1. Descriptor protocol - __get__, __set__, __delete__
2. Data vs non-data descriptors - lookup priority
3. Validation descriptors - enforce constraints
4. Type checking descriptors - runtime type validation
5. property() - built-in descriptor for getters/setters
6. Cached property - compute once, cache result
7. Read-only descriptor - prevent modification
8. Lazy property - initialize on first access

Descriptor use cases:
- Validation and type checking
- Computed properties
- Caching expensive computations
- Lazy initialization
- ORM field definitions
- Method binding

Descriptors are the mechanism behind:
- @property decorator
- @classmethod and @staticmethod
- Functions (binding to methods)
""")
