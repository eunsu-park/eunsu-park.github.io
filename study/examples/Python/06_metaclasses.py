"""
Python Metaclasses

Demonstrates:
- type() - the metaclass of all classes
- Custom metaclasses
- __init_subclass__ hook
- Class creation process
- Practical examples (registry, singleton, validation)
- ABCMeta
"""

from typing import Any, Dict
from abc import ABCMeta, abstractmethod


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Understanding type()
# =============================================================================

section("Understanding type()")

# type() has two uses:
# 1. Return the type of an object
# 2. Create a new class dynamically

print("type() as type checker:")
print(f"  type(42) = {type(42)}")
print(f"  type('hello') = {type('hello')}")
print(f"  type([]) = {type([])}")


# Create class with type()
print("\nCreating class with type():")


def init_method(self, value):
    self.value = value


def display_method(self):
    return f"MyClass(value={self.value})"


# type(name, bases, dict)
MyClass = type('MyClass', (object,), {
    '__init__': init_method,
    'display': display_method,
    'class_var': 42
})

obj = MyClass(100)
print(f"  Created: {obj.display()}")
print(f"  Class var: {MyClass.class_var}")
print(f"  type(MyClass) = {type(MyClass)}")


# =============================================================================
# Basic Metaclass
# =============================================================================

section("Basic Metaclass")


class SimpleMeta(type):
    """Simple metaclass that prints when class is created."""

    def __new__(mcs, name, bases, namespace, **kwargs):
        print(f"  Creating class '{name}' with SimpleMeta")
        print(f"    Bases: {bases}")
        print(f"    Namespace keys: {list(namespace.keys())}")
        cls = super().__new__(mcs, name, bases, namespace)
        return cls


class MyClassWithMeta(metaclass=SimpleMeta):
    """Class using SimpleMeta."""

    class_attr = "I'm a class attribute"

    def instance_method(self):
        return "I'm an instance method"


print("\nInstantiating MyClassWithMeta:")
instance = MyClassWithMeta()
print(f"  Instance created: {instance}")


# =============================================================================
# Metaclass with __init__
# =============================================================================

section("Metaclass __init__")


class InitMeta(type):
    """Metaclass with __init__ to modify class after creation."""

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        # Add attribute to class after creation
        cls.metaclass_added = f"Added by InitMeta to {name}"
        print(f"  InitMeta.__init__ called for {name}")


class MyInitClass(metaclass=InitMeta):
    """Class using InitMeta."""
    pass


print(f"MyInitClass.metaclass_added = {MyInitClass.metaclass_added}")


# =============================================================================
# Registry Pattern
# =============================================================================

section("Registry Pattern with Metaclass")


class RegistryMeta(type):
    """Metaclass that maintains a registry of all subclasses."""

    registry: Dict[str, type] = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)

        # Don't register the base class itself
        if bases:
            mcs.registry[name] = cls
            print(f"  Registered: {name}")

        return cls

    @classmethod
    def get_registry(mcs):
        return mcs.registry.copy()


class Plugin(metaclass=RegistryMeta):
    """Base plugin class."""
    pass


class PluginA(Plugin):
    """First plugin."""
    def run(self):
        return "PluginA running"


class PluginB(Plugin):
    """Second plugin."""
    def run(self):
        return "PluginB running"


class PluginC(Plugin):
    """Third plugin."""
    def run(self):
        return "PluginC running"


print("\nRegistry contents:")
for name, cls in RegistryMeta.get_registry().items():
    print(f"  {name}: {cls}")

print("\nInstantiating plugins from registry:")
for name, cls in RegistryMeta.get_registry().items():
    plugin = cls()
    print(f"  {name}: {plugin.run()}")


# =============================================================================
# Singleton Pattern
# =============================================================================

section("Singleton Pattern with Metaclass")


class SingletonMeta(type):
    """Metaclass that implements singleton pattern."""

    _instances: Dict[type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            print(f"  Creating new instance of {cls.__name__}")
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        else:
            print(f"  Returning existing instance of {cls.__name__}")

        return cls._instances[cls]


class Database(metaclass=SingletonMeta):
    """Singleton database connection."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        print(f"  Database initialized with: {connection_string}")


db1 = Database("postgresql://localhost/db1")
db2 = Database("postgresql://localhost/db2")  # Same instance!

print(f"\ndb1 is db2: {db1 is db2}")
print(f"db1.connection_string: {db1.connection_string}")


# =============================================================================
# Attribute Validation
# =============================================================================

section("Attribute Validation with Metaclass")


class ValidatedMeta(type):
    """Metaclass that validates class attributes."""

    def __new__(mcs, name, bases, namespace):
        # Check that required_fields are present
        if 'required_fields' in namespace:
            required = namespace['required_fields']
            for field in required:
                if field not in namespace:
                    raise TypeError(
                        f"Class {name} missing required field: {field}"
                    )

        return super().__new__(mcs, name, bases, namespace)


class ValidModel(metaclass=ValidatedMeta):
    """Base model with validation."""
    required_fields = ['name', 'version']

    name = "ValidModel"
    version = "1.0"


print("ValidModel created successfully")
print(f"  name: {ValidModel.name}")
print(f"  version: {ValidModel.version}")


# Try creating invalid model
try:
    class InvalidModel(metaclass=ValidatedMeta):
        required_fields = ['name', 'version']
        name = "InvalidModel"
        # Missing 'version' field
except TypeError as e:
    print(f"\nInvalidModel creation failed: {e}")


# =============================================================================
# __init_subclass__ Hook (Alternative to Metaclass)
# =============================================================================

section("__init_subclass__ Hook")


class PluginBase:
    """Base class using __init_subclass__ instead of metaclass."""

    plugins = {}

    def __init_subclass__(cls, plugin_name: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if plugin_name:
            cls.plugins[plugin_name] = cls
            print(f"  Registered plugin: {plugin_name} -> {cls.__name__}")


class ImagePlugin(PluginBase, plugin_name="image"):
    """Image processing plugin."""
    def process(self):
        return "Processing image"


class VideoPlugin(PluginBase, plugin_name="video"):
    """Video processing plugin."""
    def process(self):
        return "Processing video"


print("\nPlugins registered via __init_subclass__:")
for name, cls in PluginBase.plugins.items():
    print(f"  {name}: {cls.__name__}")


# =============================================================================
# ABCMeta - Abstract Base Classes
# =============================================================================

section("ABCMeta - Abstract Base Classes")


class Shape(metaclass=ABCMeta):
    """Abstract shape class."""

    @abstractmethod
    def area(self) -> float:
        """Calculate area."""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """Calculate perimeter."""
        pass


class Rectangle(Shape):
    """Concrete rectangle implementation."""

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


rect = Rectangle(5, 3)
print(f"Rectangle(5, 3):")
print(f"  Area: {rect.area()}")
print(f"  Perimeter: {rect.perimeter()}")


# Try to instantiate abstract class
try:
    shape = Shape()
except TypeError as e:
    print(f"\nCannot instantiate abstract class:")
    print(f"  {e}")


# =============================================================================
# Metaclass Inheritance
# =============================================================================

section("Metaclass Inheritance")


class MetaA(type):
    """First metaclass."""
    def __new__(mcs, name, bases, namespace):
        print(f"  MetaA processing {name}")
        namespace['from_meta_a'] = True
        return super().__new__(mcs, name, bases, namespace)


class MetaB(MetaA):
    """Metaclass inheriting from MetaA."""
    def __new__(mcs, name, bases, namespace):
        print(f"  MetaB processing {name}")
        namespace['from_meta_b'] = True
        return super().__new__(mcs, name, bases, namespace)


class MyDerivedClass(metaclass=MetaB):
    """Class using derived metaclass."""
    pass


print(f"\nMyDerivedClass.from_meta_a: {MyDerivedClass.from_meta_a}")
print(f"MyDerivedClass.from_meta_b: {MyDerivedClass.from_meta_b}")


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Metaclass patterns covered:
1. type() - the metaclass of all classes
2. Custom metaclasses - control class creation
3. Registry pattern - auto-register subclasses
4. Singleton pattern - single instance enforcement
5. Validation - ensure class contracts
6. __init_subclass__ - simpler alternative to metaclasses
7. ABCMeta - abstract base classes
8. Metaclass inheritance - compose metaclass behavior

When to use metaclasses:
- Framework/library development
- Enforcing API contracts
- Auto-registration patterns
- Domain-specific languages
- ORM implementations

Alternatives to consider:
- Class decorators (simpler)
- __init_subclass__ (for registration)
- Descriptors (for attribute control)

"Metaclasses are deeper magic than 99% of users should ever
worry about." - Tim Peters
""")
