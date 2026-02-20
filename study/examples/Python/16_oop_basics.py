"""
Object-Oriented Programming Basics

Demonstrates:
- Classes and instances
- Instance and class attributes
- Instance and class methods
- Static methods
- Inheritance and super()
- Method overriding
- @property decorator
- Dunder (magic) methods
- Composition vs inheritance
- Abstract base classes
"""

from abc import ABC, abstractmethod
from typing import List


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Basic Class
# =============================================================================

section("Basic Class")


class Person:
    """Simple person class."""

    def __init__(self, name: str, age: int):
        """Initialize person."""
        self.name = name
        self.age = age

    def greet(self) -> str:
        """Greet method."""
        return f"Hello, I'm {self.name}"

    def have_birthday(self):
        """Increment age."""
        self.age += 1


person = Person("Alice", 30)
print(f"Created: {person.name}, age {person.age}")
print(f"Greeting: {person.greet()}")

person.have_birthday()
print(f"After birthday: age {person.age}")


# =============================================================================
# Class and Instance Attributes
# =============================================================================

section("Class and Instance Attributes")


class Dog:
    """Dog class with class and instance attributes."""

    # Class attribute (shared by all instances)
    species = "Canis familiaris"
    count = 0

    def __init__(self, name: str, breed: str):
        """Initialize dog."""
        # Instance attributes (unique to each instance)
        self.name = name
        self.breed = breed
        Dog.count += 1


dog1 = Dog("Buddy", "Golden Retriever")
dog2 = Dog("Max", "Beagle")

print(f"dog1: {dog1.name}, {dog1.breed}, {dog1.species}")
print(f"dog2: {dog2.name}, {dog2.breed}, {dog2.species}")
print(f"Total dogs: {Dog.count}")

# Modifying class attribute
Dog.species = "Canis lupus familiaris"
print(f"Updated species: {dog1.species}, {dog2.species}")


# =============================================================================
# Class Methods and Static Methods
# =============================================================================

section("Class Methods and Static Methods")


class Circle:
    """Circle class with various method types."""

    pi = 3.14159

    def __init__(self, radius: float):
        """Initialize circle."""
        self.radius = radius

    def area(self) -> float:
        """Instance method - uses instance data."""
        return self.pi * self.radius ** 2

    @classmethod
    def from_diameter(cls, diameter: float):
        """Class method - alternative constructor."""
        return cls(diameter / 2)

    @staticmethod
    def is_valid_radius(radius: float) -> bool:
        """Static method - utility function."""
        return radius > 0


# Instance method
circle = Circle(5)
print(f"Circle(5) area: {circle.area()}")

# Class method
circle2 = Circle.from_diameter(10)
print(f"Circle.from_diameter(10) radius: {circle2.radius}")

# Static method
print(f"is_valid_radius(5): {Circle.is_valid_radius(5)}")
print(f"is_valid_radius(-1): {Circle.is_valid_radius(-1)}")


# =============================================================================
# Inheritance
# =============================================================================

section("Inheritance")


class Animal:
    """Base animal class."""

    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        """Make sound."""
        return "Some sound"


class Cat(Animal):
    """Cat inherits from Animal."""

    def speak(self) -> str:
        """Override speak method."""
        return "Meow!"


class Dog(Animal):
    """Dog inherits from Animal."""

    def __init__(self, name: str, breed: str):
        """Initialize with additional attribute."""
        super().__init__(name)  # Call parent constructor
        self.breed = breed

    def speak(self) -> str:
        """Override speak method."""
        return "Woof!"


cat = Cat("Whiskers")
dog = Dog("Buddy", "Golden Retriever")

print(f"{cat.name} says: {cat.speak()}")
print(f"{dog.name} ({dog.breed}) says: {dog.speak()}")


# =============================================================================
# Method Resolution Order (MRO)
# =============================================================================

section("Method Resolution Order (MRO)")


class A:
    def method(self):
        return "A"


class B(A):
    def method(self):
        return "B"


class C(A):
    def method(self):
        return "C"


class D(B, C):
    pass


d = D()
print(f"d.method(): {d.method()}")
print(f"MRO: {[cls.__name__ for cls in D.__mro__]}")


# =============================================================================
# Property Decorator
# =============================================================================

section("Property Decorator")


class Temperature:
    """Temperature with getter/setter."""

    def __init__(self, celsius: float):
        self._celsius = celsius

    @property
    def celsius(self) -> float:
        """Get temperature in Celsius."""
        return self._celsius

    @celsius.setter
    def celsius(self, value: float):
        """Set temperature in Celsius."""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        """Get temperature in Fahrenheit."""
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value: float):
        """Set temperature in Fahrenheit."""
        self.celsius = (value - 32) * 5/9


temp = Temperature(25)
print(f"Celsius: {temp.celsius}°C")
print(f"Fahrenheit: {temp.fahrenheit}°F")

temp.fahrenheit = 100
print(f"After setting to 100°F:")
print(f"  Celsius: {temp.celsius}°C")


# =============================================================================
# Dunder (Magic) Methods
# =============================================================================

section("Dunder (Magic) Methods")


class Vector:
    """2D vector with operator overloading."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        """String representation."""
        return f"Vector({self.x}, {self.y})"

    def __str__(self) -> str:
        """User-friendly string."""
        return f"<{self.x}, {self.y}>"

    def __add__(self, other):
        """Vector addition."""
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """Vector subtraction."""
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float):
        """Scalar multiplication."""
        return Vector(self.x * scalar, self.y * scalar)

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        return self.x == other.x and self.y == other.y

    def __len__(self) -> int:
        """Length (magnitude)."""
        return int((self.x**2 + self.y**2) ** 0.5)


v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(f"v1: {v1}")
print(f"v2: {v2}")
print(f"v1 + v2: {v1 + v2}")
print(f"v1 - v2: {v1 - v2}")
print(f"v1 * 2: {v1 * 2}")
print(f"v1 == v2: {v1 == v2}")
print(f"len(v1): {len(v1)}")


# =============================================================================
# Context Manager Protocol
# =============================================================================

section("Context Manager Protocol")


class FileManager:
    """File manager implementing context protocol."""

    def __init__(self, filename: str, mode: str):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        """Open file."""
        print(f"Opening {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file."""
        if self.file:
            print(f"Closing {self.filename}")
            self.file.close()
        return False


import tempfile
import os

temp_file = os.path.join(tempfile.gettempdir(), "oop_demo.txt")

with FileManager(temp_file, 'w') as f:
    f.write("Hello from context manager!\n")

os.remove(temp_file)


# =============================================================================
# Composition vs Inheritance
# =============================================================================

section("Composition vs Inheritance")

# Inheritance approach


class Vehicle:
    """Vehicle base class."""

    def __init__(self, brand: str):
        self.brand = brand

    def start(self):
        return f"{self.brand} starting..."


class Car(Vehicle):
    """Car inherits from Vehicle."""

    def drive(self):
        return f"{self.brand} driving"


# Composition approach


class Engine:
    """Engine component."""

    def __init__(self, horsepower: int):
        self.horsepower = horsepower

    def start(self):
        return f"Engine ({self.horsepower}hp) starting..."


class CarComposition:
    """Car with engine composition."""

    def __init__(self, brand: str, horsepower: int):
        self.brand = brand
        self.engine = Engine(horsepower)  # Composition

    def start(self):
        return self.engine.start()

    def drive(self):
        return f"{self.brand} driving"


car1 = Car("Toyota")
print(f"Inheritance: {car1.start()}")

car2 = CarComposition("Honda", 200)
print(f"Composition: {car2.start()}")

print("\nComposition is often preferred:")
print("  - More flexible")
print("  - Avoids fragile base class problem")
print("  - 'Has-a' relationship clearer than 'Is-a'")


# =============================================================================
# Abstract Base Classes
# =============================================================================

section("Abstract Base Classes")


class Shape(ABC):
    """Abstract shape base class."""

    @abstractmethod
    def area(self) -> float:
        """Calculate area (must be implemented by subclasses)."""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """Calculate perimeter (must be implemented by subclasses)."""
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


class Circle(Shape):
    """Concrete circle implementation."""

    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * 3.14159 * self.radius


rect = Rectangle(5, 3)
circ = Circle(4)

print(f"Rectangle(5, 3):")
print(f"  Area: {rect.area()}")
print(f"  Perimeter: {rect.perimeter()}")

print(f"\nCircle(4):")
print(f"  Area: {circ.area():.2f}")
print(f"  Perimeter: {circ.perimeter():.2f}")

# Cannot instantiate abstract class
try:
    shape = Shape()
except TypeError as e:
    print(f"\nCannot instantiate ABC: {e}")


# =============================================================================
# Real-World Example: Bank Account
# =============================================================================

section("Real-World Example: Bank Account")


class BankAccount:
    """Bank account with encapsulation."""

    _account_count = 0

    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner
        self._balance = balance  # Private attribute
        self._transactions: List[str] = []
        BankAccount._account_count += 1
        self._account_number = BankAccount._account_count

    @property
    def balance(self) -> float:
        """Get balance (read-only from outside)."""
        return self._balance

    def deposit(self, amount: float):
        """Deposit money."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount
        self._transactions.append(f"Deposit: +${amount:.2f}")

    def withdraw(self, amount: float):
        """Withdraw money."""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
        self._transactions.append(f"Withdrawal: -${amount:.2f}")

    def get_transaction_history(self) -> List[str]:
        """Get transaction history."""
        return self._transactions.copy()

    def __str__(self) -> str:
        return f"Account #{self._account_number}: {self.owner}, Balance: ${self._balance:.2f}"


account = BankAccount("Alice", 1000)
print(account)

account.deposit(500)
account.withdraw(200)
print(f"\nAfter transactions: {account}")

print("\nTransaction history:")
for transaction in account.get_transaction_history():
    print(f"  {transaction}")


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Object-Oriented Programming concepts:
1. Classes and objects - blueprints and instances
2. Attributes - instance and class variables
3. Methods - instance, class, and static
4. Inheritance - reuse and extend functionality
5. super() - call parent class methods
6. @property - getters and setters
7. Dunder methods - operator overloading, protocols
8. Composition - 'has-a' relationships
9. Abstract base classes - enforce interface contracts

Key principles:
- Encapsulation - hide internal details
- Inheritance - is-a relationship
- Polymorphism - same interface, different implementations
- Abstraction - hide complexity

Common dunder methods:
- __init__ - constructor
- __repr__ - developer representation
- __str__ - user-friendly representation
- __eq__, __lt__, __gt__ - comparisons
- __add__, __sub__, __mul__ - arithmetic
- __len__ - length
- __getitem__ - indexing
- __enter__, __exit__ - context manager

Best practices:
- Favor composition over inheritance
- Use @property for computed attributes
- Keep classes focused (Single Responsibility)
- Make attributes private with _ prefix
- Use abstract base classes for interfaces
- Follow naming conventions (PascalCase for classes)
""")
