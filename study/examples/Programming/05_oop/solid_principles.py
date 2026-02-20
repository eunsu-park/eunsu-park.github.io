"""
SOLID Principles Demonstration

SOLID is an acronym for five design principles that make software more:
- Understandable
- Flexible
- Maintainable

S - Single Responsibility Principle
O - Open/Closed Principle
L - Liskov Substitution Principle
I - Interface Segregation Principle
D - Dependency Inversion Principle
"""

from abc import ABC, abstractmethod
from typing import List, Protocol
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# S - SINGLE RESPONSIBILITY PRINCIPLE (SRP)
# A class should have only ONE reason to change
# =============================================================================

print("=" * 70)
print("S - SINGLE RESPONSIBILITY PRINCIPLE")
print("=" * 70)


# ❌ BEFORE: God class with multiple responsibilities
class UserManagerBad:
    """This class violates SRP - it does TOO MUCH"""

    def __init__(self, username: str, email: str):
        self.username = username
        self.email = email

    def save_to_database(self):
        """Responsibility 1: Database operations"""
        print(f"Saving {self.username} to database...")

    def send_welcome_email(self):
        """Responsibility 2: Email sending"""
        print(f"Sending welcome email to {self.email}...")

    def generate_report(self) -> str:
        """Responsibility 3: Report generation"""
        return f"User Report: {self.username}"

    # Problem: If email system changes, database logic changes,
    # or reporting format changes, we modify the SAME class!


# ✅ AFTER: Each class has a single responsibility
@dataclass
class User:
    """Only responsible for user data"""
    username: str
    email: str


class UserRepository:
    """Single responsibility: Database operations"""

    def save(self, user: User):
        print(f"Saving {user.username} to database...")


class EmailService:
    """Single responsibility: Email sending"""

    def send_welcome_email(self, user: User):
        print(f"Sending welcome email to {user.email}...")


class ReportGenerator:
    """Single responsibility: Report generation"""

    def generate_user_report(self, user: User) -> str:
        return f"User Report: {user.username}"


# =============================================================================
# O - OPEN/CLOSED PRINCIPLE (OCP)
# Open for extension, closed for modification
# =============================================================================

print("\n" + "=" * 70)
print("O - OPEN/CLOSED PRINCIPLE")
print("=" * 70)


# ❌ BEFORE: Must modify class to add new shapes
class AreaCalculatorBad:
    """Violates OCP - must modify to add new shapes"""

    def calculate_area(self, shape_type: str, **kwargs) -> float:
        if shape_type == "circle":
            return 3.14159 * kwargs["radius"] ** 2
        elif shape_type == "rectangle":
            return kwargs["width"] * kwargs["height"]
        elif shape_type == "triangle":
            return 0.5 * kwargs["base"] * kwargs["height"]
        # Adding a new shape requires MODIFYING this method!
        else:
            raise ValueError(f"Unknown shape: {shape_type}")


# ✅ AFTER: Extensible through inheritance/protocols
class Shape(ABC):
    """Abstract base - closed for modification"""

    @abstractmethod
    def area(self) -> float:
        pass


class Circle(Shape):
    """Extends Shape without modifying it"""

    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2


class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height


class Triangle(Shape):
    def __init__(self, base: float, height: float):
        self.base = base
        self.height = height

    def area(self) -> float:
        return 0.5 * self.base * self.height


class AreaCalculator:
    """Closed for modification - works with any Shape"""

    def calculate_total_area(self, shapes: List[Shape]) -> float:
        return sum(shape.area() for shape in shapes)


# =============================================================================
# L - LISKOV SUBSTITUTION PRINCIPLE (LSP)
# Subtypes must be substitutable for their base types
# =============================================================================

print("\n" + "=" * 70)
print("L - LISKOV SUBSTITUTION PRINCIPLE")
print("=" * 70)


# ❌ BEFORE: Square violates LSP
class RectangleBad:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def set_width(self, width: float):
        self.width = width

    def set_height(self, height: float):
        self.height = height

    def area(self) -> float:
        return self.width * self.height


class SquareBad(RectangleBad):
    """Violates LSP - changes behavior unexpectedly"""

    def set_width(self, width: float):
        self.width = width
        self.height = width  # Breaks rectangle's assumption!

    def set_height(self, height: float):
        self.height = height
        self.width = height  # Breaks rectangle's assumption!


# ✅ AFTER: Proper abstraction respects LSP
class ShapeLSP(ABC):
    @abstractmethod
    def area(self) -> float:
        pass


class RectangleLSP(ShapeLSP):
    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height

    def area(self) -> float:
        return self._width * self._height


class SquareLSP(ShapeLSP):
    """Independent class - doesn't inherit problematic behavior"""

    def __init__(self, side: float):
        self._side = side

    def area(self) -> float:
        return self._side ** 2


# =============================================================================
# I - INTERFACE SEGREGATION PRINCIPLE (ISP)
# Clients should not depend on interfaces they don't use
# =============================================================================

print("\n" + "=" * 70)
print("I - INTERFACE SEGREGATION PRINCIPLE")
print("=" * 70)


# ❌ BEFORE: Fat interface forces unnecessary implementation
class WorkerBad(ABC):
    """Fat interface - not all workers do everything"""

    @abstractmethod
    def work(self):
        pass

    @abstractmethod
    def eat(self):
        pass

    @abstractmethod
    def sleep(self):
        pass


class HumanWorkerBad(WorkerBad):
    def work(self):
        print("Human working...")

    def eat(self):
        print("Human eating...")

    def sleep(self):
        print("Human sleeping...")


class RobotWorkerBad(WorkerBad):
    def work(self):
        print("Robot working...")

    def eat(self):
        raise NotImplementedError("Robots don't eat!")  # Forced to implement!

    def sleep(self):
        raise NotImplementedError("Robots don't sleep!")  # Forced to implement!


# ✅ AFTER: Segregated interfaces
class Workable(Protocol):
    """Small, focused interface"""

    def work(self):
        pass


class Eatable(Protocol):
    def eat(self):
        pass


class Sleepable(Protocol):
    def sleep(self):
        pass


class HumanWorker:
    """Implements only needed interfaces"""

    def work(self):
        print("Human working...")

    def eat(self):
        print("Human eating...")

    def sleep(self):
        print("Human sleeping...")


class RobotWorker:
    """Only implements what makes sense"""

    def work(self):
        print("Robot working...")


# =============================================================================
# D - DEPENDENCY INVERSION PRINCIPLE (DIP)
# Depend on abstractions, not concretions
# =============================================================================

print("\n" + "=" * 70)
print("D - DEPENDENCY INVERSION PRINCIPLE")
print("=" * 70)


# ❌ BEFORE: High-level module depends on low-level module
class MySQLDatabase:
    """Low-level module (concrete implementation)"""

    def save(self, data: str):
        print(f"Saving to MySQL: {data}")


class UserServiceBad:
    """High-level module depends on concrete MySQL"""

    def __init__(self):
        self.db = MySQLDatabase()  # Tight coupling!

    def save_user(self, username: str):
        self.db.save(username)
        # Can't easily switch to PostgreSQL or MongoDB!


# ✅ AFTER: Both depend on abstraction
class Database(Protocol):
    """Abstraction (interface)"""

    def save(self, data: str):
        pass


class MySQLDatabaseGood:
    def save(self, data: str):
        print(f"Saving to MySQL: {data}")


class PostgreSQLDatabase:
    def save(self, data: str):
        print(f"Saving to PostgreSQL: {data}")


class MongoDBDatabase:
    def save(self, data: str):
        print(f"Saving to MongoDB: {data}")


class UserService:
    """High-level module depends on abstraction"""

    def __init__(self, database: Database):
        self.db = database  # Depends on interface, not concrete class!

    def save_user(self, username: str):
        self.db.save(username)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_solid():
    print("\n" + "=" * 70)
    print("SOLID PRINCIPLES DEMONSTRATION")
    print("=" * 70)

    # Single Responsibility
    print("\n[SRP] Single Responsibility Principle:")
    user = User("alice", "alice@example.com")
    repo = UserRepository()
    email_service = EmailService()
    report_gen = ReportGenerator()

    repo.save(user)
    email_service.send_welcome_email(user)
    print(report_gen.generate_user_report(user))

    # Open/Closed
    print("\n[OCP] Open/Closed Principle:")
    shapes = [
        Circle(5),
        Rectangle(4, 6),
        Triangle(3, 8)
    ]
    calculator = AreaCalculator()
    total = calculator.calculate_total_area(shapes)
    print(f"Total area: {total:.2f}")

    # Liskov Substitution
    print("\n[LSP] Liskov Substitution Principle:")
    shapes_lsp: List[ShapeLSP] = [
        RectangleLSP(4, 6),
        SquareLSP(5)
    ]
    for shape in shapes_lsp:
        print(f"Area: {shape.area()}")

    # Interface Segregation
    print("\n[ISP] Interface Segregation Principle:")
    human = HumanWorker()
    robot = RobotWorker()

    human.work()
    human.eat()
    robot.work()
    # robot.eat()  # Not forced to implement!

    # Dependency Inversion
    print("\n[DIP] Dependency Inversion Principle:")
    mysql_service = UserService(MySQLDatabaseGood())
    postgres_service = UserService(PostgreSQLDatabase())
    mongo_service = UserService(MongoDBDatabase())

    mysql_service.save_user("alice")
    postgres_service.save_user("bob")
    mongo_service.save_user("charlie")


def print_summary():
    print("\n" + "=" * 70)
    print("SOLID PRINCIPLES SUMMARY")
    print("=" * 70)

    print("""
[S] Single Responsibility Principle
    ✓ One class = one reason to change
    ✓ Easier to understand and maintain
    ✓ Better testability

[O] Open/Closed Principle
    ✓ Open for extension (add new features)
    ✓ Closed for modification (don't change existing code)
    ✓ Use abstraction and polymorphism

[L] Liskov Substitution Principle
    ✓ Subtypes must be substitutable for base types
    ✓ Don't violate base class contracts
    ✓ Prefer composition over problematic inheritance

[I] Interface Segregation Principle
    ✓ Many small interfaces > one large interface
    ✓ Clients use only what they need
    ✓ Reduces coupling

[D] Dependency Inversion Principle
    ✓ Depend on abstractions, not concretions
    ✓ High-level modules independent of low-level details
    ✓ Enables easy swapping of implementations

BENEFITS OF SOLID:
  • More maintainable code
  • Easier to test
  • Better extensibility
  • Reduced coupling
  • Improved reusability
""")


if __name__ == "__main__":
    demonstrate_solid()
    print_summary()
