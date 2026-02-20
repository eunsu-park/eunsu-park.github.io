"""
Composition vs Inheritance

This demonstrates why "Favor Composition over Inheritance" is a key OOP principle.

Problems with deep inheritance hierarchies:
1. Tight coupling
2. Fragile base class problem
3. Diamond problem (multiple inheritance)
4. Difficult to reason about
5. Hard to extend without breaking existing code

Composition provides:
1. Loose coupling
2. Flexibility
3. Better testability
4. Easier to understand
"""

from abc import ABC, abstractmethod
from typing import List, Optional


# =============================================================================
# PROBLEM: Inheritance Hierarchy Gets Out of Control
# =============================================================================

print("=" * 70)
print("THE INHERITANCE PROBLEM")
print("=" * 70)


# ❌ BAD: Deep inheritance hierarchy
class Animal:
    """Base class"""

    def __init__(self, name: str):
        self.name = name

    def eat(self):
        print(f"{self.name} is eating...")


class FlyingAnimal(Animal):
    """Animals that can fly"""

    def fly(self):
        print(f"{self.name} is flying...")


class SwimmingAnimal(Animal):
    """Animals that can swim"""

    def swim(self):
        print(f"{self.name} is swimming...")


# Problem 1: What about a duck that can both fly AND swim?
# We'd need multiple inheritance...
class Duck(FlyingAnimal, SwimmingAnimal):
    """Duck can fly and swim - uses multiple inheritance"""

    pass


# Problem 2: What about a penguin? It swims but doesn't fly!
class Penguin(SwimmingAnimal):
    """Penguin swims but doesn't fly"""

    pass


# But what if we made Penguin inherit from FlyingAnimal by mistake?
class PenguinBad(FlyingAnimal):
    """Inherits fly() but penguins can't fly!"""

    def fly(self):
        # Have to override and raise error or do nothing
        raise NotImplementedError("Penguins can't fly!")


# Problem 3: What about a robotic dog? It's not an Animal!
# But we want similar behavior...
# Can't inherit from Animal (it's not biological)
# Have to duplicate code or create weird hierarchies


# =============================================================================
# SOLUTION: Composition over Inheritance
# =============================================================================

print("\n" + "=" * 70)
print("THE COMPOSITION SOLUTION")
print("=" * 70)


# ✅ GOOD: Define behaviors as separate components
class FlyingBehavior(ABC):
    """Abstract flying behavior"""

    @abstractmethod
    def fly(self, name: str):
        pass


class CanFly(FlyingBehavior):
    """Concrete flying implementation"""

    def fly(self, name: str):
        print(f"{name} is flying high in the sky!")


class CannotFly(FlyingBehavior):
    """Concrete non-flying implementation"""

    def fly(self, name: str):
        print(f"{name} can't fly.")


class SwimmingBehavior(ABC):
    """Abstract swimming behavior"""

    @abstractmethod
    def swim(self, name: str):
        pass


class CanSwim(SwimmingBehavior):
    """Concrete swimming implementation"""

    def swim(self, name: str):
        print(f"{name} is swimming gracefully!")


class CannotSwim(SwimmingBehavior):
    """Concrete non-swimming implementation"""

    def swim(self, name: str):
        print(f"{name} can't swim.")


class EatingBehavior(ABC):
    """Abstract eating behavior"""

    @abstractmethod
    def eat(self, name: str):
        pass


class Herbivore(EatingBehavior):
    def eat(self, name: str):
        print(f"{name} is eating plants...")


class Carnivore(EatingBehavior):
    def eat(self, name: str):
        print(f"{name} is eating meat...")


class Omnivore(EatingBehavior):
    def eat(self, name: str):
        print(f"{name} is eating everything...")


# Now compose behaviors into entities
class ComposedAnimal:
    """Animal with composable behaviors"""

    def __init__(
        self,
        name: str,
        flying_behavior: FlyingBehavior,
        swimming_behavior: SwimmingBehavior,
        eating_behavior: EatingBehavior,
    ):
        self.name = name
        self._flying_behavior = flying_behavior
        self._swimming_behavior = swimming_behavior
        self._eating_behavior = eating_behavior

    def fly(self):
        """Delegate to flying behavior"""
        self._flying_behavior.fly(self.name)

    def swim(self):
        """Delegate to swimming behavior"""
        self._swimming_behavior.swim(self.name)

    def eat(self):
        """Delegate to eating behavior"""
        self._eating_behavior.eat(self.name)

    def set_flying_behavior(self, behavior: FlyingBehavior):
        """Can change behavior at runtime!"""
        self._flying_behavior = behavior


# =============================================================================
# REAL-WORLD EXAMPLE: Employee Management System
# =============================================================================

print("\n" + "=" * 70)
print("REAL-WORLD EXAMPLE: Employee System")
print("=" * 70)


# ❌ BAD: Inheritance-based approach
class EmployeeBad:
    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary

    def calculate_bonus(self) -> float:
        return self.salary * 0.1


class ManagerBad(EmployeeBad):
    def calculate_bonus(self) -> float:
        return self.salary * 0.2


class DeveloperBad(EmployeeBad):
    def code(self):
        print(f"{self.name} is coding...")


class DesignerBad(EmployeeBad):
    def design(self):
        print(f"{self.name} is designing...")


# Problem: What about a developer who becomes a manager?
# Or someone who codes AND designs?
# Inheritance forces rigid categories!


# ✅ GOOD: Composition-based approach
class BonusCalculator(ABC):
    @abstractmethod
    def calculate(self, salary: float) -> float:
        pass


class StandardBonus(BonusCalculator):
    def calculate(self, salary: float) -> float:
        return salary * 0.1


class ManagerBonus(BonusCalculator):
    def calculate(self, salary: float) -> float:
        return salary * 0.2


class SeniorBonus(BonusCalculator):
    def calculate(self, salary: float) -> float:
        return salary * 0.25


class Skill(ABC):
    """Represents a skill an employee can have"""

    @abstractmethod
    def perform(self, name: str):
        pass


class CodingSkill(Skill):
    def perform(self, name: str):
        print(f"{name} is coding...")


class DesignSkill(Skill):
    def perform(self, name: str):
        print(f"{name} is designing...")


class ManagementSkill(Skill):
    def perform(self, name: str):
        print(f"{name} is managing the team...")


class Employee:
    """Flexible employee with composable skills and bonus calculation"""

    def __init__(
        self,
        name: str,
        salary: float,
        bonus_calculator: BonusCalculator,
        skills: Optional[List[Skill]] = None,
    ):
        self.name = name
        self.salary = salary
        self._bonus_calculator = bonus_calculator
        self._skills = skills or []

    def calculate_bonus(self) -> float:
        return self._bonus_calculator.calculate(self.salary)

    def add_skill(self, skill: Skill):
        """Can add skills dynamically!"""
        self._skills.append(skill)

    def perform_skills(self):
        """Execute all skills"""
        for skill in self._skills:
            skill.perform(self.name)

    def promote(self, new_bonus_calculator: BonusCalculator):
        """Change bonus calculation (e.g., promotion)"""
        self._bonus_calculator = new_bonus_calculator


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_inheritance_problems():
    print("\n[INHERITANCE APPROACH]")
    print("-" * 50)

    duck = Duck("Donald")
    duck.eat()
    duck.fly()
    duck.swim()

    penguin = Penguin("Pingu")
    penguin.eat()
    penguin.swim()
    # penguin.fly()  # Doesn't have fly method

    penguin_bad = PenguinBad("BadPingu")
    try:
        penguin_bad.fly()  # Will raise error
    except NotImplementedError as e:
        print(f"Error: {e}")

    print("\nProblems with inheritance:")
    print("  ✗ Rigid hierarchies")
    print("  ✗ Hard to mix behaviors (flying + swimming)")
    print("  ✗ Breaking Liskov Substitution Principle")
    print("  ✗ Can't change behavior at runtime")


def demonstrate_composition_solution():
    print("\n[COMPOSITION APPROACH]")
    print("-" * 50)

    # Duck: can fly, can swim, eats everything
    duck = ComposedAnimal(
        "Donald",
        flying_behavior=CanFly(),
        swimming_behavior=CanSwim(),
        eating_behavior=Omnivore(),
    )
    duck.fly()
    duck.swim()
    duck.eat()

    # Penguin: cannot fly, can swim, eats fish
    penguin = ComposedAnimal(
        "Pingu",
        flying_behavior=CannotFly(),
        swimming_behavior=CanSwim(),
        eating_behavior=Carnivore(),
    )
    penguin.fly()
    penguin.swim()
    penguin.eat()

    # Robot bird: can fly (with motors), cannot swim, doesn't eat
    robot = ComposedAnimal(
        "RoboBird",
        flying_behavior=CanFly(),
        swimming_behavior=CannotSwim(),
        eating_behavior=Herbivore(),  # Charges battery instead
    )
    robot.fly()

    # Change behavior at runtime!
    print("\n[RUNTIME BEHAVIOR CHANGE]")
    print("Penguin learns to fly (with jetpack):")
    penguin.set_flying_behavior(CanFly())
    penguin.fly()

    print("\nBenefits of composition:")
    print("  ✓ Flexible behavior mixing")
    print("  ✓ Runtime behavior changes")
    print("  ✓ Easy to test (mock behaviors)")
    print("  ✓ Follows Open/Closed Principle")
    print("  ✓ No inheritance coupling")


def demonstrate_employee_system():
    print("\n[EMPLOYEE SYSTEM EXAMPLE]")
    print("-" * 50)

    # Junior developer
    alice = Employee(
        "Alice",
        salary=60000,
        bonus_calculator=StandardBonus(),
        skills=[CodingSkill()],
    )
    alice.perform_skills()
    print(f"Bonus: ${alice.calculate_bonus()}")

    # Senior developer who also designs
    bob = Employee(
        "Bob",
        salary=90000,
        bonus_calculator=SeniorBonus(),
        skills=[CodingSkill(), DesignSkill()],
    )
    print(f"\n{bob.name}'s skills:")
    bob.perform_skills()
    print(f"Bonus: ${bob.calculate_bonus()}")

    # Promote Alice to manager
    print(f"\n{alice.name} gets promoted to manager!")
    alice.promote(ManagerBonus())
    alice.add_skill(ManagementSkill())
    alice.perform_skills()
    print(f"New bonus: ${alice.calculate_bonus()}")

    print("\nComposition allows:")
    print("  ✓ Employees with multiple skills")
    print("  ✓ Dynamic skill addition")
    print("  ✓ Easy promotions (change bonus calculator)")
    print("  ✓ No rigid job categories")


def print_comparison():
    print("\n" + "=" * 70)
    print("COMPOSITION VS INHERITANCE")
    print("=" * 70)

    print("""
WHEN TO USE INHERITANCE:
  ✓ True "is-a" relationship (Cat IS-A Animal)
  ✓ Shared implementation across subtypes
  ✓ Shallow hierarchies (1-2 levels)
  ✓ Liskov Substitution Principle holds

WHEN TO USE COMPOSITION:
  ✓ "has-a" or "can-do" relationships
  ✓ Need to mix multiple behaviors
  ✓ Want to change behavior at runtime
  ✓ Need flexibility and loose coupling
  ✓ Want to avoid fragile base class problem

KEY PRINCIPLE:
  "Favor composition over inheritance"

  This doesn't mean NEVER use inheritance.
  It means: think carefully before inheriting,
  and consider if composition is better.

COMPOSITION ADVANTAGES:
  • More flexible
  • Better testability
  • Easier to reason about
  • No deep hierarchies
  • Runtime behavior changes
  • Follows SOLID principles better

INHERITANCE ADVANTAGES:
  • More intuitive for true "is-a" relationships
  • Less boilerplate for simple cases
  • Built-in language support
""")


if __name__ == "__main__":
    demonstrate_inheritance_problems()
    demonstrate_composition_solution()
    demonstrate_employee_system()
    print_comparison()
