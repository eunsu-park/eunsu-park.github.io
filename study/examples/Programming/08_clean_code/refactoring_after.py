"""
GOOD CODE EXAMPLE - After Refactoring

This code demonstrates clean code principles:
1. Single Responsibility Principle
2. Named constants instead of magic numbers
3. Clear, descriptive naming
4. Small, focused functions
5. Proper error handling
6. No deep nesting (early returns)
7. Separation of concerns
8. Dependency injection
9. Type hints for clarity
"""

from typing import List, Dict, Optional, Protocol
from dataclasses import dataclass
from enum import Enum


# ✅ Named constants instead of magic numbers
class EmployeeConstants:
    """Constants for employee management"""
    MINIMUM_AGE = 18
    MINIMUM_NAME_LENGTH = 2
    PREMIUM_SALARY_THRESHOLD = 150_000
    BONUS_MULTIPLIER = 1.2

    # Bonus rate thresholds
    SALARY_TIER_1 = 50_000
    SALARY_TIER_2 = 100_000
    SALARY_TIER_3 = 150_000

    BONUS_RATE_TIER_1 = 0.10
    BONUS_RATE_TIER_2 = 0.15
    BONUS_RATE_TIER_3 = 0.20
    BONUS_RATE_PREMIUM = 0.25
    BONUS_PREMIUM_EXTRA = 5_000

    TEAM_BONUS_THRESHOLD = 10
    TEAM_BONUS_MULTIPLIER = 1.10
    BULK_DISCOUNT_THRESHOLD = 100_000
    BULK_DISCOUNT_RATE = 0.30


# ✅ Enum for employee tier (better than boolean or magic strings)
class EmployeeTier(Enum):
    STANDARD = "standard"
    PREMIUM = "premium"


# ✅ Data class with type hints - clear structure
@dataclass
class Employee:
    """
    Represents an employee with all relevant information.
    Immutable after creation (frozen=True would make it fully immutable).
    """
    name: str
    email: str
    age: int
    salary: float
    tier: EmployeeTier = EmployeeTier.STANDARD

    def calculate_bonus(self) -> float:
        """Calculate employee bonus based on salary tier"""
        if self.salary < EmployeeConstants.SALARY_TIER_1:
            return self.salary * EmployeeConstants.BONUS_RATE_TIER_1

        elif self.salary < EmployeeConstants.SALARY_TIER_2:
            return self.salary * EmployeeConstants.BONUS_RATE_TIER_2

        elif self.salary < EmployeeConstants.SALARY_TIER_3:
            return self.salary * EmployeeConstants.BONUS_RATE_TIER_3

        else:
            base_bonus = self.salary * EmployeeConstants.BONUS_RATE_PREMIUM
            return base_bonus + EmployeeConstants.BONUS_PREMIUM_EXTRA


# ✅ Custom exceptions for specific error cases
class ValidationError(Exception):
    """Raised when validation fails"""
    pass


class DuplicateEmailError(ValidationError):
    """Raised when trying to add duplicate email"""
    pass


class EmployeeNotFoundError(Exception):
    """Raised when employee doesn't exist"""
    pass


# ✅ Single Responsibility: Validator class only validates
class EmployeeValidator:
    """Validates employee data before creation"""

    @staticmethod
    def validate_age(age: int) -> None:
        """Validate employee age"""
        if age < EmployeeConstants.MINIMUM_AGE:
            raise ValidationError(
                f"Employee must be at least {EmployeeConstants.MINIMUM_AGE} years old"
            )

    @staticmethod
    def validate_name(name: str) -> None:
        """Validate employee name"""
        if len(name) < EmployeeConstants.MINIMUM_NAME_LENGTH:
            raise ValidationError(
                f"Name must be at least {EmployeeConstants.MINIMUM_NAME_LENGTH} characters"
            )

    @staticmethod
    def validate_email(email: str) -> None:
        """Validate email format (basic check)"""
        if "@" not in email or "." not in email:
            raise ValidationError("Invalid email format")

    @staticmethod
    def validate_salary(salary: float) -> None:
        """Validate salary is positive"""
        if salary < 0:
            raise ValidationError("Salary cannot be negative")

    @classmethod
    def validate_employee(cls, name: str, email: str, age: int, salary: float) -> None:
        """Validate all employee fields"""
        cls.validate_name(name)
        cls.validate_email(email)
        cls.validate_age(age)
        cls.validate_salary(salary)


# ✅ Protocol for data storage (dependency inversion)
class EmployeeRepository(Protocol):
    """Interface for employee storage"""

    def add(self, employee: Employee) -> None:
        """Add employee to repository"""
        ...

    def find_by_email(self, email: str) -> Optional[Employee]:
        """Find employee by email"""
        ...

    def get_all(self) -> List[Employee]:
        """Get all employees"""
        ...

    def remove(self, email: str) -> None:
        """Remove employee by email"""
        ...

    def update(self, email: str, **kwargs) -> None:
        """Update employee fields"""
        ...


# ✅ Concrete implementation of repository
class InMemoryEmployeeRepository:
    """In-memory storage for employees"""

    def __init__(self):
        self._employees: List[Employee] = []

    def add(self, employee: Employee) -> None:
        """Add employee, checking for duplicates"""
        if self.find_by_email(employee.email):
            raise DuplicateEmailError(f"Email {employee.email} already exists")
        self._employees.append(employee)

    def find_by_email(self, email: str) -> Optional[Employee]:
        """Find employee by email, return None if not found"""
        for employee in self._employees:
            if employee.email == email:
                return employee
        return None

    def get_all(self) -> List[Employee]:
        """Return copy of all employees (encapsulation)"""
        return self._employees.copy()

    def remove(self, email: str) -> None:
        """Remove employee by email"""
        employee = self.find_by_email(email)
        if not employee:
            raise EmployeeNotFoundError(f"Employee with email {email} not found")
        self._employees.remove(employee)

    def update(self, email: str, **kwargs) -> None:
        """Update employee fields with validation"""
        employee = self.find_by_email(email)
        if not employee:
            raise EmployeeNotFoundError(f"Employee with email {email} not found")

        # Update allowed fields
        if 'name' in kwargs:
            EmployeeValidator.validate_name(kwargs['name'])
            employee.name = kwargs['name']

        if 'salary' in kwargs:
            EmployeeValidator.validate_salary(kwargs['salary'])
            employee.salary = kwargs['salary']

        if 'age' in kwargs:
            EmployeeValidator.validate_age(kwargs['age'])
            employee.age = kwargs['age']


# ✅ Service class with single responsibility
class EmployeeManager:
    """
    Manages employee operations.
    Uses dependency injection for flexibility.
    """

    def __init__(self, repository: EmployeeRepository):
        self._repository = repository
        self._employee_count = 0

    def add_employee(
        self,
        name: str,
        email: str,
        age: int,
        salary: float
    ) -> Employee:
        """
        Add new employee with validation.
        Clear name, clear purpose, proper error handling.
        """
        # Validate input (early return on error)
        EmployeeValidator.validate_employee(name, email, age, salary)

        # Determine tier based on salary
        tier = (EmployeeTier.PREMIUM
                if salary >= EmployeeConstants.PREMIUM_SALARY_THRESHOLD
                else EmployeeTier.STANDARD)

        # Create employee
        employee = Employee(name, email, age, salary, tier)

        # Add to repository (repository handles duplicate check)
        self._repository.add(employee)
        self._employee_count += 1

        return employee

    def get_employee(self, email: str) -> Optional[Employee]:
        """Get employee by email, return None if not found"""
        return self._repository.find_by_email(email)

    def remove_employee(self, email: str) -> None:
        """Remove employee by email"""
        self._repository.remove(email)
        self._employee_count -= 1

    def update_employee(self, email: str, **kwargs) -> None:
        """Update employee with validation"""
        self._repository.update(email, **kwargs)

    def get_employee_count(self) -> int:
        """Get total number of employees"""
        return self._employee_count

    def calculate_total_bonuses(self) -> float:
        """
        Calculate total bonuses for all employees.
        Clear method name, single responsibility.
        """
        employees = self._repository.get_all()

        # Calculate base bonuses
        total_bonuses = sum(emp.calculate_bonus() for emp in employees)

        # Apply team bonus if threshold met
        if self._should_apply_team_bonus(employees):
            total_bonuses *= EmployeeConstants.TEAM_BONUS_MULTIPLIER

        # Apply bulk discount if threshold met
        if self._should_apply_bulk_discount(total_bonuses):
            discount = total_bonuses * EmployeeConstants.BULK_DISCOUNT_RATE
            total_bonuses -= discount

        return total_bonuses

    def _should_apply_team_bonus(self, employees: List[Employee]) -> bool:
        """Check if team bonus should be applied (private helper)"""
        return len(employees) > EmployeeConstants.TEAM_BONUS_THRESHOLD

    def _should_apply_bulk_discount(self, total: float) -> bool:
        """Check if bulk discount should be applied (private helper)"""
        return total > EmployeeConstants.BULK_DISCOUNT_THRESHOLD


# ✅ Pure functions with clear purpose
def transform_data(data: List[int], threshold: int, high_multiplier: int,
                   low_multiplier: int, offset: int) -> List[int]:
    """
    Transform data based on threshold.
    Clear parameters instead of magic numbers.
    """
    return [
        value * high_multiplier + offset if value > threshold
        else value * low_multiplier
        for value in data
    ]


# ✅ Proper error handling with context manager
def read_and_process_csv(filename: str) -> List[Dict[str, str]]:
    """
    Read and process CSV file with proper error handling.
    Uses context manager to ensure file is closed.
    """
    try:
        with open(filename, 'r') as file:
            data = file.read()

        lines = data.strip().split('\n')
        result = []

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            parts = line.split(',')

            # Validate format
            if len(parts) < 3:
                raise ValueError(
                    f"Line {line_num}: Expected at least 3 fields, got {len(parts)}"
                )

            try:
                result.append({
                    'name': parts[0].strip(),
                    'value': int(parts[1].strip()),
                    'category': parts[2].strip()
                })
            except ValueError as e:
                raise ValueError(f"Line {line_num}: Invalid number format") from e

        return result

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filename}")
    except PermissionError:
        raise PermissionError(f"Permission denied: {filename}")


# ✅ Split function instead of boolean parameter
def get_active_employees(repository: EmployeeRepository) -> List[Employee]:
    """Get only active employees (clear, single purpose)"""
    # Implementation would filter active employees
    return repository.get_all()


def get_all_employees_including_inactive(repository: EmployeeRepository) -> List[Employee]:
    """Get all employees including inactive (clear, single purpose)"""
    return repository.get_all()


# ✅ Clear function name and extracted calculation
def calculate_weighted_sum(x: float, y: float, z: float) -> float:
    """
    Calculate weighted sum: ((x + y) * z - 10) / 2 * 1.5
    Formula is now clear from implementation, not from comments.
    """
    sum_xy = x + y
    multiplied = sum_xy * z
    adjusted = (multiplied - 10) / 2
    result = adjusted * 1.5
    return result


# ✅ Demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("CLEAN CODE DEMONSTRATION")
    print("=" * 70)
    print("\nThis code demonstrates:")
    print("  ✓ Named constants (no magic numbers)")
    print("  ✓ Clear, descriptive names")
    print("  ✓ Single Responsibility Principle")
    print("  ✓ Small, focused functions")
    print("  ✓ Proper error handling")
    print("  ✓ Type hints")
    print("  ✓ Separation of concerns")
    print("  ✓ Dependency injection")

    print("\n" + "-" * 70)
    print("Running clean code example:")
    print("-" * 70)

    # Create repository and manager
    repository = InMemoryEmployeeRepository()
    manager = EmployeeManager(repository)

    # Add employees with clear error handling
    try:
        alice = manager.add_employee("Alice Smith", "alice@example.com", 25, 60_000)
        print(f"✓ Added: {alice.name} ({alice.tier.value})")

        bob = manager.add_employee("Bob Johnson", "bob@example.com", 30, 180_000)
        print(f"✓ Added: {bob.name} ({bob.tier.value})")

        # This will fail validation
        try:
            manager.add_employee("Charlie", "charlie@example.com", 15, 0)
        except ValidationError as e:
            print(f"✗ Validation failed: {e}")

        # This will fail duplicate check
        try:
            manager.add_employee("Alice Clone", "alice@example.com", 30, 50_000)
        except DuplicateEmailError as e:
            print(f"✗ Duplicate email: {e}")

    except Exception as e:
        print(f"Error: {e}")

    print(f"\nTotal employees: {manager.get_employee_count()}")
    print(f"Total bonuses: ${manager.calculate_total_bonuses():,.2f}")

    # Find employee
    employee = manager.get_employee("alice@example.com")
    if employee:
        print(f"\nFound employee: {employee.name}")
        print(f"  Email: {employee.email}")
        print(f"  Salary: ${employee.salary:,.2f}")
        print(f"  Bonus: ${employee.calculate_bonus():,.2f}")

    print("\n" + "=" * 70)
    print("IMPROVEMENTS FROM REFACTORING:")
    print("=" * 70)
    print("""
1. READABILITY
   ✓ Clear, descriptive names
   ✓ Named constants explain meaning
   ✓ Type hints document expected types
   ✓ Single purpose per function

2. MAINTAINABILITY
   ✓ Changes localized to specific classes
   ✓ No code duplication
   ✓ Easy to modify without breaking
   ✓ Constants in one place

3. TESTABILITY
   ✓ Small functions easy to test
   ✓ Dependency injection allows mocking
   ✓ Clear error cases
   ✓ Pure functions where possible

4. EXTENSIBILITY
   ✓ Protocol/Interface for flexibility
   ✓ Follows SOLID principles
   ✓ Easy to add new features
   ✓ Loose coupling

5. RELIABILITY
   ✓ Proper error handling
   ✓ Input validation
   ✓ Custom exceptions
   ✓ Resource management (context managers)

COMPARISON TO BEFORE:
  Before: UsrMgr.p(n, e, a, s)  → What does this do?
  After:  manager.add_employee(name, email, age, salary)  → Clear!

  Before: Magic number 18  → Why 18?
  After:  EmployeeConstants.MINIMUM_AGE  → Clear meaning!

  Before: Deep nesting with duplicated logic
  After:  Small functions with single purpose

  Before: No error handling, crashes on bad input
  After:  Validation and custom exceptions
""")
