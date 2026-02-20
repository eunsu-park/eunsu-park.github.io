"""
Calculator Module

Simple calculator implementation to demonstrate testing.
This module will be tested in test_calculator.py.
"""

from typing import Union, List
import math


class Calculator:
    """
    Basic calculator with arithmetic operations.
    Demonstrates testable code design.
    """

    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        return a + b

    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a"""
        return a - b

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        return a * b

    def divide(self, a: float, b: float) -> float:
        """
        Divide a by b.

        Raises:
            ValueError: If b is zero
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent"""
        return base ** exponent

    def square_root(self, x: float) -> float:
        """
        Calculate square root.

        Raises:
            ValueError: If x is negative
        """
        if x < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return math.sqrt(x)

    def modulo(self, a: float, b: float) -> float:
        """
        Calculate a modulo b.

        Raises:
            ValueError: If b is zero
        """
        if b == 0:
            raise ValueError("Cannot calculate modulo with zero divisor")
        return a % b


class ScientificCalculator(Calculator):
    """
    Extended calculator with scientific functions.
    Demonstrates inheritance testing.
    """

    def sin(self, x: float) -> float:
        """Calculate sine (x in radians)"""
        return math.sin(x)

    def cos(self, x: float) -> float:
        """Calculate cosine (x in radians)"""
        return math.cos(x)

    def tan(self, x: float) -> float:
        """Calculate tangent (x in radians)"""
        return math.tan(x)

    def log(self, x: float, base: float = math.e) -> float:
        """
        Calculate logarithm.

        Args:
            x: Value to calculate log of
            base: Logarithm base (default: e for natural log)

        Raises:
            ValueError: If x <= 0 or base <= 0 or base == 1
        """
        if x <= 0:
            raise ValueError("Logarithm input must be positive")
        if base <= 0 or base == 1:
            raise ValueError("Logarithm base must be positive and not equal to 1")
        return math.log(x, base)

    def factorial(self, n: int) -> int:
        """
        Calculate factorial.

        Raises:
            ValueError: If n is negative
            TypeError: If n is not an integer
        """
        if not isinstance(n, int):
            raise TypeError("Factorial requires integer input")
        if n < 0:
            raise ValueError("Factorial not defined for negative numbers")
        return math.factorial(n)


class StatisticsCalculator:
    """
    Calculator for statistical operations.
    Demonstrates testing with data structures.
    """

    def mean(self, numbers: List[float]) -> float:
        """
        Calculate arithmetic mean.

        Raises:
            ValueError: If list is empty
        """
        if not numbers:
            raise ValueError("Cannot calculate mean of empty list")
        return sum(numbers) / len(numbers)

    def median(self, numbers: List[float]) -> float:
        """
        Calculate median.

        Raises:
            ValueError: If list is empty
        """
        if not numbers:
            raise ValueError("Cannot calculate median of empty list")

        sorted_numbers = sorted(numbers)
        n = len(sorted_numbers)

        if n % 2 == 0:
            # Even number of elements - average of two middle values
            return (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2
        else:
            # Odd number of elements - middle value
            return sorted_numbers[n // 2]

    def mode(self, numbers: List[float]) -> float:
        """
        Calculate mode (most frequent value).

        Raises:
            ValueError: If list is empty or no unique mode
        """
        if not numbers:
            raise ValueError("Cannot calculate mode of empty list")

        frequency = {}
        for num in numbers:
            frequency[num] = frequency.get(num, 0) + 1

        max_freq = max(frequency.values())
        modes = [num for num, freq in frequency.items() if freq == max_freq]

        if len(modes) > 1:
            raise ValueError("No unique mode exists")

        return modes[0]

    def variance(self, numbers: List[float]) -> float:
        """
        Calculate variance.

        Raises:
            ValueError: If list is empty
        """
        if not numbers:
            raise ValueError("Cannot calculate variance of empty list")

        avg = self.mean(numbers)
        return sum((x - avg) ** 2 for x in numbers) / len(numbers)

    def standard_deviation(self, numbers: List[float]) -> float:
        """
        Calculate standard deviation.

        Raises:
            ValueError: If list is empty
        """
        return math.sqrt(self.variance(numbers))


class CalculatorMemory:
    """
    Calculator with memory functionality.
    Demonstrates stateful testing.
    """

    def __init__(self):
        self._memory: float = 0.0
        self._history: List[float] = []

    def store(self, value: float) -> None:
        """Store value in memory"""
        self._memory = value
        self._history.append(value)

    def recall(self) -> float:
        """Recall value from memory"""
        return self._memory

    def clear(self) -> None:
        """Clear memory"""
        self._memory = 0.0

    def add_to_memory(self, value: float) -> None:
        """Add value to current memory"""
        self._memory += value
        self._history.append(self._memory)

    def get_history(self) -> List[float]:
        """Get calculation history"""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear calculation history"""
        self._history.clear()


# Helper functions for demonstrating pure function testing
def is_even(n: int) -> bool:
    """Check if number is even"""
    return n % 2 == 0


def is_prime(n: int) -> bool:
    """
    Check if number is prime.

    Args:
        n: Number to check

    Returns:
        True if prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False

    return True


def fibonacci(n: int) -> int:
    """
    Calculate nth Fibonacci number (0-indexed).

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Fibonacci not defined for negative numbers")
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def gcd(a: int, b: int) -> int:
    """
    Calculate greatest common divisor using Euclidean algorithm.

    Raises:
        ValueError: If either number is negative
    """
    if a < 0 or b < 0:
        raise ValueError("GCD not defined for negative numbers")

    while b:
        a, b = b, a % b
    return a


if __name__ == "__main__":
    # Simple demonstration
    calc = Calculator()
    print("Calculator Demo:")
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"10 - 4 = {calc.subtract(10, 4)}")
    print(f"6 * 7 = {calc.multiply(6, 7)}")
    print(f"15 / 3 = {calc.divide(15, 3)}")
    print(f"2^8 = {calc.power(2, 8)}")
    print(f"√16 = {calc.square_root(16)}")

    sci_calc = ScientificCalculator()
    print(f"\nScientific Calculator Demo:")
    print(f"sin(π/2) = {sci_calc.sin(math.pi / 2):.4f}")
    print(f"cos(0) = {sci_calc.cos(0):.4f}")
    print(f"5! = {sci_calc.factorial(5)}")

    stats_calc = StatisticsCalculator()
    data = [1, 2, 3, 4, 5]
    print(f"\nStatistics Demo:")
    print(f"Mean of {data} = {stats_calc.mean(data)}")
    print(f"Median of {data} = {stats_calc.median(data)}")
    print(f"Std Dev of {data} = {stats_calc.standard_deviation(data):.4f}")

    print(f"\nHelper Functions Demo:")
    print(f"is_prime(17) = {is_prime(17)}")
    print(f"fibonacci(10) = {fibonacci(10)}")
    print(f"gcd(48, 18) = {gcd(48, 18)}")
