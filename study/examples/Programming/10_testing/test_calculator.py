"""
Test Suite for Calculator Module

Demonstrates comprehensive testing with pytest:
1. Basic unit tests
2. Parametrized tests
3. Fixtures
4. Mocking
5. Edge cases
6. Exception testing
7. Test organization
8. TDD style

Run with: pytest test_calculator.py -v
"""

import pytest
import math
from unittest.mock import Mock, patch
from calculator import (
    Calculator,
    ScientificCalculator,
    StatisticsCalculator,
    CalculatorMemory,
    is_even,
    is_prime,
    fibonacci,
    gcd
)


# =============================================================================
# FIXTURES - Reusable test setup
# =============================================================================

@pytest.fixture
def calculator():
    """Fixture providing Calculator instance"""
    return Calculator()


@pytest.fixture
def sci_calculator():
    """Fixture providing ScientificCalculator instance"""
    return ScientificCalculator()


@pytest.fixture
def stats_calculator():
    """Fixture providing StatisticsCalculator instance"""
    return StatisticsCalculator()


@pytest.fixture
def calculator_memory():
    """Fixture providing CalculatorMemory instance"""
    return CalculatorMemory()


@pytest.fixture
def sample_data():
    """Fixture providing sample test data"""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def sample_data_with_duplicates():
    """Fixture providing data with duplicates"""
    return [1, 2, 2, 3, 3, 3, 4]


# =============================================================================
# BASIC CALCULATOR TESTS
# =============================================================================

class TestCalculatorBasicOperations:
    """Test basic arithmetic operations"""

    def test_add_positive_numbers(self, calculator):
        """Test addition of positive numbers"""
        result = calculator.add(5, 3)
        assert result == 8

    def test_add_negative_numbers(self, calculator):
        """Test addition of negative numbers"""
        result = calculator.add(-5, -3)
        assert result == -8

    def test_subtract(self, calculator):
        """Test subtraction"""
        result = calculator.subtract(10, 4)
        assert result == 6

    def test_multiply(self, calculator):
        """Test multiplication"""
        result = calculator.multiply(6, 7)
        assert result == 42

    def test_divide(self, calculator):
        """Test division"""
        result = calculator.divide(15, 3)
        assert result == 5.0

    def test_divide_by_zero_raises_error(self, calculator):
        """Test that division by zero raises ValueError"""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calculator.divide(10, 0)

    def test_power(self, calculator):
        """Test exponentiation"""
        result = calculator.power(2, 8)
        assert result == 256

    def test_square_root(self, calculator):
        """Test square root"""
        result = calculator.square_root(16)
        assert result == 4.0

    def test_square_root_negative_raises_error(self, calculator):
        """Test that square root of negative raises ValueError"""
        with pytest.raises(ValueError, match="Cannot calculate square root"):
            calculator.square_root(-1)

    def test_modulo(self, calculator):
        """Test modulo operation"""
        result = calculator.modulo(10, 3)
        assert result == 1

    def test_modulo_zero_raises_error(self, calculator):
        """Test that modulo by zero raises ValueError"""
        with pytest.raises(ValueError, match="Cannot calculate modulo"):
            calculator.modulo(10, 0)


# =============================================================================
# PARAMETRIZED TESTS - Test multiple inputs efficiently
# =============================================================================

class TestCalculatorParametrized:
    """Parametrized tests for comprehensive coverage"""

    @pytest.mark.parametrize("a, b, expected", [
        (1, 1, 2),
        (0, 0, 0),
        (-1, 1, 0),
        (100, 200, 300),
        (0.1, 0.2, pytest.approx(0.3)),  # Floating point comparison
    ])
    def test_add_parametrized(self, calculator, a, b, expected):
        """Test addition with multiple input combinations"""
        assert calculator.add(a, b) == expected

    @pytest.mark.parametrize("a, b, expected", [
        (10, 2, 5.0),
        (7, 2, 3.5),
        (-10, 2, -5.0),
        (10, -2, -5.0),
    ])
    def test_divide_parametrized(self, calculator, a, b, expected):
        """Test division with multiple inputs"""
        assert calculator.divide(a, b) == expected

    @pytest.mark.parametrize("base, exponent, expected", [
        (2, 0, 1),
        (2, 1, 2),
        (2, 10, 1024),
        (5, 2, 25),
        (10, -1, 0.1),
    ])
    def test_power_parametrized(self, calculator, base, exponent, expected):
        """Test power with various bases and exponents"""
        assert calculator.power(base, exponent) == pytest.approx(expected)


# =============================================================================
# SCIENTIFIC CALCULATOR TESTS
# =============================================================================

class TestScientificCalculator:
    """Test scientific calculator functions"""

    def test_sin(self, sci_calculator):
        """Test sine function"""
        result = sci_calculator.sin(math.pi / 2)
        assert result == pytest.approx(1.0)

    def test_cos(self, sci_calculator):
        """Test cosine function"""
        result = sci_calculator.cos(0)
        assert result == pytest.approx(1.0)

    def test_tan(self, sci_calculator):
        """Test tangent function"""
        result = sci_calculator.tan(math.pi / 4)
        assert result == pytest.approx(1.0)

    @pytest.mark.parametrize("n, expected", [
        (0, 1),
        (1, 1),
        (5, 120),
        (10, 3628800),
    ])
    def test_factorial(self, sci_calculator, n, expected):
        """Test factorial with various inputs"""
        assert sci_calculator.factorial(n) == expected

    def test_factorial_negative_raises_error(self, sci_calculator):
        """Test that factorial of negative raises ValueError"""
        with pytest.raises(ValueError, match="not defined for negative"):
            sci_calculator.factorial(-1)

    def test_factorial_non_integer_raises_error(self, sci_calculator):
        """Test that factorial of non-integer raises TypeError"""
        with pytest.raises(TypeError, match="requires integer"):
            sci_calculator.factorial(3.5)

    def test_log_natural(self, sci_calculator):
        """Test natural logarithm"""
        result = sci_calculator.log(math.e)
        assert result == pytest.approx(1.0)

    def test_log_base_10(self, sci_calculator):
        """Test logarithm base 10"""
        result = sci_calculator.log(100, 10)
        assert result == pytest.approx(2.0)

    def test_log_zero_raises_error(self, sci_calculator):
        """Test that log(0) raises ValueError"""
        with pytest.raises(ValueError, match="must be positive"):
            sci_calculator.log(0)

    def test_log_negative_raises_error(self, sci_calculator):
        """Test that log of negative raises ValueError"""
        with pytest.raises(ValueError, match="must be positive"):
            sci_calculator.log(-1)


# =============================================================================
# STATISTICS CALCULATOR TESTS
# =============================================================================

class TestStatisticsCalculator:
    """Test statistical functions"""

    def test_mean(self, stats_calculator, sample_data):
        """Test mean calculation"""
        result = stats_calculator.mean(sample_data)
        assert result == 3.0

    def test_mean_empty_raises_error(self, stats_calculator):
        """Test that mean of empty list raises ValueError"""
        with pytest.raises(ValueError, match="empty list"):
            stats_calculator.mean([])

    def test_median_odd_length(self, stats_calculator):
        """Test median with odd number of elements"""
        result = stats_calculator.median([1, 2, 3, 4, 5])
        assert result == 3

    def test_median_even_length(self, stats_calculator):
        """Test median with even number of elements"""
        result = stats_calculator.median([1, 2, 3, 4])
        assert result == 2.5

    def test_median_unsorted(self, stats_calculator):
        """Test median with unsorted data"""
        result = stats_calculator.median([5, 1, 3, 2, 4])
        assert result == 3

    def test_mode(self, stats_calculator, sample_data_with_duplicates):
        """Test mode calculation"""
        result = stats_calculator.mode(sample_data_with_duplicates)
        assert result == 3

    def test_mode_no_unique_raises_error(self, stats_calculator):
        """Test that mode with no unique value raises error"""
        with pytest.raises(ValueError, match="No unique mode"):
            stats_calculator.mode([1, 1, 2, 2])

    def test_variance(self, stats_calculator, sample_data):
        """Test variance calculation"""
        result = stats_calculator.variance(sample_data)
        assert result == pytest.approx(2.0)

    def test_standard_deviation(self, stats_calculator, sample_data):
        """Test standard deviation"""
        result = stats_calculator.standard_deviation(sample_data)
        assert result == pytest.approx(math.sqrt(2.0))


# =============================================================================
# STATEFUL TESTS - Testing objects with state
# =============================================================================

class TestCalculatorMemory:
    """Test calculator with memory functionality"""

    def test_store_and_recall(self, calculator_memory):
        """Test storing and recalling values"""
        calculator_memory.store(42)
        assert calculator_memory.recall() == 42

    def test_initial_memory_is_zero(self, calculator_memory):
        """Test that initial memory is 0"""
        assert calculator_memory.recall() == 0

    def test_clear_memory(self, calculator_memory):
        """Test clearing memory"""
        calculator_memory.store(100)
        calculator_memory.clear()
        assert calculator_memory.recall() == 0

    def test_add_to_memory(self, calculator_memory):
        """Test adding to memory"""
        calculator_memory.store(10)
        calculator_memory.add_to_memory(5)
        assert calculator_memory.recall() == 15

    def test_history_tracking(self, calculator_memory):
        """Test that history is tracked correctly"""
        calculator_memory.store(10)
        calculator_memory.store(20)
        calculator_memory.add_to_memory(5)

        history = calculator_memory.get_history()
        assert history == [10, 20, 25]

    def test_clear_history(self, calculator_memory):
        """Test clearing history"""
        calculator_memory.store(10)
        calculator_memory.store(20)
        calculator_memory.clear_history()

        assert calculator_memory.get_history() == []


# =============================================================================
# PURE FUNCTION TESTS
# =============================================================================

class TestHelperFunctions:
    """Test pure helper functions"""

    @pytest.mark.parametrize("n, expected", [
        (0, True),
        (1, False),
        (2, True),
        (10, True),
        (11, False),
    ])
    def test_is_even(self, n, expected):
        """Test even number detection"""
        assert is_even(n) == expected

    @pytest.mark.parametrize("n, expected", [
        (0, False),
        (1, False),
        (2, True),
        (3, True),
        (4, False),
        (17, True),
        (20, False),
        (97, True),
    ])
    def test_is_prime(self, n, expected):
        """Test prime number detection"""
        assert is_prime(n) == expected

    @pytest.mark.parametrize("n, expected", [
        (0, 0),
        (1, 1),
        (5, 5),
        (10, 55),
        (15, 610),
    ])
    def test_fibonacci(self, n, expected):
        """Test Fibonacci sequence"""
        assert fibonacci(n) == expected

    def test_fibonacci_negative_raises_error(self):
        """Test that negative Fibonacci raises error"""
        with pytest.raises(ValueError, match="not defined for negative"):
            fibonacci(-1)

    @pytest.mark.parametrize("a, b, expected", [
        (48, 18, 6),
        (100, 50, 50),
        (17, 19, 1),  # Coprime
        (0, 5, 5),
    ])
    def test_gcd(self, a, b, expected):
        """Test greatest common divisor"""
        assert gcd(a, b) == expected


# =============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_add_large_numbers(self, calculator):
        """Test addition with very large numbers"""
        result = calculator.add(1e15, 1e15)
        assert result == 2e15

    def test_divide_very_small_result(self, calculator):
        """Test division resulting in very small number"""
        result = calculator.divide(1, 1e10)
        assert result == pytest.approx(1e-10)

    def test_power_zero_exponent(self, calculator):
        """Test that any number to power 0 equals 1"""
        assert calculator.power(1000, 0) == 1
        assert calculator.power(-5, 0) == 1

    def test_square_root_zero(self, calculator):
        """Test square root of zero"""
        assert calculator.square_root(0) == 0

    def test_mean_single_element(self, stats_calculator):
        """Test mean of single element"""
        assert stats_calculator.mean([42]) == 42

    def test_median_single_element(self, stats_calculator):
        """Test median of single element"""
        assert stats_calculator.median([42]) == 42


# =============================================================================
# MOCKING EXAMPLE
# =============================================================================

class TestMocking:
    """Demonstrate testing with mocks"""

    def test_calculator_with_mock_dependency(self):
        """Example of mocking external dependencies"""
        # Create a mock for external API or service
        mock_service = Mock()
        mock_service.get_exchange_rate.return_value = 1.2

        # Use the mock in calculation
        euros = 100
        rate = mock_service.get_exchange_rate("EUR", "USD")
        dollars = euros * rate

        assert dollars == 120
        mock_service.get_exchange_rate.assert_called_once_with("EUR", "USD")

    @patch('math.sqrt')
    def test_square_root_with_patch(self, mock_sqrt, calculator):
        """Example of patching built-in functions"""
        mock_sqrt.return_value = 5.0

        result = calculator.square_root(25)

        assert result == 5.0
        mock_sqrt.assert_called_once_with(25)


# =============================================================================
# TEST ORGANIZATION AND MARKERS
# =============================================================================

@pytest.mark.slow
class TestSlowOperations:
    """Tests that might be slow (can be skipped with -m "not slow")"""

    def test_factorial_large_number(self, sci_calculator):
        """Test factorial of large number"""
        result = sci_calculator.factorial(100)
        assert result > 0  # Just check it completes


@pytest.mark.integration
class TestIntegration:
    """Integration tests (can be run separately with -m integration)"""

    def test_complex_calculation_workflow(self):
        """Test complete calculation workflow"""
        calc = Calculator()
        sci_calc = ScientificCalculator()
        stats_calc = StatisticsCalculator()

        # Multi-step calculation
        step1 = calc.add(10, 5)
        step2 = calc.multiply(step1, 2)
        step3 = calc.divide(step2, 5)
        step4 = sci_calc.power(step3, 2)

        assert step4 == pytest.approx(36.0)


# =============================================================================
# RUN SUMMARY
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CALCULATOR TEST SUITE")
    print("=" * 70)
    print("""
This test suite demonstrates:

1. FIXTURES
   ✓ Reusable test setup
   ✓ Reduces code duplication
   ✓ Clean test organization

2. PARAMETRIZED TESTS
   ✓ Test multiple inputs efficiently
   ✓ Better coverage with less code
   ✓ Clear test cases

3. EXCEPTION TESTING
   ✓ Verify error handling
   ✓ Check error messages
   ✓ Ensure robustness

4. EDGE CASES
   ✓ Boundary conditions
   ✓ Zero, negative, large numbers
   ✓ Empty collections

5. MOCKING
   ✓ Isolate code under test
   ✓ Simulate external dependencies
   ✓ Verify interactions

6. TEST ORGANIZATION
   ✓ Grouped by functionality
   ✓ Clear naming
   ✓ Test markers for categorization

Run with:
  pytest test_calculator.py -v              # Verbose output
  pytest test_calculator.py -v -m "not slow" # Skip slow tests
  pytest test_calculator.py -k "mean"       # Run specific tests
  pytest test_calculator.py --cov           # With coverage report
""")
