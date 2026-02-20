"""
Python Testing

Demonstrates:
- unittest basics
- Test fixtures and setup/teardown
- Assertions
- pytest-style testing
- Mocking and patching
- Parametrized tests
- Test organization
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Code to Test
# =============================================================================

class Calculator:
    """Simple calculator for testing."""

    def add(self, a: float, b: float) -> float:
        return a + b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def is_even(self, n: int) -> bool:
        return n % 2 == 0


class BankAccount:
    """Bank account for testing."""

    def __init__(self, balance: float = 0):
        self._balance = balance

    @property
    def balance(self) -> float:
        return self._balance

    def deposit(self, amount: float):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount

    def withdraw(self, amount: float):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount


class UserService:
    """User service that depends on database."""

    def __init__(self, database):
        self.database = database

    def get_user(self, user_id: int):
        return self.database.get(user_id)

    def create_user(self, name: str, email: str):
        user = {"name": name, "email": email}
        return self.database.save(user)


# =============================================================================
# unittest Basic Tests
# =============================================================================

section("unittest Basic Tests")


class TestCalculator(unittest.TestCase):
    """Test cases for Calculator."""

    def setUp(self):
        """Set up test fixtures - runs before each test."""
        self.calc = Calculator()

    def tearDown(self):
        """Clean up after each test."""
        self.calc = None

    def test_add(self):
        """Test addition."""
        result = self.calc.add(2, 3)
        self.assertEqual(result, 5)

    def test_add_negative(self):
        """Test addition with negative numbers."""
        result = self.calc.add(-1, -2)
        self.assertEqual(result, -3)

    def test_divide(self):
        """Test division."""
        result = self.calc.divide(10, 2)
        self.assertEqual(result, 5)

    def test_divide_by_zero(self):
        """Test that dividing by zero raises error."""
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)

    def test_is_even(self):
        """Test even number detection."""
        self.assertTrue(self.calc.is_even(4))
        self.assertFalse(self.calc.is_even(5))


# =============================================================================
# Assertion Methods
# =============================================================================

section("Common Assertion Methods")


class TestAssertions(unittest.TestCase):
    """Demonstrate various assertion methods."""

    def test_equality(self):
        """Test equality assertions."""
        self.assertEqual(1 + 1, 2)
        self.assertNotEqual(1 + 1, 3)

    def test_boolean(self):
        """Test boolean assertions."""
        self.assertTrue(True)
        self.assertFalse(False)

    def test_none(self):
        """Test None assertions."""
        self.assertIsNone(None)
        self.assertIsNotNone(42)

    def test_in(self):
        """Test membership assertions."""
        self.assertIn(1, [1, 2, 3])
        self.assertNotIn(4, [1, 2, 3])

    def test_instance(self):
        """Test type assertions."""
        self.assertIsInstance("hello", str)
        self.assertNotIsInstance("hello", int)

    def test_almost_equal(self):
        """Test floating point equality."""
        self.assertAlmostEqual(0.1 + 0.2, 0.3, places=7)

    def test_greater_less(self):
        """Test comparison assertions."""
        self.assertGreater(10, 5)
        self.assertLess(5, 10)
        self.assertGreaterEqual(10, 10)
        self.assertLessEqual(5, 5)


# =============================================================================
# Test Fixtures
# =============================================================================

section("Test Fixtures and Setup")


class TestBankAccount(unittest.TestCase):
    """Test bank account with fixtures."""

    @classmethod
    def setUpClass(cls):
        """Run once before all tests in class."""
        print("  setUpClass: Initialize shared resources")
        cls.bank_name = "Test Bank"

    @classmethod
    def tearDownClass(cls):
        """Run once after all tests in class."""
        print("  tearDownClass: Cleanup shared resources")

    def setUp(self):
        """Run before each test."""
        self.account = BankAccount(100)

    def test_initial_balance(self):
        """Test initial balance."""
        self.assertEqual(self.account.balance, 100)

    def test_deposit(self):
        """Test deposit."""
        self.account.deposit(50)
        self.assertEqual(self.account.balance, 150)

    def test_deposit_negative(self):
        """Test that negative deposit raises error."""
        with self.assertRaises(ValueError):
            self.account.deposit(-10)

    def test_withdraw(self):
        """Test withdrawal."""
        self.account.withdraw(30)
        self.assertEqual(self.account.balance, 70)

    def test_withdraw_insufficient_funds(self):
        """Test withdrawal with insufficient funds."""
        with self.assertRaises(ValueError):
            self.account.withdraw(200)


# =============================================================================
# Mocking and Patching
# =============================================================================

section("Mocking and Patching")


class TestUserServiceWithMock(unittest.TestCase):
    """Test UserService using mocks."""

    def test_get_user_with_mock(self):
        """Test get_user with mocked database."""
        # Create mock database
        mock_db = Mock()
        mock_db.get.return_value = {"id": 1, "name": "Alice"}

        service = UserService(mock_db)
        user = service.get_user(1)

        # Verify mock was called correctly
        mock_db.get.assert_called_once_with(1)
        self.assertEqual(user["name"], "Alice")

    def test_create_user_with_mock(self):
        """Test create_user with mocked database."""
        mock_db = Mock()
        mock_db.save.return_value = {"id": 2, "name": "Bob", "email": "bob@example.com"}

        service = UserService(mock_db)
        user = service.create_user("Bob", "bob@example.com")

        # Verify save was called
        self.assertTrue(mock_db.save.called)
        call_args = mock_db.save.call_args[0][0]
        self.assertEqual(call_args["name"], "Bob")


class TestWithPatch(unittest.TestCase):
    """Test using patch decorator."""

    @patch('__main__.BankAccount')
    def test_with_patch_decorator(self, mock_account_class):
        """Test using patch as decorator."""
        # Configure mock
        mock_instance = mock_account_class.return_value
        mock_instance.balance = 1000

        # Create instance (will be mock)
        account = BankAccount()

        self.assertEqual(account.balance, 1000)

    def test_with_patch_context_manager(self):
        """Test using patch as context manager."""
        with patch('__main__.Calculator') as mock_calc_class:
            mock_instance = mock_calc_class.return_value
            mock_instance.add.return_value = 42

            calc = Calculator()
            result = calc.add(1, 2)

            self.assertEqual(result, 42)
            mock_instance.add.assert_called_once_with(1, 2)


# =============================================================================
# Mock Object Behavior
# =============================================================================

section("Mock Object Behavior")


class TestMockBehavior(unittest.TestCase):
    """Demonstrate mock object features."""

    def test_mock_return_value(self):
        """Test setting mock return value."""
        mock = Mock()
        mock.method.return_value = 42

        result = mock.method()
        self.assertEqual(result, 42)

    def test_mock_side_effect(self):
        """Test mock side effect for multiple calls."""
        mock = Mock()
        mock.method.side_effect = [1, 2, 3]

        self.assertEqual(mock.method(), 1)
        self.assertEqual(mock.method(), 2)
        self.assertEqual(mock.method(), 3)

    def test_mock_side_effect_exception(self):
        """Test mock raising exception."""
        mock = Mock()
        mock.method.side_effect = ValueError("Test error")

        with self.assertRaises(ValueError):
            mock.method()

    def test_mock_call_count(self):
        """Test tracking call count."""
        mock = Mock()

        mock.method()
        mock.method()
        mock.method()

        self.assertEqual(mock.method.call_count, 3)

    def test_magic_mock(self):
        """Test MagicMock with magic methods."""
        mock = MagicMock()

        # MagicMock supports magic methods
        mock.__len__.return_value = 5
        self.assertEqual(len(mock), 5)

        mock.__getitem__.return_value = 42
        self.assertEqual(mock[0], 42)


# =============================================================================
# Subtest Pattern
# =============================================================================

section("Subtests")


class TestWithSubtests(unittest.TestCase):
    """Demonstrate subtests for multiple similar tests."""

    def test_is_even_with_subtests(self):
        """Test multiple values using subtests."""
        calc = Calculator()

        test_cases = [
            (2, True),
            (4, True),
            (6, True),
            (1, False),
            (3, False),
            (5, False),
        ]

        for number, expected in test_cases:
            with self.subTest(number=number):
                result = calc.is_even(number)
                self.assertEqual(result, expected)


# =============================================================================
# Pytest-Style Tests (Without pytest)
# =============================================================================

section("Pytest-Style Tests")


def test_calculator_add():
    """Simple test function (pytest-style)."""
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.add(-1, 1) == 0


def test_calculator_divide():
    """Test with exception (pytest-style)."""
    calc = Calculator()
    assert calc.divide(10, 2) == 5

    try:
        calc.divide(10, 0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected


# =============================================================================
# Parametrized Test Pattern
# =============================================================================

section("Parametrized Test Pattern")


class TestParametrized(unittest.TestCase):
    """Parametrized test pattern."""

    def test_add_parametrized(self):
        """Test add with multiple parameter sets."""
        calc = Calculator()

        test_cases = [
            (2, 3, 5),
            (0, 0, 0),
            (-1, 1, 0),
            (10, -5, 5),
            (1.5, 2.5, 4.0),
        ]

        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b):
                result = calc.add(a, b)
                self.assertAlmostEqual(result, expected)


# =============================================================================
# Running Tests
# =============================================================================

section("Running Tests")

print("""
Run tests with:
  python -m unittest test_module.py
  python -m unittest test_module.TestClass
  python -m unittest test_module.TestClass.test_method
  python -m unittest discover

Verbose output:
  python -m unittest -v test_module.py

With pytest (if installed):
  pytest test_module.py
  pytest -v test_module.py
  pytest -k "test_add"  # Run tests matching pattern
""")


# Run a subset of tests for demonstration
if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add specific test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestBankAccount))
    suite.addTests(loader.loadTestsFromTestCase(TestUserServiceWithMock))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    print("\n" + "=" * 60)
    print("  Running Test Suite")
    print("=" * 60)
    result = runner.run(suite)

    # Summary
    section("Test Summary")
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
