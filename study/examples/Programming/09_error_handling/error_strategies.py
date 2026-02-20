"""
Error Handling Strategies

Demonstrates various error handling approaches:
1. Exception handling (try/except/finally)
2. Custom exceptions
3. Result pattern (type-safe error handling)
4. Context managers (resource management)
5. Retry logic with exponential backoff
6. Error recovery strategies
"""

from typing import Optional, TypeVar, Generic, Callable, Any
from dataclasses import dataclass
from enum import Enum
import time
import random
from contextlib import contextmanager


# =============================================================================
# 1. EXCEPTION HANDLING BASICS
# =============================================================================

print("=" * 70)
print("1. EXCEPTION HANDLING BASICS")
print("=" * 70)


def divide_with_error_handling(a: float, b: float) -> Optional[float]:
    """Division with proper error handling"""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print(f"Error: Cannot divide {a} by zero")
        return None
    except TypeError as e:
        print(f"Error: Invalid types - {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def read_file_with_cleanup(filename: str) -> Optional[str]:
    """File reading with guaranteed cleanup using finally"""
    file = None
    try:
        file = open(filename, 'r')
        content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None
    except PermissionError:
        print(f"Error: Permission denied for '{filename}'")
        return None
    finally:
        # Always executes, even if exception occurs
        if file:
            file.close()
            print("File closed in finally block")


# =============================================================================
# 2. CUSTOM EXCEPTIONS
# =============================================================================

print("\n" + "=" * 70)
print("2. CUSTOM EXCEPTIONS")
print("=" * 70)


class ApplicationError(Exception):
    """Base exception for application errors"""
    pass


class ValidationError(ApplicationError):
    """Raised when validation fails"""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"Validation error in '{field}': {message}")


class AuthenticationError(ApplicationError):
    """Raised when authentication fails"""

    def __init__(self, username: str):
        self.username = username
        super().__init__(f"Authentication failed for user '{username}'")


class ResourceNotFoundError(ApplicationError):
    """Raised when resource doesn't exist"""

    def __init__(self, resource_type: str, resource_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(
            f"{resource_type} with id '{resource_id}' not found"
        )


class RateLimitError(ApplicationError):
    """Raised when rate limit is exceeded"""

    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window} seconds"
        )


# Using custom exceptions
def validate_user_age(age: int) -> None:
    """Validate age with custom exception"""
    if age < 0:
        raise ValidationError("age", "Age cannot be negative")
    if age < 18:
        raise ValidationError("age", "Must be at least 18 years old")
    if age > 150:
        raise ValidationError("age", "Age seems unrealistic")


# =============================================================================
# 3. RESULT PATTERN (Type-safe error handling)
# =============================================================================

print("\n" + "=" * 70)
print("3. RESULT PATTERN")
print("=" * 70)

T = TypeVar('T')
E = TypeVar('E')


@dataclass
class Ok(Generic[T]):
    """Represents successful result"""
    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value

    def map(self, func: Callable[[T], Any]) -> 'Result':
        """Transform the value if Ok"""
        try:
            return Ok(func(self.value))
        except Exception as e:
            return Err(str(e))


@dataclass
class Err(Generic[E]):
    """Represents error result"""
    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self):
        raise RuntimeError(f"Called unwrap on Err: {self.error}")

    def unwrap_or(self, default):
        return default

    def map(self, func: Callable) -> 'Result':
        """Do nothing if Err"""
        return self


# Type alias for Result
Result = Ok[T] | Err[E]


def divide_result(a: float, b: float) -> Result[float, str]:
    """Division using Result pattern"""
    if b == 0:
        return Err("Division by zero")
    return Ok(a / b)


def parse_int_result(value: str) -> Result[int, str]:
    """Parse integer using Result pattern"""
    try:
        return Ok(int(value))
    except ValueError:
        return Err(f"Cannot parse '{value}' as integer")


def calculate_with_result(x: str, y: str) -> Result[float, str]:
    """
    Chain operations using Result pattern.
    No exceptions thrown - all errors are values!
    """
    # Parse first number
    x_result = parse_int_result(x)
    if x_result.is_err():
        return x_result

    # Parse second number
    y_result = parse_int_result(y)
    if y_result.is_err():
        return y_result

    # Divide
    return divide_result(float(x_result.unwrap()), float(y_result.unwrap()))


# =============================================================================
# 4. CONTEXT MANAGERS (Resource management)
# =============================================================================

print("\n" + "=" * 70)
print("4. CONTEXT MANAGERS")
print("=" * 70)


class DatabaseConnection:
    """Simulated database connection with context manager"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connected = False

    def __enter__(self):
        """Called when entering 'with' block"""
        print(f"Opening connection to {self.connection_string}")
        self.connected = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting 'with' block (even if exception)"""
        print(f"Closing connection to {self.connection_string}")
        self.connected = False

        # Return False to propagate exception, True to suppress
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_val}")
        return False  # Don't suppress exceptions

    def query(self, sql: str) -> str:
        if not self.connected:
            raise RuntimeError("Not connected to database")
        return f"Executing: {sql}"


@contextmanager
def timer(operation_name: str):
    """Context manager for timing operations"""
    start = time.time()
    print(f"Starting {operation_name}...")

    try:
        yield  # Code in 'with' block executes here
    finally:
        # Always executes, even if exception
        elapsed = time.time() - start
        print(f"{operation_name} took {elapsed:.4f} seconds")


@contextmanager
def error_handler(operation_name: str):
    """Context manager for error handling"""
    try:
        yield
    except Exception as e:
        print(f"Error in {operation_name}: {e}")
        # Could log, send alert, etc.
        raise  # Re-raise after handling


# =============================================================================
# 5. RETRY LOGIC
# =============================================================================

print("\n" + "=" * 70)
print("5. RETRY LOGIC")
print("=" * 70)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        print(
                            f"Attempt {attempt} failed: {e}. "
                            f"Retrying in {current_delay:.2f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        print(f"All {max_attempts} attempts failed")

            # All retries exhausted
            raise last_exception

        return wrapper

    return decorator


# Simulated unreliable function
@retry(max_attempts=3, delay=0.1, backoff=2.0)
def unreliable_network_call(success_rate: float = 0.3) -> str:
    """Simulates unreliable network call"""
    if random.random() > success_rate:
        raise ConnectionError("Network timeout")
    return "Success!"


# =============================================================================
# 6. ERROR RECOVERY STRATEGIES
# =============================================================================

print("\n" + "=" * 70)
print("6. ERROR RECOVERY STRATEGIES")
print("=" * 70)


class ErrorRecoveryStrategy(Enum):
    """Different error recovery strategies"""
    FAIL_FAST = "fail_fast"  # Fail immediately
    RETURN_DEFAULT = "return_default"  # Return default value
    RETRY = "retry"  # Retry operation
    LOG_AND_CONTINUE = "log_and_continue"  # Log error but continue


def process_items(
    items: list,
    processor: Callable,
    strategy: ErrorRecoveryStrategy = ErrorRecoveryStrategy.FAIL_FAST,
    default_value: Any = None
) -> list:
    """
    Process items with different error recovery strategies.
    """
    results = []

    for i, item in enumerate(items):
        try:
            result = processor(item)
            results.append(result)

        except Exception as e:
            if strategy == ErrorRecoveryStrategy.FAIL_FAST:
                # Stop processing immediately
                raise

            elif strategy == ErrorRecoveryStrategy.RETURN_DEFAULT:
                # Use default value for failed items
                print(f"Item {i} failed: {e}, using default value")
                results.append(default_value)

            elif strategy == ErrorRecoveryStrategy.LOG_AND_CONTINUE:
                # Log error and skip item
                print(f"Item {i} failed: {e}, skipping")
                continue

    return results


# =============================================================================
# DEMONSTRATIONS
# =============================================================================

def demonstrate_basic_exceptions():
    print("\n[BASIC EXCEPTION HANDLING]")
    print("-" * 50)

    # Normal division
    result = divide_with_error_handling(10, 2)
    print(f"10 / 2 = {result}")

    # Division by zero
    result = divide_with_error_handling(10, 0)
    print(f"Result: {result}")


def demonstrate_custom_exceptions():
    print("\n[CUSTOM EXCEPTIONS]")
    print("-" * 50)

    try:
        validate_user_age(25)
        print("✓ Age 25 is valid")
    except ValidationError as e:
        print(f"✗ {e}")

    try:
        validate_user_age(15)
    except ValidationError as e:
        print(f"✗ {e} (field: {e.field})")

    try:
        raise ResourceNotFoundError("User", "12345")
    except ResourceNotFoundError as e:
        print(f"✗ {e}")


def demonstrate_result_pattern():
    print("\n[RESULT PATTERN]")
    print("-" * 50)

    # Successful division
    result = divide_result(10, 2)
    if result.is_ok():
        print(f"Success: 10 / 2 = {result.unwrap()}")

    # Division by zero (no exception!)
    result = divide_result(10, 0)
    if result.is_err():
        print(f"Error: {result.error}")

    # Use unwrap_or for default value
    value = result.unwrap_or(0)
    print(f"Value or default: {value}")

    # Chaining operations
    result = calculate_with_result("10", "2")
    print(f"Calculate '10' / '2': {result.unwrap() if result.is_ok() else result.error}")

    result = calculate_with_result("10", "abc")
    print(f"Calculate '10' / 'abc': {result.error if result.is_err() else result.unwrap()}")


def demonstrate_context_managers():
    print("\n[CONTEXT MANAGERS]")
    print("-" * 50)

    # Database connection (resource cleanup guaranteed)
    print("Normal usage:")
    with DatabaseConnection("postgresql://localhost/mydb") as db:
        print(db.query("SELECT * FROM users"))
    print()

    # Even with exception, connection is closed
    print("With exception:")
    try:
        with DatabaseConnection("postgresql://localhost/mydb") as db:
            print(db.query("SELECT * FROM users"))
            raise ValueError("Something went wrong!")
    except ValueError:
        print("Exception caught, but connection was still closed\n")

    # Timer context manager
    with timer("Sleep operation"):
        time.sleep(0.1)


def demonstrate_retry():
    print("\n[RETRY LOGIC]")
    print("-" * 50)

    # Will retry on failure
    try:
        result = unreliable_network_call(success_rate=0.5)
        print(f"Result: {result}")
    except ConnectionError:
        print("Failed after all retries")


def demonstrate_recovery_strategies():
    print("\n[ERROR RECOVERY STRATEGIES]")
    print("-" * 50)

    items = [1, 2, "invalid", 4, 5]

    def process(x):
        return int(x) * 2

    # Fail fast
    print("Strategy: FAIL_FAST")
    try:
        results = process_items(items, process, ErrorRecoveryStrategy.FAIL_FAST)
    except ValueError as e:
        print(f"Stopped at error: {e}\n")

    # Return default
    print("Strategy: RETURN_DEFAULT")
    results = process_items(
        items, process,
        ErrorRecoveryStrategy.RETURN_DEFAULT,
        default_value=0
    )
    print(f"Results: {results}\n")

    # Log and continue
    print("Strategy: LOG_AND_CONTINUE")
    results = process_items(items, process, ErrorRecoveryStrategy.LOG_AND_CONTINUE)
    print(f"Results: {results}")


def print_summary():
    print("\n" + "=" * 70)
    print("ERROR HANDLING STRATEGIES SUMMARY")
    print("=" * 70)

    print("""
1. EXCEPTION HANDLING
   ✓ Use try/except for expected errors
   ✓ Use finally for cleanup
   ✓ Catch specific exceptions
   ✓ Don't catch Exception unless necessary

2. CUSTOM EXCEPTIONS
   ✓ Create hierarchy of exceptions
   ✓ Add context (fields, messages)
   ✓ Makes error handling more specific
   ✓ Better than generic exceptions

3. RESULT PATTERN
   ✓ Errors as values (no exceptions)
   ✓ Type-safe error handling
   ✓ Explicit error handling required
   ✓ Good for functional style

4. CONTEXT MANAGERS
   ✓ Guarantee resource cleanup
   ✓ Use 'with' statement
   ✓ Cleanup even with exceptions
   ✓ Makes resource management safe

5. RETRY LOGIC
   ✓ Handle transient failures
   ✓ Exponential backoff
   ✓ Configurable attempts
   ✓ Don't retry non-idempotent operations!

6. RECOVERY STRATEGIES
   ✓ Fail fast - stop immediately
   ✓ Return default - continue with fallback
   ✓ Log and continue - skip failed items
   ✓ Choose based on requirements

BEST PRACTICES:
  • Be specific with exceptions
  • Use context managers for resources
  • Don't silently ignore errors
  • Log errors appropriately
  • Fail fast when appropriate
  • Use Result pattern for predictable errors
  • Always clean up resources
  • Consider retry for transient failures
""")


if __name__ == "__main__":
    demonstrate_basic_exceptions()
    demonstrate_custom_exceptions()
    demonstrate_result_pattern()
    demonstrate_context_managers()
    demonstrate_retry()
    demonstrate_recovery_strategies()
    print_summary()
