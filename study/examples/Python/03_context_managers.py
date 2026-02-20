"""
Python Context Managers

Demonstrates:
- with statement and context protocol
- __enter__ and __exit__ methods
- contextlib.contextmanager decorator
- contextlib.suppress
- contextlib.ExitStack
- Practical examples (timer, file transaction, resource management)
"""

import contextlib
import time
from typing import Any, Optional


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Basic Context Manager Protocol
# =============================================================================

section("Basic Context Manager Protocol")


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "block"):
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self):
        """Called when entering 'with' block."""
        print(f"  [{self.name}] Starting timer")
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting 'with' block."""
        end_time = time.perf_counter()
        self.elapsed = end_time - self.start_time
        print(f"  [{self.name}] Elapsed: {self.elapsed * 1000:.4f} ms")

        # Return False to propagate exceptions, True to suppress
        return False


with Timer("computation") as timer:
    print("    Doing some work...")
    time.sleep(0.1)
    result = sum(range(100000))

print(f"Timer object elapsed: {timer.elapsed * 1000:.4f} ms")


# =============================================================================
# Resource Management
# =============================================================================

section("Resource Management")


class ManagedFile:
    """Context manager for file-like resource."""

    def __init__(self, filename: str):
        self.filename = filename
        self.file = None
        print(f"  Initializing ManagedFile({filename})")

    def __enter__(self):
        print(f"  Opening {self.filename}")
        self.file = open(self.filename, 'w')
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            print(f"  Closing {self.filename}")
            self.file.close()

        if exc_type is not None:
            print(f"  Exception occurred: {exc_type.__name__}: {exc_val}")

        return False  # Don't suppress exceptions


# Temporary file for demo
import tempfile
import os

temp_path = os.path.join(tempfile.gettempdir(), "demo.txt")

with ManagedFile(temp_path) as f:
    f.write("Hello, context managers!\n")
    print(f"    Written to {temp_path}")

print(f"File closed automatically")


# =============================================================================
# contextlib.contextmanager Decorator
# =============================================================================

section("contextlib.contextmanager Decorator")


@contextlib.contextmanager
def simple_timer(name: str):
    """Simpler timer using contextmanager decorator."""
    print(f"  [{name}] Starting...")
    start = time.perf_counter()

    try:
        yield  # Control returns to with block here
    finally:
        # This runs when exiting with block
        elapsed = time.perf_counter() - start
        print(f"  [{name}] Finished in {elapsed * 1000:.4f} ms")


with simple_timer("generator-based"):
    print("    Executing code block...")
    time.sleep(0.05)


@contextlib.contextmanager
def atomic_write(filename: str):
    """Atomic file write - write to temp, then rename."""
    temp_file = filename + ".tmp"

    print(f"  Writing to temporary file: {temp_file}")
    f = open(temp_file, 'w')

    try:
        yield f
    except Exception:
        print(f"  Error occurred, removing temp file")
        f.close()
        os.remove(temp_file)
        raise
    else:
        print(f"  Success, committing to {filename}")
        f.close()
        os.rename(temp_file, filename)


atomic_path = os.path.join(tempfile.gettempdir(), "atomic.txt")

with atomic_write(atomic_path) as f:
    f.write("Data written atomically\n")
    print("    Writing data...")

print(f"File committed successfully")


# =============================================================================
# contextlib.suppress
# =============================================================================

section("contextlib.suppress")

print("Without suppress:")
try:
    with open("/nonexistent/file.txt") as f:
        pass
except FileNotFoundError as e:
    print(f"  Caught exception: {e.__class__.__name__}")

print("\nWith suppress:")
with contextlib.suppress(FileNotFoundError):
    with open("/nonexistent/file.txt") as f:
        pass
    print("  This won't print")

print("  No exception raised, execution continues")


# =============================================================================
# Nested Context Managers
# =============================================================================

section("Nested Context Managers")


@contextlib.contextmanager
def tag(name: str):
    """HTML tag context manager."""
    print(f"<{name}>", end="")
    yield
    print(f"</{name}>", end="")


print("Manual nesting:")
with tag("html"):
    with tag("body"):
        with tag("h1"):
            print("Hello", end="")
print()


# =============================================================================
# contextlib.ExitStack
# =============================================================================

section("contextlib.ExitStack")


@contextlib.contextmanager
def resource(name: str):
    """Simulate acquiring a resource."""
    print(f"  Acquiring {name}")
    yield name
    print(f"  Releasing {name}")


# Dynamically manage multiple context managers
with contextlib.ExitStack() as stack:
    resources = []
    for i in range(3):
        r = stack.enter_context(resource(f"Resource-{i}"))
        resources.append(r)

    print(f"  All resources acquired: {resources}")
    print(f"  Using resources...")

print("  All resources released in reverse order")


# =============================================================================
# Transaction Pattern
# =============================================================================

section("Transaction Pattern")


class Transaction:
    """Database-like transaction context manager."""

    def __init__(self, name: str):
        self.name = name
        self.committed = False

    def __enter__(self):
        print(f"  BEGIN TRANSACTION '{self.name}'")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self.committed:
            print(f"  COMMIT '{self.name}'")
        else:
            print(f"  ROLLBACK '{self.name}'")
            if exc_type:
                print(f"    Reason: {exc_type.__name__}: {exc_val}")
        return False

    def commit(self):
        """Mark transaction for commit."""
        self.committed = True


print("Successful transaction:")
with Transaction("transfer-funds") as txn:
    print("    Deducting from account A")
    print("    Adding to account B")
    txn.commit()

print("\nFailed transaction:")
try:
    with Transaction("invalid-transfer") as txn:
        print("    Deducting from account A")
        raise ValueError("Insufficient funds")
        txn.commit()  # Never reached
except ValueError:
    pass


# =============================================================================
# Reentrant Context Manager
# =============================================================================

section("Reentrant Context Manager")


class ReentrantResource:
    """Context manager that can be entered multiple times."""

    def __init__(self, name: str):
        self.name = name
        self.level = 0

    def __enter__(self):
        self.level += 1
        print(f"  [{self.name}] Enter level {self.level}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"  [{self.name}] Exit level {self.level}")
        self.level -= 1
        return False


resource = ReentrantResource("lock")

with resource:
    print("    Outer block")
    with resource:
        print("      Inner block")
        with resource:
            print("        Innermost block")


# =============================================================================
# Cleanup Actions
# =============================================================================

section("Cleanup Actions with ExitStack.callback")


with contextlib.ExitStack() as stack:
    # Register cleanup callbacks
    stack.callback(lambda: print("  Cleanup 1"))
    stack.callback(lambda: print("  Cleanup 2"))
    stack.callback(lambda: print("  Cleanup 3"))

    print("  Main code executing...")

print("  Callbacks executed in LIFO order")


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Context manager patterns covered:
1. __enter__/__exit__ protocol - manual implementation
2. contextlib.contextmanager - decorator-based approach
3. contextlib.suppress - ignore specific exceptions
4. contextlib.ExitStack - dynamic context manager composition
5. Timer pattern - measure execution time
6. Transaction pattern - commit/rollback semantics
7. Resource management - automatic cleanup
8. Atomic operations - all-or-nothing file writes

Context managers ensure proper resource cleanup and
provide clean syntax for setup/teardown operations.
""")

# Cleanup temp files
try:
    os.remove(temp_path)
    os.remove(atomic_path)
except:
    pass
