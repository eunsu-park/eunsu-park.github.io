"""
Transactions and ACID Properties

Demonstrates ACID properties of database transactions:
- Atomicity: All or nothing execution
- Consistency: Database moves from one valid state to another
- Isolation: Concurrent transactions don't interfere
- Durability: Committed changes persist

Theory:
- Transaction: logical unit of work (BEGIN...COMMIT/ROLLBACK)
- Atomicity via logging and rollback mechanisms
- Consistency enforced by constraints and triggers
- Isolation levels: READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE
- Durability via write-ahead logging (WAL)

Common anomalies:
- Dirty read: Reading uncommitted data
- Non-repeatable read: Same query returns different results
- Phantom read: New rows appear in range queries
"""

import sqlite3
import time
from threading import Thread
from typing import List


def demonstrate_atomicity():
    """Demonstrate atomicity: all-or-nothing execution."""
    print("=" * 60)
    print("ATOMICITY: All or Nothing")
    print("=" * 60)
    print()

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Setup: Bank accounts
    cursor.execute('''
        CREATE TABLE Accounts (
            account_id INTEGER PRIMARY KEY,
            balance REAL CHECK(balance >= 0)
        )
    ''')
    cursor.execute("INSERT INTO Accounts VALUES (1, 1000), (2, 500)")
    conn.commit()

    print("Initial state:")
    print("-" * 60)
    cursor.execute("SELECT * FROM Accounts")
    for row in cursor.fetchall():
        print(f"  Account {row[0]}: ${row[1]:.2f}")

    # Successful transaction
    print("\n\n1. SUCCESSFUL TRANSACTION: Transfer $200 from Account 1 to 2")
    print("-" * 60)
    try:
        cursor.execute("BEGIN TRANSACTION")
        cursor.execute("UPDATE Accounts SET balance = balance - 200 WHERE account_id = 1")
        cursor.execute("UPDATE Accounts SET balance = balance + 200 WHERE account_id = 2")
        cursor.execute("COMMIT")
        print("✓ Transaction committed")
    except sqlite3.Error as e:
        cursor.execute("ROLLBACK")
        print(f"✗ Transaction rolled back: {e}")

    cursor.execute("SELECT * FROM Accounts")
    print("\nAfter successful transfer:")
    for row in cursor.fetchall():
        print(f"  Account {row[0]}: ${row[1]:.2f}")

    # Failed transaction (constraint violation)
    print("\n\n2. FAILED TRANSACTION: Try to transfer $1000 from Account 1")
    print("-" * 60)
    print("   (Would violate CHECK constraint: balance >= 0)")
    try:
        cursor.execute("BEGIN TRANSACTION")
        cursor.execute("UPDATE Accounts SET balance = balance - 1000 WHERE account_id = 1")
        cursor.execute("UPDATE Accounts SET balance = balance + 1000 WHERE account_id = 2")
        cursor.execute("COMMIT")
        print("ERROR: Should have failed!")
    except sqlite3.IntegrityError as e:
        cursor.execute("ROLLBACK")
        print(f"✓ Transaction rolled back: {e}")

    cursor.execute("SELECT * FROM Accounts")
    print("\nAfter failed transfer (unchanged):")
    for row in cursor.fetchall():
        print(f"  Account {row[0]}: ${row[1]:.2f}")

    # Explicit rollback
    print("\n\n3. EXPLICIT ROLLBACK: Start transfer but cancel")
    print("-" * 60)
    cursor.execute("BEGIN TRANSACTION")
    cursor.execute("UPDATE Accounts SET balance = balance - 100 WHERE account_id = 1")
    print("Transfer started... changed Account 1 balance")
    cursor.execute("ROLLBACK")
    print("✓ Explicitly rolled back")

    cursor.execute("SELECT * FROM Accounts")
    print("\nAfter rollback (unchanged):")
    for row in cursor.fetchall():
        print(f"  Account {row[0]}: ${row[1]:.2f}")

    conn.close()
    print()


def demonstrate_consistency():
    """Demonstrate consistency: maintain invariants."""
    print("=" * 60)
    print("CONSISTENCY: Maintain Database Invariants")
    print("=" * 60)
    print()

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Setup: Inventory system
    cursor.execute('''
        CREATE TABLE Inventory (
            product_id INTEGER PRIMARY KEY,
            quantity INTEGER CHECK(quantity >= 0),
            reserved INTEGER DEFAULT 0 CHECK(reserved >= 0),
            CHECK(reserved <= quantity)
        )
    ''')
    cursor.execute("INSERT INTO Inventory VALUES (1, 100, 0)")
    conn.commit()

    print("Invariant: reserved <= quantity (can't reserve more than available)")
    print("-" * 60)

    print("\nInitial state:")
    cursor.execute("SELECT * FROM Inventory")
    print(f"  Product 1: quantity={cursor.fetchone()[1]}, reserved=0")

    # Valid reservation
    print("\n1. Valid operation: Reserve 30 units")
    print("-" * 60)
    try:
        cursor.execute("BEGIN TRANSACTION")
        cursor.execute("UPDATE Inventory SET reserved = 30 WHERE product_id = 1")
        cursor.execute("COMMIT")
        print("✓ Reservation successful")
    except sqlite3.IntegrityError as e:
        cursor.execute("ROLLBACK")
        print(f"✗ Failed: {e}")

    cursor.execute("SELECT quantity, reserved FROM Inventory WHERE product_id = 1")
    qty, res = cursor.fetchone()
    print(f"  State: quantity={qty}, reserved={res}")

    # Invalid reservation
    print("\n2. Invalid operation: Try to reserve 150 units (more than available)")
    print("-" * 60)
    try:
        cursor.execute("BEGIN TRANSACTION")
        cursor.execute("UPDATE Inventory SET reserved = 150 WHERE product_id = 1")
        cursor.execute("COMMIT")
        print("ERROR: Should have failed!")
    except sqlite3.IntegrityError as e:
        cursor.execute("ROLLBACK")
        print(f"✓ Constraint violation prevented: {e}")

    cursor.execute("SELECT quantity, reserved FROM Inventory WHERE product_id = 1")
    qty, res = cursor.fetchone()
    print(f"  State: quantity={qty}, reserved={res} (unchanged)")

    # Using triggers for complex constraints
    print("\n3. Trigger for complex consistency rule")
    print("-" * 60)
    cursor.execute('''
        CREATE TRIGGER validate_fulfillment
        BEFORE UPDATE OF quantity ON Inventory
        BEGIN
            SELECT RAISE(ABORT, 'Cannot reduce quantity below reserved')
            WHERE NEW.quantity < OLD.reserved;
        END
    ''')

    print("Trigger: Cannot reduce quantity below reserved amount")
    try:
        cursor.execute("UPDATE Inventory SET quantity = 20 WHERE product_id = 1")
        print("ERROR: Should have failed!")
    except sqlite3.IntegrityError as e:
        print(f"✓ Trigger prevented inconsistency: {e}")

    conn.close()
    print()


def demonstrate_isolation_levels():
    """Demonstrate isolation levels and read phenomena."""
    print("=" * 60)
    print("ISOLATION: Concurrent Transaction Isolation")
    print("=" * 60)
    print()

    print("Isolation Levels (from least to most strict):")
    print("-" * 60)
    print("1. READ UNCOMMITTED: Allows dirty reads")
    print("2. READ COMMITTED:   Prevents dirty reads")
    print("3. REPEATABLE READ:  Prevents non-repeatable reads")
    print("4. SERIALIZABLE:     Prevents phantom reads")
    print()

    # SQLite uses SERIALIZABLE by default when WAL mode is enabled
    print("SQLite Isolation:")
    print("-" * 60)
    print("Default: SERIALIZABLE (most strict)")
    print("Writers block writers, readers don't block writers (MVCC)")
    print()

    conn1 = sqlite3.connect(':memory:')
    conn2 = sqlite3.connect(':memory:')

    # Setup
    c1 = conn1.cursor()
    c1.execute("CREATE TABLE Data (id INTEGER PRIMARY KEY, value INTEGER)")
    c1.execute("INSERT INTO Data VALUES (1, 100)")
    conn1.commit()

    # Share database (in real scenario, both would connect to same file)
    # For demonstration, we'll show concepts

    print("Scenario: Dirty Read Prevention")
    print("-" * 60)
    print("Transaction 1: BEGIN")
    print("Transaction 1: UPDATE Data SET value = 200 WHERE id = 1")
    print("Transaction 2: SELECT value FROM Data WHERE id = 1")
    print()
    print("READ UNCOMMITTED: T2 would see 200 (dirty read)")
    print("READ COMMITTED:   T2 would see 100 (prevents dirty read)")
    print("✓ SQLite prevents dirty reads")
    print()

    print("Scenario: Non-Repeatable Read")
    print("-" * 60)
    print("Transaction 1: SELECT value FROM Data WHERE id = 1  → 100")
    print("Transaction 2: UPDATE Data SET value = 200 WHERE id = 1; COMMIT")
    print("Transaction 1: SELECT value FROM Data WHERE id = 1  → ???")
    print()
    print("READ COMMITTED:   T1 would see 200 (non-repeatable read)")
    print("REPEATABLE READ:  T1 would see 100 (snapshot isolation)")
    print("✓ SQLite provides snapshot isolation")

    conn1.close()
    conn2.close()
    print()


def demonstrate_durability():
    """Demonstrate durability: persistence of committed changes."""
    print("=" * 60)
    print("DURABILITY: Persistence After Commit")
    print("=" * 60)
    print()

    import os
    import tempfile

    # Create temporary database file
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    try:
        # Write data
        print("1. Writing data to database file")
        print("-" * 60)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE Logs (id INTEGER PRIMARY KEY, message TEXT, timestamp REAL)")
        cursor.execute("INSERT INTO Logs VALUES (1, 'System started', ?)", (time.time(),))
        conn.commit()
        print(f"✓ Committed 1 log entry to {db_path}")
        conn.close()

        # Simulate crash and recovery
        print("\n2. Simulating database crash and recovery")
        print("-" * 60)
        print("Closing connection (simulating crash)...")
        time.sleep(0.1)
        print("Reopening database...")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Logs")
        rows = cursor.fetchall()
        print(f"✓ Recovered {len(rows)} log entries after 'crash'")
        for row in rows:
            print(f"  {row}")

        # WAL mode for better durability
        print("\n3. Write-Ahead Logging (WAL) mode")
        print("-" * 60)
        cursor.execute("PRAGMA journal_mode=WAL")
        print("✓ Enabled WAL mode")
        print("  Benefits:")
        print("  - Readers don't block writers")
        print("  - Better crash recovery")
        print("  - More durable commits")

        cursor.execute("INSERT INTO Logs VALUES (2, 'WAL enabled', ?)", (time.time(),))
        conn.commit()
        conn.close()

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
        wal_path = db_path + '-wal'
        if os.path.exists(wal_path):
            os.unlink(wal_path)
        shm_path = db_path + '-shm'
        if os.path.exists(shm_path):
            os.unlink(shm_path)

    print()


def demonstrate_transaction_problems():
    """Demonstrate problems without proper transaction handling."""
    print("=" * 60)
    print("PROBLEMS WITHOUT TRANSACTIONS")
    print("=" * 60)
    print()

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    cursor.execute("CREATE TABLE Counter (value INTEGER)")
    cursor.execute("INSERT INTO Counter VALUES (0)")
    conn.commit()

    print("Problem: Lost Update")
    print("-" * 60)
    print("Without proper locking, concurrent updates can be lost")
    print()

    # Read-modify-write without transaction
    print("Bad approach (no transaction):")
    cursor.execute("SELECT value FROM Counter")
    value = cursor.fetchone()[0]
    print(f"  Read value: {value}")
    print(f"  Compute new value: {value + 1}")
    # Another transaction could modify value here!
    cursor.execute("UPDATE Counter SET value = ?", (value + 1,))
    conn.commit()
    print("  ✗ Lost update possible if concurrent access")

    print("\n✓ Good approach (atomic update in transaction):")
    cursor.execute("BEGIN TRANSACTION")
    cursor.execute("UPDATE Counter SET value = value + 1")
    cursor.execute("COMMIT")
    print("  Use UPDATE with expression (atomic)")

    cursor.execute("SELECT value FROM Counter")
    print(f"  Final value: {cursor.fetchone()[0]}")

    conn.close()
    print()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║            TRANSACTIONS AND ACID PROPERTIES                  ║
║  Atomicity, Consistency, Isolation, Durability               ║
╚══════════════════════════════════════════════════════════════╝
""")

    demonstrate_atomicity()
    demonstrate_consistency()
    demonstrate_isolation_levels()
    demonstrate_durability()
    demonstrate_transaction_problems()

    print("=" * 60)
    print("SUMMARY: ACID PROPERTIES")
    print("=" * 60)
    print("Atomicity:    All or nothing (COMMIT/ROLLBACK)")
    print("Consistency:  Maintain invariants (constraints, triggers)")
    print("Isolation:    Concurrent transactions don't interfere")
    print("Durability:   Committed changes persist (WAL)")
    print()
    print("Best Practices:")
    print("  ✓ Use transactions for multi-step operations")
    print("  ✓ Keep transactions short")
    print("  ✓ Use appropriate isolation level")
    print("  ✓ Handle errors with ROLLBACK")
    print("=" * 60)
