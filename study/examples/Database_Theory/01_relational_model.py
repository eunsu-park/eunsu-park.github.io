"""
Relational Model Fundamentals

Demonstrates core concepts of the relational model:
- Relations (tables) with domains, tuples (rows), attributes (columns)
- Keys: superkey, candidate key, primary key, foreign key
- Integrity constraints: entity integrity, referential integrity
- NULL handling and 3-valued logic (TRUE, FALSE, UNKNOWN)

Theory:
- A relation is a subset of the Cartesian product of domains
- Entity integrity: Primary key cannot be NULL
- Referential integrity: Foreign key must reference existing primary key or be NULL
- 3-valued logic: NULL comparisons yield UNKNOWN, affecting WHERE clause results
"""

import sqlite3
from typing import List, Tuple


def create_database() -> sqlite3.Connection:
    """Create an in-memory database with proper constraints."""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Domain constraints enforced through data types and CHECK constraints
    cursor.execute('''
        CREATE TABLE Departments (
            dept_id INTEGER PRIMARY KEY,  -- Entity integrity: NOT NULL implicit
            dept_name TEXT NOT NULL UNIQUE,  -- Candidate key
            budget REAL CHECK(budget >= 0),  -- Domain constraint
            manager_id INTEGER  -- Can be NULL (no manager yet)
        )
    ''')

    cursor.execute('''
        CREATE TABLE Employees (
            emp_id INTEGER PRIMARY KEY,  -- Entity integrity
            emp_name TEXT NOT NULL,
            salary REAL CHECK(salary > 0),  -- Domain constraint
            dept_id INTEGER,  -- Foreign key (can be NULL for unassigned employees)
            hire_date TEXT,  -- ISO format: YYYY-MM-DD
            FOREIGN KEY (dept_id) REFERENCES Departments(dept_id)
                ON DELETE SET NULL  -- Referential integrity action
                ON UPDATE CASCADE
        )
    ''')

    # Self-referencing foreign key (employee reports to another employee)
    cursor.execute('''
        CREATE TABLE ReportsTo (
            emp_id INTEGER PRIMARY KEY,
            manager_id INTEGER,
            FOREIGN KEY (emp_id) REFERENCES Employees(emp_id)
                ON DELETE CASCADE,
            FOREIGN KEY (manager_id) REFERENCES Employees(emp_id)
                ON DELETE SET NULL
        )
    ''')

    conn.commit()
    return conn


def demonstrate_keys(conn: sqlite3.Connection):
    """Demonstrate different types of keys."""
    print("=" * 60)
    print("KEYS AND CONSTRAINTS")
    print("=" * 60)

    cursor = conn.cursor()

    # Insert departments
    cursor.execute("INSERT INTO Departments VALUES (1, 'Engineering', 500000, NULL)")
    cursor.execute("INSERT INTO Departments VALUES (2, 'Sales', 300000, NULL)")
    cursor.execute("INSERT INTO Departments VALUES (3, 'HR', 150000, NULL)")

    # Try to violate primary key constraint
    print("\n1. PRIMARY KEY (Entity Integrity)")
    print("-" * 60)
    try:
        cursor.execute("INSERT INTO Departments VALUES (1, 'Marketing', 200000, NULL)")
        print("ERROR: Should have failed (duplicate primary key)")
    except sqlite3.IntegrityError as e:
        print(f"✓ Primary key violation caught: {e}")

    # Try to insert NULL primary key
    try:
        cursor.execute("INSERT INTO Departments VALUES (NULL, 'Marketing', 200000, NULL)")
        print("ERROR: Should have failed (NULL primary key)")
    except sqlite3.IntegrityError as e:
        print(f"✓ NULL primary key violation caught: {e}")

    # Candidate key (UNIQUE constraint)
    print("\n2. CANDIDATE KEY (dept_name is UNIQUE)")
    print("-" * 60)
    try:
        cursor.execute("INSERT INTO Departments VALUES (4, 'Engineering', 100000, NULL)")
        print("ERROR: Should have failed (duplicate candidate key)")
    except sqlite3.IntegrityError as e:
        print(f"✓ Candidate key violation caught: {e}")

    conn.commit()


def demonstrate_referential_integrity(conn: sqlite3.Connection):
    """Demonstrate referential integrity constraints."""
    print("\n" + "=" * 60)
    print("REFERENTIAL INTEGRITY")
    print("=" * 60)

    cursor = conn.cursor()

    # Insert employees
    cursor.execute("INSERT INTO Employees VALUES (101, 'Alice', 90000, 1, '2020-01-15')")
    cursor.execute("INSERT INTO Employees VALUES (102, 'Bob', 85000, 1, '2020-03-10')")
    cursor.execute("INSERT INTO Employees VALUES (103, 'Charlie', 70000, 2, '2021-06-01')")
    cursor.execute("INSERT INTO Employees VALUES (104, 'Diana', 65000, NULL, '2022-02-14')")  # NULL dept_id allowed

    # Try to violate foreign key constraint
    print("\n1. FOREIGN KEY Constraint")
    print("-" * 60)
    try:
        cursor.execute("INSERT INTO Employees VALUES (105, 'Eve', 75000, 999, '2022-05-20')")
        print("ERROR: Should have failed (invalid foreign key)")
    except sqlite3.IntegrityError as e:
        print(f"✓ Foreign key violation caught: {e}")

    # Test ON DELETE SET NULL
    print("\n2. ON DELETE SET NULL")
    print("-" * 60)
    print("Before delete:")
    cursor.execute("SELECT emp_id, emp_name, dept_id FROM Employees WHERE dept_id = 2")
    print(f"  Employee in dept 2: {cursor.fetchall()}")

    cursor.execute("DELETE FROM Departments WHERE dept_id = 2")
    cursor.execute("SELECT emp_id, emp_name, dept_id FROM Employees WHERE emp_id = 103")
    print(f"After deleting dept 2:")
    print(f"  Employee 103 dept_id: {cursor.fetchone()}")

    # Test ON UPDATE CASCADE
    print("\n3. ON UPDATE CASCADE")
    print("-" * 60)
    cursor.execute("UPDATE Departments SET dept_id = 10 WHERE dept_id = 1")
    cursor.execute("SELECT emp_id, emp_name, dept_id FROM Employees WHERE emp_id IN (101, 102)")
    print("After updating dept_id 1 → 10:")
    for row in cursor.fetchall():
        print(f"  Emp {row[0]} ({row[1]}): dept_id = {row[2]}")

    conn.commit()


def demonstrate_null_logic(conn: sqlite3.Connection):
    """Demonstrate 3-valued logic with NULL."""
    print("\n" + "=" * 60)
    print("NULL HANDLING AND 3-VALUED LOGIC")
    print("=" * 60)

    cursor = conn.cursor()

    # Show employees with their dept_id
    print("\n1. Current employees:")
    print("-" * 60)
    cursor.execute("SELECT emp_id, emp_name, dept_id FROM Employees")
    for row in cursor.fetchall():
        print(f"  Emp {row[0]} ({row[1]}): dept_id = {row[2]}")

    # NULL comparisons
    print("\n2. NULL Comparisons (3-valued logic)")
    print("-" * 60)

    # This returns only rows where dept_id is not NULL and equals 10
    cursor.execute("SELECT COUNT(*) FROM Employees WHERE dept_id = 10")
    print(f"  dept_id = 10: {cursor.fetchone()[0]} rows")

    # This returns only rows where dept_id is not NULL and not equals 10
    cursor.execute("SELECT COUNT(*) FROM Employees WHERE dept_id != 10")
    print(f"  dept_id != 10: {cursor.fetchone()[0]} rows")

    # NULL is neither equal nor not equal to anything
    cursor.execute("SELECT COUNT(*) FROM Employees WHERE dept_id IS NULL")
    null_count = cursor.fetchone()[0]
    print(f"  dept_id IS NULL: {null_count} rows")

    cursor.execute("SELECT COUNT(*) FROM Employees WHERE dept_id IS NOT NULL")
    not_null_count = cursor.fetchone()[0]
    print(f"  dept_id IS NOT NULL: {not_null_count} rows")

    cursor.execute("SELECT COUNT(*) FROM Employees")
    total = cursor.fetchone()[0]
    print(f"  Total rows: {total}")
    print(f"  ✓ NULL + NOT NULL = Total: {null_count + not_null_count} = {total}")

    # COALESCE for NULL handling
    print("\n3. COALESCE for NULL handling")
    print("-" * 60)
    cursor.execute("""
        SELECT emp_id, emp_name,
               COALESCE(dept_id, -1) as dept_id_or_default
        FROM Employees
    """)
    for row in cursor.fetchall():
        print(f"  Emp {row[0]} ({row[1]}): dept_id = {row[2]} (-1 means unassigned)")


def demonstrate_domains(conn: sqlite3.Connection):
    """Demonstrate domain constraints."""
    print("\n" + "=" * 60)
    print("DOMAIN CONSTRAINTS")
    print("=" * 60)

    cursor = conn.cursor()

    print("\n1. CHECK Constraint (salary > 0)")
    print("-" * 60)
    try:
        cursor.execute("INSERT INTO Employees VALUES (106, 'Frank', -1000, 10, '2023-01-01')")
        print("ERROR: Should have failed (negative salary)")
    except sqlite3.IntegrityError as e:
        print(f"✓ Domain constraint violation: {e}")

    print("\n2. NOT NULL Constraint")
    print("-" * 60)
    try:
        cursor.execute("INSERT INTO Employees VALUES (106, NULL, 75000, 10, '2023-01-01')")
        print("ERROR: Should have failed (NULL name)")
    except sqlite3.IntegrityError as e:
        print(f"✓ NOT NULL constraint violation: {e}")

    print("\n3. Type Affinity (SQLite feature)")
    print("-" * 60)
    # SQLite has type affinity, not strict typing
    cursor.execute("INSERT INTO Employees VALUES (106, 'Frank', '75000', 10, '2023-01-01')")
    cursor.execute("SELECT emp_id, salary, typeof(salary) FROM Employees WHERE emp_id = 106")
    row = cursor.fetchone()
    print(f"Inserted salary as string '75000':")
    print(f"  Retrieved value: {row[1]}, type: {row[2]}")
    print("  (SQLite converts to REAL due to column affinity)")

    conn.commit()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           RELATIONAL MODEL FUNDAMENTALS                      ║
║  Domains, Keys, Integrity Constraints, NULL Logic            ║
╚══════════════════════════════════════════════════════════════╝
""")

    conn = create_database()

    try:
        demonstrate_keys(conn)
        demonstrate_referential_integrity(conn)
        demonstrate_null_logic(conn)
        demonstrate_domains(conn)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✓ Relational model enforces data integrity through:")
        print("  - Entity integrity (primary keys)")
        print("  - Referential integrity (foreign keys)")
        print("  - Domain constraints (types, CHECK, NOT NULL)")
        print("  - 3-valued logic for NULL handling")
        print("=" * 60)

    finally:
        conn.close()
