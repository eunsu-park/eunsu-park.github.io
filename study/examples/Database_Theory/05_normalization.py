"""
Normalization and Normal Forms

Demonstrates database normalization process:
- 1NF (First Normal Form): Atomic values, no repeating groups
- 2NF (Second Normal Form): 1NF + no partial dependencies
- 3NF (Third Normal Form): 2NF + no transitive dependencies
- BCNF (Boyce-Codd Normal Form): 3NF + every determinant is a candidate key
- Lossless-join decomposition
- Dependency-preserving decomposition

Theory:
- Normalization eliminates redundancy and update anomalies
- Higher normal forms reduce redundancy but may require more joins
- BCNF is stricter than 3NF (every FD X→Y requires X to be a superkey)
- 3NF preserves dependencies; BCNF may not
- Decomposition must be lossless and preferably dependency-preserving
"""

import sqlite3
from typing import Set, List, Tuple
from itertools import combinations


class Relation:
    """Represents a database relation with attributes and FDs."""

    def __init__(self, name: str, attributes: Set[str], fds: Set[Tuple[Set[str], Set[str]]]):
        self.name = name
        self.attributes = attributes
        self.fds = fds

    def __repr__(self):
        return f"{self.name}({', '.join(sorted(self.attributes))})"


def demonstrate_1nf():
    """Demonstrate First Normal Form (1NF)."""
    print("=" * 60)
    print("FIRST NORMAL FORM (1NF)")
    print("=" * 60)
    print("Requirements:")
    print("  1. Each attribute contains atomic (indivisible) values")
    print("  2. No repeating groups")
    print("  3. Each row is unique (has a primary key)")
    print()

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Violation: Non-atomic values
    print("VIOLATION 1: Non-atomic values")
    print("-" * 60)
    print("Employee(emp_id, name, phones)")
    print("Data:")
    print("  (1, 'Alice', '555-1234, 555-5678')  ← phones is non-atomic")
    print("  (2, 'Bob', '555-9999')")

    print("\n✓ Fix: Separate table for phones")
    print("Employee(emp_id, name)")
    print("Phone(emp_id, phone_number)")
    cursor.execute("CREATE TABLE Employee (emp_id INTEGER PRIMARY KEY, name TEXT)")
    cursor.execute("CREATE TABLE Phone (emp_id INTEGER, phone_number TEXT, PRIMARY KEY(emp_id, phone_number))")
    cursor.execute("INSERT INTO Employee VALUES (1, 'Alice'), (2, 'Bob')")
    cursor.execute("INSERT INTO Phone VALUES (1, '555-1234'), (1, '555-5678'), (2, '555-9999')")

    # Violation: Repeating groups
    print("\n\nVIOLATION 2: Repeating groups")
    print("-" * 60)
    print("Student(student_id, name, course1, course2, course3)")
    print("Data:")
    print("  (1, 'Alice', 'CS101', 'MATH101', NULL)  ← fixed number of courses")

    print("\n✓ Fix: Separate table for enrollments")
    print("Student(student_id, name)")
    print("Enrollment(student_id, course_id)")

    conn.close()
    print()


def compute_closure(attributes: Set[str], fds: Set[Tuple[Set[str], Set[str]]]) -> Set[str]:
    """Compute attribute closure."""
    closure = set(attributes)
    changed = True
    while changed:
        changed = False
        for lhs, rhs in fds:
            if set(lhs).issubset(closure):
                before = len(closure)
                closure.update(rhs)
                if len(closure) > before:
                    changed = True
    return closure


def find_candidate_keys(attributes: Set[str], fds: Set[Tuple[Set[str], Set[str]]]) -> List[Set[str]]:
    """Find all candidate keys."""
    # Attributes that never appear on RHS must be in every key
    rhs_attrs = set()
    for _, rhs in fds:
        rhs_attrs.update(rhs)
    must_include = attributes - rhs_attrs

    if compute_closure(must_include, fds) == attributes:
        return [must_include]

    keys = []
    other_attrs = list(attributes - must_include)

    for size in range(1, len(other_attrs) + 1):
        for combo in combinations(other_attrs, size):
            candidate = must_include | set(combo)
            if compute_closure(candidate, fds) == attributes:
                # Check minimality
                is_minimal = True
                for attr in combo:
                    subset = candidate - {attr}
                    if compute_closure(subset, fds) == attributes:
                        is_minimal = False
                        break
                if is_minimal:
                    keys.append(candidate)

    return keys if keys else [must_include]


def is_2nf(attributes: Set[str], fds: Set[Tuple[Set[str], Set[str]]], keys: List[Set[str]]) -> Tuple[bool, str]:
    """
    Check if relation is in 2NF.
    2NF: No partial dependencies (no non-prime attribute depends on proper subset of any candidate key)
    """
    # Find prime attributes (attributes in any candidate key)
    prime_attrs = set()
    for key in keys:
        prime_attrs.update(key)

    non_prime = attributes - prime_attrs

    # Check for partial dependencies
    for lhs, rhs in fds:
        lhs_set = set(lhs)
        rhs_set = set(rhs)

        # Check if RHS contains non-prime attributes
        non_prime_in_rhs = rhs_set & non_prime
        if not non_prime_in_rhs:
            continue

        # Check if LHS is a proper subset of any candidate key
        for key in keys:
            if lhs_set < key:  # Proper subset
                attrs = ', '.join(sorted(non_prime_in_rhs))
                return False, f"Partial dependency: {{{', '.join(sorted(lhs))}}} → {{{attrs}}} (part of key {{{', '.join(sorted(key))}}})"

    return True, "No partial dependencies"


def is_3nf(attributes: Set[str], fds: Set[Tuple[Set[str], Set[str]]], keys: List[Set[str]]) -> Tuple[bool, str]:
    """
    Check if relation is in 3NF.
    3NF: For every FD X→A, either:
      - A is in X (trivial), OR
      - X is a superkey, OR
      - A is a prime attribute
    """
    prime_attrs = set()
    for key in keys:
        prime_attrs.update(key)

    for lhs, rhs in fds:
        lhs_set = set(lhs)
        for attr in rhs:
            # Skip if trivial
            if attr in lhs_set:
                continue

            # Check if LHS is a superkey
            if compute_closure(lhs_set, fds) == attributes:
                continue

            # Check if attr is prime
            if attr not in prime_attrs:
                return False, f"Transitive dependency: {{{', '.join(sorted(lhs))}}} → {attr} (non-prime, LHS not superkey)"

    return True, "No transitive dependencies on non-prime attributes"


def is_bcnf(attributes: Set[str], fds: Set[Tuple[Set[str], Set[str]]]) -> Tuple[bool, str]:
    """
    Check if relation is in BCNF.
    BCNF: For every non-trivial FD X→Y, X must be a superkey
    """
    for lhs, rhs in fds:
        lhs_set = set(lhs)
        rhs_set = set(rhs)

        # Skip trivial dependencies
        if rhs_set.issubset(lhs_set):
            continue

        # Check if LHS is a superkey
        if compute_closure(lhs_set, fds) != attributes:
            return False, f"Violating FD: {{{', '.join(sorted(lhs))}}} → {{{', '.join(sorted(rhs))}}} (LHS not a superkey)"

    return True, "All determinants are superkeys"


def demonstrate_2nf():
    """Demonstrate Second Normal Form (2NF)."""
    print("=" * 60)
    print("SECOND NORMAL FORM (2NF)")
    print("=" * 60)
    print("Requirements: 1NF + No partial dependencies")
    print("(No non-prime attribute depends on part of a candidate key)")
    print()

    # Example: Student enrollments
    print("Example: StudentCourse(student_id, course_id, student_name, instructor)")
    print("-" * 60)
    attributes = {'student_id', 'course_id', 'student_name', 'instructor'}
    fds = {
        (frozenset({'student_id', 'course_id'}), frozenset({'instructor'})),
        (frozenset({'student_id'}), frozenset({'student_name'})),  # Partial dependency!
        (frozenset({'course_id'}), frozenset({'instructor'}))       # Partial dependency!
    }

    print("FDs:")
    print("  {student_id, course_id} → instructor")
    print("  student_id → student_name (partial!)")
    print("  course_id → instructor (partial!)")
    print()

    keys = find_candidate_keys(attributes, fds)
    print(f"Candidate key: {{{', '.join(sorted(keys[0]))}}}")

    is_2, reason = is_2nf(attributes, fds, keys)
    print(f"\n2NF: {'✓' if is_2 else '✗'} {reason}")

    print("\n✓ Fix: Decompose into 3 relations")
    print("  Student(student_id, student_name)")
    print("  Course(course_id, instructor)")
    print("  Enrollment(student_id, course_id)")
    print()


def demonstrate_3nf():
    """Demonstrate Third Normal Form (3NF)."""
    print("=" * 60)
    print("THIRD NORMAL FORM (3NF)")
    print("=" * 60)
    print("Requirements: 2NF + No transitive dependencies")
    print("(Non-prime attributes depend only on candidate keys)")
    print()

    # Example: Employee with department location
    print("Example: Employee(emp_id, dept_id, dept_location)")
    print("-" * 60)
    attributes = {'emp_id', 'dept_id', 'dept_location'}
    fds = {
        (frozenset({'emp_id'}), frozenset({'dept_id'})),
        (frozenset({'dept_id'}), frozenset({'dept_location'}))  # Transitive!
    }

    print("FDs:")
    print("  emp_id → dept_id")
    print("  dept_id → dept_location (transitive!)")
    print()

    keys = find_candidate_keys(attributes, fds)
    print(f"Candidate key: {{{', '.join(sorted(keys[0]))}}}")
    print("Prime attributes: {emp_id}")
    print("Non-prime attributes: {dept_id, dept_location}")

    is_3, reason = is_3nf(attributes, fds, keys)
    print(f"\n3NF: {'✓' if is_3 else '✗'} {reason}")

    print("\nExplanation:")
    print("  emp_id → dept_id → dept_location")
    print("  dept_location depends on emp_id through dept_id (transitive)")

    print("\n✓ Fix: Decompose into 2 relations")
    print("  Employee(emp_id, dept_id)")
    print("  Department(dept_id, dept_location)")
    print()


def demonstrate_bcnf():
    """Demonstrate Boyce-Codd Normal Form (BCNF)."""
    print("=" * 60)
    print("BOYCE-CODD NORMAL FORM (BCNF)")
    print("=" * 60)
    print("Requirements: For every non-trivial FD X→Y, X is a superkey")
    print()

    # Classic BCNF example: Course enrollment with time conflicts
    print("Example: CourseSchedule(student_id, course, instructor)")
    print("-" * 60)
    print("Constraint: Each instructor teaches only one course")
    print()

    attributes = {'student_id', 'course', 'instructor'}
    fds = {
        (frozenset({'student_id', 'course'}), frozenset({'instructor'})),
        (frozenset({'instructor'}), frozenset({'course'}))  # BCNF violation!
    }

    print("FDs:")
    print("  {student_id, course} → instructor")
    print("  instructor → course (BCNF violation!)")
    print()

    keys = find_candidate_keys(attributes, fds)
    print(f"Candidate key: {{{', '.join(sorted(keys[0]))}}}")

    # Check 3NF
    is_3, reason_3 = is_3nf(attributes, fds, keys)
    print(f"\n3NF: {'✓' if is_3 else '✗'}")
    if is_3:
        print("  (instructor is prime attribute, so 3NF is satisfied)")

    # Check BCNF
    is_b, reason_b = is_bcnf(attributes, fds)
    print(f"\nBCNF: {'✓' if is_b else '✗'} {reason_b}")

    print("\nExplanation:")
    print("  instructor → course, but instructor is not a superkey")
    print("  This causes redundancy: instructor→course mapping repeated")

    print("\n✓ Fix: Decompose into 2 relations")
    print("  Enrollment(student_id, instructor)")
    print("  Teaching(instructor, course)")
    print()


def demonstrate_lossless_join():
    """Demonstrate lossless-join decomposition."""
    print("=" * 60)
    print("LOSSLESS-JOIN DECOMPOSITION")
    print("=" * 60)
    print()

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Original relation
    cursor.execute('''
        CREATE TABLE Employee (
            emp_id INTEGER,
            dept_id INTEGER,
            dept_name TEXT,
            PRIMARY KEY (emp_id)
        )
    ''')

    cursor.execute("INSERT INTO Employee VALUES (1, 10, 'Engineering')")
    cursor.execute("INSERT INTO Employee VALUES (2, 10, 'Engineering')")
    cursor.execute("INSERT INTO Employee VALUES (3, 20, 'Sales')")

    print("Original relation: Employee(emp_id, dept_id, dept_name)")
    print("-" * 60)
    cursor.execute("SELECT * FROM Employee")
    for row in cursor.fetchall():
        print(f"  {row}")

    # Lossless decomposition
    print("\nLossless decomposition:")
    print("  R1(emp_id, dept_id)")
    print("  R2(dept_id, dept_name)")
    print()

    cursor.execute("CREATE TABLE R1 AS SELECT emp_id, dept_id FROM Employee")
    cursor.execute("CREATE TABLE R2 AS SELECT DISTINCT dept_id, dept_name FROM Employee")

    cursor.execute("SELECT * FROM R1")
    print("R1:")
    for row in cursor.fetchall():
        print(f"  {row}")

    cursor.execute("SELECT * FROM R2")
    print("\nR2:")
    for row in cursor.fetchall():
        print(f"  {row}")

    # Reconstruct
    cursor.execute("""
        SELECT R1.emp_id, R1.dept_id, R2.dept_name
        FROM R1 JOIN R2 ON R1.dept_id = R2.dept_id
    """)
    print("\nReconstructed (R1 ⋈ R2):")
    for row in cursor.fetchall():
        print(f"  {row}")

    print("\n✓ Lossless: Original = R1 ⋈ R2 (FD: dept_id → dept_name)")

    # Lossy decomposition example
    print("\n" + "-" * 60)
    print("Lossy decomposition (bad):")
    print("  S1(emp_id, dept_name)")
    print("  S2(dept_id, dept_name)")
    print()

    cursor.execute("CREATE TABLE S1 AS SELECT emp_id, dept_name FROM Employee")
    cursor.execute("CREATE TABLE S2 AS SELECT dept_id, dept_name FROM Employee")

    cursor.execute("""
        SELECT S1.emp_id, S2.dept_id, S1.dept_name
        FROM S1 JOIN S2 ON S1.dept_name = S2.dept_name
    """)
    print("Reconstructed (S1 ⋈ S2):")
    for row in cursor.fetchall():
        print(f"  {row}")

    print("\n✗ Lossy: Created spurious tuples (extra rows)")
    print("  (emp 1 and 2 both get dept_id 10 due to same dept_name)")

    conn.close()
    print()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║              DATABASE NORMALIZATION                          ║
║  1NF, 2NF, 3NF, BCNF, Lossless Decomposition                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    demonstrate_1nf()
    demonstrate_2nf()
    demonstrate_3nf()
    demonstrate_bcnf()
    demonstrate_lossless_join()

    print("=" * 60)
    print("SUMMARY OF NORMAL FORMS")
    print("=" * 60)
    print("1NF: Atomic values, no repeating groups")
    print("2NF: 1NF + no partial dependencies")
    print("3NF: 2NF + no transitive dependencies")
    print("BCNF: Every determinant is a superkey")
    print()
    print("Trade-offs:")
    print("  - Higher NF → less redundancy, more tables/joins")
    print("  - 3NF preserves dependencies, BCNF may not")
    print("  - Decomposition must be lossless")
    print("=" * 60)
