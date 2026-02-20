"""
Relational Algebra Operations

Demonstrates fundamental relational algebra operations using SQL equivalents:
- Selection (σ): filter rows based on predicate
- Projection (π): select specific columns
- Cartesian Product (×): combine all rows from two relations
- Join (⋈): combine relations based on condition
- Union (∪): combine tuples from two relations
- Intersection (∩): common tuples
- Difference (−): tuples in first but not second
- Rename (ρ): rename relation/attributes
- Division (÷): find tuples that match all values in another relation

Theory:
- Relational algebra is procedural (how to compute)
- Relational calculus is declarative (what to compute)
- SQL is based on both but closer to relational algebra
- Operations are closed: output is always a relation
"""

import sqlite3
from typing import List, Tuple


def setup_database() -> sqlite3.Connection:
    """Create sample database for relational algebra operations."""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Students relation
    cursor.execute('''
        CREATE TABLE Students (
            student_id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            major TEXT
        )
    ''')

    # Courses relation
    cursor.execute('''
        CREATE TABLE Courses (
            course_id TEXT PRIMARY KEY,
            course_name TEXT,
            credits INTEGER
        )
    ''')

    # Enrollment relation
    cursor.execute('''
        CREATE TABLE Enrollment (
            student_id INTEGER,
            course_id TEXT,
            grade TEXT,
            PRIMARY KEY (student_id, course_id)
        )
    ''')

    # Insert sample data
    students = [
        (1, 'Alice', 20, 'CS'),
        (2, 'Bob', 21, 'Math'),
        (3, 'Charlie', 20, 'CS'),
        (4, 'Diana', 22, 'Physics'),
        (5, 'Eve', 21, 'Math')
    ]
    cursor.executemany("INSERT INTO Students VALUES (?, ?, ?, ?)", students)

    courses = [
        ('CS101', 'Intro to Programming', 3),
        ('CS201', 'Data Structures', 4),
        ('MATH101', 'Calculus I', 4),
        ('PHYS101', 'Physics I', 4)
    ]
    cursor.executemany("INSERT INTO Courses VALUES (?, ?, ?)", courses)

    enrollments = [
        (1, 'CS101', 'A'),
        (1, 'CS201', 'B'),
        (1, 'MATH101', 'A'),
        (2, 'MATH101', 'A'),
        (2, 'PHYS101', 'B'),
        (3, 'CS101', 'A'),
        (3, 'CS201', 'A'),
        (4, 'PHYS101', 'A'),
        (5, 'MATH101', 'B')
    ]
    cursor.executemany("INSERT INTO Enrollment VALUES (?, ?, ?)", enrollments)

    conn.commit()
    return conn


def selection(conn: sqlite3.Connection):
    """σ (sigma): Select rows matching a predicate."""
    print("=" * 60)
    print("SELECTION (σ)")
    print("=" * 60)
    print("Notation: σ_predicate(Relation)")
    print()

    cursor = conn.cursor()

    # σ_age>20(Students)
    print("1. σ_age>20(Students) - Students older than 20")
    print("-" * 60)
    cursor.execute("SELECT * FROM Students WHERE age > 20")
    for row in cursor.fetchall():
        print(f"  {row}")

    # σ_major='CS'(Students)
    print("\n2. σ_major='CS'(Students) - CS majors")
    print("-" * 60)
    cursor.execute("SELECT * FROM Students WHERE major = 'CS'")
    for row in cursor.fetchall():
        print(f"  {row}")

    # Complex predicate: σ_age=20 ∧ major='CS'(Students)
    print("\n3. σ_age=20 ∧ major='CS'(Students) - 20-year-old CS majors")
    print("-" * 60)
    cursor.execute("SELECT * FROM Students WHERE age = 20 AND major = 'CS'")
    for row in cursor.fetchall():
        print(f"  {row}")


def projection(conn: sqlite3.Connection):
    """π (pi): Select specific columns."""
    print("\n" + "=" * 60)
    print("PROJECTION (π)")
    print("=" * 60)
    print("Notation: π_attributes(Relation)")
    print()

    cursor = conn.cursor()

    # π_name,major(Students)
    print("1. π_name,major(Students) - Only names and majors")
    print("-" * 60)
    cursor.execute("SELECT name, major FROM Students")
    for row in cursor.fetchall():
        print(f"  {row}")

    # π_major(Students) - automatically removes duplicates
    print("\n2. π_major(Students) - Distinct majors (duplicates removed)")
    print("-" * 60)
    cursor.execute("SELECT DISTINCT major FROM Students")
    for row in cursor.fetchall():
        print(f"  {row[0]}")

    # Combined: π_name(σ_age>20(Students))
    print("\n3. π_name(σ_age>20(Students)) - Names of students older than 20")
    print("-" * 60)
    cursor.execute("SELECT name FROM Students WHERE age > 20")
    for row in cursor.fetchall():
        print(f"  {row[0]}")


def cartesian_product(conn: sqlite3.Connection):
    """× (times): Cartesian product of two relations."""
    print("\n" + "=" * 60)
    print("CARTESIAN PRODUCT (×)")
    print("=" * 60)
    print("Notation: R × S")
    print()

    cursor = conn.cursor()

    # Create small tables for demonstration
    cursor.execute("CREATE TEMP TABLE R (A INTEGER, B TEXT)")
    cursor.execute("CREATE TEMP TABLE S (C INTEGER, D TEXT)")
    cursor.execute("INSERT INTO R VALUES (1, 'a'), (2, 'b')")
    cursor.execute("INSERT INTO S VALUES (3, 'x'), (4, 'y')")

    print("Table R:")
    cursor.execute("SELECT * FROM R")
    for row in cursor.fetchall():
        print(f"  {row}")

    print("\nTable S:")
    cursor.execute("SELECT * FROM S")
    for row in cursor.fetchall():
        print(f"  {row}")

    print("\nR × S (Cartesian Product):")
    print("-" * 60)
    cursor.execute("SELECT * FROM R, S")  # Equivalent to: SELECT * FROM R CROSS JOIN S
    for row in cursor.fetchall():
        print(f"  {row}")

    print(f"\n|R| = 2, |S| = 2  →  |R × S| = 4")


def joins(conn: sqlite3.Connection):
    """⋈ (bowtie): Various join operations."""
    print("\n" + "=" * 60)
    print("JOIN OPERATIONS (⋈)")
    print("=" * 60)
    print("Notation: R ⋈_condition S")
    print()

    cursor = conn.cursor()

    # Natural join: Students ⋈ Enrollment
    print("1. Students ⋈ Enrollment (Natural Join on student_id)")
    print("-" * 60)
    cursor.execute("""
        SELECT S.student_id, S.name, E.course_id, E.grade
        FROM Students S
        JOIN Enrollment E ON S.student_id = E.student_id
        ORDER BY S.student_id
    """)
    for row in cursor.fetchall():
        print(f"  Student {row[0]} ({row[1]}): {row[2]} - Grade {row[3]}")

    # Theta join with condition
    print("\n2. Students ⋈_age≥22 Enrollment (Students aged 22+ with their enrollments)")
    print("-" * 60)
    cursor.execute("""
        SELECT S.name, S.age, E.course_id, E.grade
        FROM Students S
        JOIN Enrollment E ON S.student_id = E.student_id
        WHERE S.age >= 22
    """)
    for row in cursor.fetchall():
        print(f"  {row}")

    # Three-way join
    print("\n3. Students ⋈ Enrollment ⋈ Courses (Full enrollment details)")
    print("-" * 60)
    cursor.execute("""
        SELECT S.name, C.course_name, E.grade
        FROM Students S
        JOIN Enrollment E ON S.student_id = E.student_id
        JOIN Courses C ON E.course_id = C.course_id
        ORDER BY S.name
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:8} | {row[1]:25} | {row[2]}")

    # Left outer join (students who haven't enrolled in PHYS101)
    print("\n4. LEFT OUTER JOIN (All students, with PHYS101 grade if exists)")
    print("-" * 60)
    cursor.execute("""
        SELECT S.name,
               COALESCE(E.grade, 'Not Enrolled') as phys_grade
        FROM Students S
        LEFT JOIN Enrollment E
            ON S.student_id = E.student_id AND E.course_id = 'PHYS101'
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:8} | {row[1]}")


def set_operations(conn: sqlite3.Connection):
    """Union, Intersection, Difference."""
    print("\n" + "=" * 60)
    print("SET OPERATIONS (∪, ∩, −)")
    print("=" * 60)
    print()

    cursor = conn.cursor()

    # Union: Students taking CS101 OR CS201
    print("1. UNION (∪) - Students in CS101 ∪ Students in CS201")
    print("-" * 60)
    cursor.execute("""
        SELECT S.name, 'CS101' as course
        FROM Students S JOIN Enrollment E ON S.student_id = E.student_id
        WHERE E.course_id = 'CS101'
        UNION
        SELECT S.name, 'CS201'
        FROM Students S JOIN Enrollment E ON S.student_id = E.student_id
        WHERE E.course_id = 'CS201'
        ORDER BY name
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]} ({row[1]})")

    # Intersection: Students taking BOTH CS101 AND MATH101
    print("\n2. INTERSECTION (∩) - Students in CS101 ∩ Students in MATH101")
    print("-" * 60)
    cursor.execute("""
        SELECT S.name
        FROM Students S JOIN Enrollment E ON S.student_id = E.student_id
        WHERE E.course_id = 'CS101'
        INTERSECT
        SELECT S.name
        FROM Students S JOIN Enrollment E ON S.student_id = E.student_id
        WHERE E.course_id = 'MATH101'
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}")

    # Difference: CS majors NOT taking CS201
    print("\n3. DIFFERENCE (−) - CS majors − Students in CS201")
    print("-" * 60)
    cursor.execute("""
        SELECT name FROM Students WHERE major = 'CS'
        EXCEPT
        SELECT S.name
        FROM Students S JOIN Enrollment E ON S.student_id = E.student_id
        WHERE E.course_id = 'CS201'
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}")


def division_operation(conn: sqlite3.Connection):
    """÷ (division): Find tuples matching all values."""
    print("\n" + "=" * 60)
    print("DIVISION (÷)")
    print("=" * 60)
    print("Find students who took ALL CS courses")
    print()

    cursor = conn.cursor()

    # Find all CS courses
    print("1. All CS courses:")
    print("-" * 60)
    cursor.execute("SELECT course_id, course_name FROM Courses WHERE course_id LIKE 'CS%'")
    cs_courses = cursor.fetchall()
    for row in cs_courses:
        print(f"  {row[0]}: {row[1]}")

    # Division: Students ÷ CS_Courses
    # "Which students are enrolled in ALL CS courses?"
    print("\n2. Students enrolled in ALL CS courses:")
    print("-" * 60)
    cursor.execute("""
        SELECT S.student_id, S.name
        FROM Students S
        WHERE NOT EXISTS (
            -- No CS course exists that this student hasn't taken
            SELECT C.course_id
            FROM Courses C
            WHERE C.course_id LIKE 'CS%'
            AND NOT EXISTS (
                SELECT 1
                FROM Enrollment E
                WHERE E.student_id = S.student_id
                AND E.course_id = C.course_id
            )
        )
    """)
    result = cursor.fetchall()
    if result:
        for row in result:
            print(f"  Student {row[0]}: {row[1]}")
    else:
        print("  (None - no student has taken all CS courses)")

    # Alternative formulation using COUNT
    print("\n3. Alternative using COUNT (same result):")
    print("-" * 60)
    cursor.execute("""
        SELECT S.student_id, S.name
        FROM Students S
        WHERE (
            SELECT COUNT(DISTINCT E.course_id)
            FROM Enrollment E
            WHERE E.student_id = S.student_id
            AND E.course_id LIKE 'CS%'
        ) = (
            SELECT COUNT(*)
            FROM Courses
            WHERE course_id LIKE 'CS%'
        )
    """)
    for row in cursor.fetchall():
        print(f"  Student {row[0]}: {row[1]}")


def rename_operation(conn: sqlite3.Connection):
    """ρ (rho): Rename relations and attributes."""
    print("\n" + "=" * 60)
    print("RENAME (ρ)")
    print("=" * 60)
    print("Notation: ρ_new_name(old_name) or ρ_(A1,A2,...)(Relation)")
    print()

    cursor = conn.cursor()

    # Self-join requires renaming
    print("1. Self-join with aliases (find pairs of students with same age)")
    print("-" * 60)
    cursor.execute("""
        SELECT S1.name as student1, S2.name as student2, S1.age
        FROM Students S1
        JOIN Students S2 ON S1.age = S2.age AND S1.student_id < S2.student_id
        ORDER BY S1.age
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]} and {row[1]} are both {row[2]} years old")

    # Rename columns in result
    print("\n2. Column renaming (attribute renaming)")
    print("-" * 60)
    cursor.execute("""
        SELECT
            student_id AS id,
            name AS full_name,
            age AS years_old,
            major AS department
        FROM Students
        WHERE age = 20
    """)
    print("  " + str(tuple([desc[0] for desc in cursor.description])))
    for row in cursor.fetchall():
        print(f"  {row}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║              RELATIONAL ALGEBRA OPERATIONS                   ║
║  σ, π, ×, ⋈, ∪, ∩, −, ρ, ÷                                  ║
╚══════════════════════════════════════════════════════════════╝
""")

    conn = setup_database()

    try:
        selection(conn)
        projection(conn)
        cartesian_product(conn)
        joins(conn)
        set_operations(conn)
        division_operation(conn)
        rename_operation(conn)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("Relational algebra provides:")
        print("  σ (selection)   - Filter rows")
        print("  π (projection)  - Select columns")
        print("  × (product)     - Combine all pairs")
        print("  ⋈ (join)        - Combine on condition")
        print("  ∪ (union)       - Combine sets")
        print("  ∩ (intersect)   - Common elements")
        print("  − (difference)  - Elements in first but not second")
        print("  ÷ (division)    - Universal quantification")
        print("  ρ (rename)      - Rename relations/attributes")
        print("=" * 60)

    finally:
        conn.close()
