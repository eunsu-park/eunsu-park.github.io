"""
ER Model to Relational Schema Mapping

Demonstrates the 7-step ER-to-relational mapping algorithm:

1. Regular entity types → relations
2. Weak entity types → relations (include owner's PK)
3. 1:1 relationships → foreign key in either side
4. 1:N relationships → foreign key on N side
5. M:N relationships → separate relation with both PKs
6. Multivalued attributes → separate relation
7. N-ary relationships → separate relation with all PKs

Theory:
- ER model is conceptual (design phase)
- Relational model is logical (implementation phase)
- Mapping preserves semantics and constraints
- Choice of where to place FK in 1:1 depends on participation constraints
"""

import sqlite3
from datetime import date


def create_university_er_model(conn: sqlite3.Connection):
    """
    Create a university database following ER-to-relational mapping.

    ER Design:
    - Entities: Student, Course, Department, Professor
    - Relationships:
        * Student -[Enrolls]-> Course (M:N)
        * Department -[Offers]-> Course (1:N)
        * Professor -[Teaches]-> Course (1:N)
        * Professor -[Chairs]-> Department (1:1)
        * Student -[Advises]- Professor (M:N)
    - Weak Entity: Dependent (depends on Student)
    - Multivalued: Student.phone_numbers
    """

    cursor = conn.cursor()

    print("=" * 60)
    print("STEP 1: REGULAR ENTITY TYPES → RELATIONS")
    print("=" * 60)
    print()

    # Regular entity: Student
    print("Entity: Student(student_id, name, dob, gpa)")
    cursor.execute('''
        CREATE TABLE Student (
            student_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            dob DATE,
            gpa REAL CHECK(gpa BETWEEN 0.0 AND 4.0)
        )
    ''')
    print("✓ Created table Student\n")

    # Regular entity: Course
    print("Entity: Course(course_id, title, credits)")
    cursor.execute('''
        CREATE TABLE Course (
            course_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            credits INTEGER CHECK(credits > 0)
        )
    ''')
    print("✓ Created table Course\n")

    # Regular entity: Department
    print("Entity: Department(dept_id, name, building)")
    cursor.execute('''
        CREATE TABLE Department (
            dept_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            building TEXT
        )
    ''')
    print("✓ Created table Department\n")

    # Regular entity: Professor
    print("Entity: Professor(prof_id, name, salary, dept_id)")
    cursor.execute('''
        CREATE TABLE Professor (
            prof_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            salary REAL CHECK(salary > 0),
            dept_id INTEGER,
            FOREIGN KEY (dept_id) REFERENCES Department(dept_id)
        )
    ''')
    print("✓ Created table Professor\n")

    print("=" * 60)
    print("STEP 2: WEAK ENTITY TYPES → RELATIONS")
    print("=" * 60)
    print()

    # Weak entity: Dependent (depends on Student)
    # PK includes owner's key (student_id) + partial key (dependent_name)
    print("Weak Entity: Dependent(student_id, dependent_name, relationship)")
    cursor.execute('''
        CREATE TABLE Dependent (
            student_id INTEGER,
            dependent_name TEXT,
            relationship TEXT,  -- 'spouse', 'child', etc.
            PRIMARY KEY (student_id, dependent_name),
            FOREIGN KEY (student_id) REFERENCES Student(student_id)
                ON DELETE CASCADE
        )
    ''')
    print("✓ Created table Dependent (weak entity)")
    print("  - Composite PK: (student_id, dependent_name)")
    print("  - student_id is FK to Student\n")

    print("=" * 60)
    print("STEP 3: 1:1 RELATIONSHIPS → FOREIGN KEY")
    print("=" * 60)
    print()

    # 1:1 relationship: Professor chairs Department
    # FK can go on either side; we choose Professor side
    print("Relationship: Professor -[Chairs]-> Department (1:1)")
    cursor.execute('ALTER TABLE Department ADD COLUMN chair_id INTEGER')
    cursor.execute('''
        CREATE UNIQUE INDEX idx_dept_chair ON Department(chair_id)
    ''')
    print("✓ Added chair_id to Department (UNIQUE ensures 1:1)")
    print("  - Each department has at most one chair")
    print("  - Each professor chairs at most one department\n")

    print("=" * 60)
    print("STEP 4: 1:N RELATIONSHIPS → FOREIGN KEY ON N SIDE")
    print("=" * 60)
    print()

    # 1:N: Department offers many Courses
    print("Relationship: Department -[Offers]-> Course (1:N)")
    cursor.execute('ALTER TABLE Course ADD COLUMN dept_id INTEGER')
    cursor.execute('''
        CREATE INDEX idx_course_dept ON Course(dept_id)
    ''')
    print("✓ Added dept_id FK to Course")
    print("  - Each course belongs to one department")
    print("  - Each department offers many courses\n")

    # 1:N: Professor teaches many Courses
    print("Relationship: Professor -[Teaches]-> Course (1:N)")
    cursor.execute('ALTER TABLE Course ADD COLUMN prof_id INTEGER')
    cursor.execute('''
        CREATE INDEX idx_course_prof ON Course(prof_id)
    ''')
    print("✓ Added prof_id FK to Course")
    print("  - Each course is taught by one professor")
    print("  - Each professor teaches many courses\n")

    print("=" * 60)
    print("STEP 5: M:N RELATIONSHIPS → SEPARATE RELATION")
    print("=" * 60)
    print()

    # M:N: Students enroll in many Courses, Courses have many Students
    print("Relationship: Student -[Enrolls]-> Course (M:N)")
    cursor.execute('''
        CREATE TABLE Enrolls (
            student_id INTEGER,
            course_id TEXT,
            semester TEXT,
            grade TEXT,
            PRIMARY KEY (student_id, course_id, semester),
            FOREIGN KEY (student_id) REFERENCES Student(student_id),
            FOREIGN KEY (course_id) REFERENCES Course(course_id)
        )
    ''')
    print("✓ Created table Enrolls (relationship table)")
    print("  - PK: (student_id, course_id, semester)")
    print("  - Attributes: grade (relationship attribute)\n")

    # M:N: Professor advises many Students
    print("Relationship: Professor -[Advises]-> Student (M:N)")
    cursor.execute('''
        CREATE TABLE Advises (
            prof_id INTEGER,
            student_id INTEGER,
            start_date DATE,
            PRIMARY KEY (prof_id, student_id),
            FOREIGN KEY (prof_id) REFERENCES Professor(prof_id),
            FOREIGN KEY (student_id) REFERENCES Student(student_id)
        )
    ''')
    print("✓ Created table Advises (relationship table)")
    print("  - PK: (prof_id, student_id)")
    print("  - Attributes: start_date\n")

    print("=" * 60)
    print("STEP 6: MULTIVALUED ATTRIBUTES → SEPARATE RELATION")
    print("=" * 60)
    print()

    # Multivalued attribute: Student.phone_numbers
    print("Multivalued Attribute: Student.phone_numbers")
    cursor.execute('''
        CREATE TABLE StudentPhone (
            student_id INTEGER,
            phone_number TEXT,
            PRIMARY KEY (student_id, phone_number),
            FOREIGN KEY (student_id) REFERENCES Student(student_id)
                ON DELETE CASCADE
        )
    ''')
    print("✓ Created table StudentPhone")
    print("  - PK: (student_id, phone_number)")
    print("  - One row per phone number\n")

    conn.commit()


def insert_sample_data(conn: sqlite3.Connection):
    """Insert sample data to demonstrate the mapping."""
    cursor = conn.cursor()

    print("=" * 60)
    print("INSERTING SAMPLE DATA")
    print("=" * 60)
    print()

    # Departments
    cursor.execute("INSERT INTO Department (dept_id, name, building) VALUES (1, 'Computer Science', 'Gates')")
    cursor.execute("INSERT INTO Department (dept_id, name, building) VALUES (2, 'Mathematics', 'Eckhart')")

    # Professors
    cursor.execute("INSERT INTO Professor VALUES (101, 'Dr. Smith', 95000, 1)")
    cursor.execute("INSERT INTO Professor VALUES (102, 'Dr. Johnson', 90000, 1)")
    cursor.execute("INSERT INTO Professor VALUES (103, 'Dr. Williams', 85000, 2)")

    # Set department chairs (1:1 relationship)
    cursor.execute("UPDATE Department SET chair_id = 101 WHERE dept_id = 1")
    cursor.execute("UPDATE Department SET chair_id = 103 WHERE dept_id = 2")

    # Students
    cursor.execute("INSERT INTO Student VALUES (1001, 'Alice Brown', '2002-05-15', 3.8)")
    cursor.execute("INSERT INTO Student VALUES (1002, 'Bob Davis', '2001-09-20', 3.5)")
    cursor.execute("INSERT INTO Student VALUES (1003, 'Charlie Wilson', '2003-01-10', 3.9)")

    # Dependents (weak entity)
    cursor.execute("INSERT INTO Dependent VALUES (1001, 'Emma Brown', 'spouse')")
    cursor.execute("INSERT INTO Dependent VALUES (1002, 'Lily Davis', 'child')")

    # Student phones (multivalued attribute)
    cursor.execute("INSERT INTO StudentPhone VALUES (1001, '555-1234')")
    cursor.execute("INSERT INTO StudentPhone VALUES (1001, '555-5678')")
    cursor.execute("INSERT INTO StudentPhone VALUES (1002, '555-9999')")

    # Courses
    cursor.execute("INSERT INTO Course VALUES ('CS101', 'Intro to CS', 3, 1, 101)")
    cursor.execute("INSERT INTO Course VALUES ('CS201', 'Data Structures', 4, 1, 102)")
    cursor.execute("INSERT INTO Course VALUES ('MATH101', 'Calculus I', 4, 2, 103)")

    # Enrollments (M:N relationship)
    cursor.execute("INSERT INTO Enrolls VALUES (1001, 'CS101', 'Fall2023', 'A')")
    cursor.execute("INSERT INTO Enrolls VALUES (1001, 'MATH101', 'Fall2023', 'B')")
    cursor.execute("INSERT INTO Enrolls VALUES (1002, 'CS101', 'Fall2023', 'B')")
    cursor.execute("INSERT INTO Enrolls VALUES (1003, 'CS201', 'Spring2024', 'A')")

    # Advising (M:N relationship)
    cursor.execute("INSERT INTO Advises VALUES (101, 1001, '2023-09-01')")
    cursor.execute("INSERT INTO Advises VALUES (101, 1003, '2023-09-01')")
    cursor.execute("INSERT INTO Advises VALUES (102, 1002, '2023-09-01')")

    conn.commit()
    print("✓ Sample data inserted\n")


def demonstrate_queries(conn: sqlite3.Connection):
    """Demonstrate queries on the mapped relational schema."""
    cursor = conn.cursor()

    print("=" * 60)
    print("QUERY EXAMPLES")
    print("=" * 60)
    print()

    # Query 1: Show all relationships
    print("1. Students with their advisors (M:N relationship)")
    print("-" * 60)
    cursor.execute("""
        SELECT S.name, P.name, A.start_date
        FROM Student S
        JOIN Advises A ON S.student_id = A.student_id
        JOIN Professor P ON A.prof_id = P.prof_id
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]} advised by {row[1]} since {row[2]}")

    # Query 2: Weak entity access
    print("\n2. Students and their dependents (weak entity)")
    print("-" * 60)
    cursor.execute("""
        SELECT S.name, D.dependent_name, D.relationship
        FROM Student S
        JOIN Dependent D ON S.student_id = D.student_id
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}'s {row[2]}: {row[1]}")

    # Query 3: Multivalued attribute
    print("\n3. Students with multiple phone numbers (multivalued attribute)")
    print("-" * 60)
    cursor.execute("""
        SELECT S.name, GROUP_CONCAT(SP.phone_number, ', ') as phones
        FROM Student S
        JOIN StudentPhone SP ON S.student_id = SP.student_id
        GROUP BY S.student_id, S.name
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    # Query 4: 1:1 relationship
    print("\n4. Department chairs (1:1 relationship)")
    print("-" * 60)
    cursor.execute("""
        SELECT D.name as dept, P.name as chair
        FROM Department D
        LEFT JOIN Professor P ON D.chair_id = P.prof_id
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]} chaired by {row[1]}")

    # Query 5: 1:N relationships
    print("\n5. Courses with department and professor (1:N relationships)")
    print("-" * 60)
    cursor.execute("""
        SELECT C.course_id, C.title, D.name as dept, P.name as prof
        FROM Course C
        JOIN Department D ON C.dept_id = D.dept_id
        JOIN Professor P ON C.prof_id = P.prof_id
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]} ({row[1]})")
        print(f"    Offered by: {row[2]}")
        print(f"    Taught by: {row[3]}")

    # Query 6: M:N with attributes
    print("\n6. Student enrollments with grades (M:N with attributes)")
    print("-" * 60)
    cursor.execute("""
        SELECT S.name, C.course_id, E.semester, E.grade
        FROM Student S
        JOIN Enrolls E ON S.student_id = E.student_id
        JOIN Course C ON E.course_id = C.course_id
        ORDER BY S.name
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} ({row[2]}) - Grade: {row[3]}")


def demonstrate_cascade_delete(conn: sqlite3.Connection):
    """Demonstrate cascading delete for weak entities."""
    print("\n" + "=" * 60)
    print("WEAK ENTITY CASCADE DELETE")
    print("=" * 60)
    print()

    cursor = conn.cursor()

    # Show dependent before delete
    cursor.execute("SELECT * FROM Dependent WHERE student_id = 1001")
    print("Before deleting student 1001:")
    for row in cursor.fetchall():
        print(f"  Dependent: {row}")

    # Delete student (should cascade to dependent)
    cursor.execute("DELETE FROM Student WHERE student_id = 1001")

    cursor.execute("SELECT * FROM Dependent WHERE student_id = 1001")
    result = cursor.fetchall()
    print("\nAfter deleting student 1001:")
    if result:
        for row in result:
            print(f"  Dependent: {row}")
    else:
        print("  ✓ Dependent also deleted (CASCADE)")

    conn.commit()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          ER MODEL TO RELATIONAL SCHEMA MAPPING               ║
║  7-Step Algorithm: Entities → Relations                      ║
╚══════════════════════════════════════════════════════════════╝
""")

    conn = sqlite3.connect(':memory:')
    conn.execute('PRAGMA foreign_keys = ON')  # Enable FK constraints

    try:
        create_university_er_model(conn)
        insert_sample_data(conn)
        demonstrate_queries(conn)
        demonstrate_cascade_delete(conn)

        print("\n" + "=" * 60)
        print("SUMMARY OF ER-TO-RELATIONAL MAPPING")
        print("=" * 60)
        print("1. Regular entity    → Table with PK")
        print("2. Weak entity       → Table with owner's PK + partial key")
        print("3. 1:1 relationship  → FK in either table (UNIQUE)")
        print("4. 1:N relationship  → FK on N side")
        print("5. M:N relationship  → Separate table with both PKs")
        print("6. Multivalued attr  → Separate table")
        print("7. N-ary relation    → Table with all entity PKs")
        print("=" * 60)

    finally:
        conn.close()
