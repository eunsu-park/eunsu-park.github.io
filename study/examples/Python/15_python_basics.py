"""
Python Basics

Demonstrates:
- Variables and data types
- Operators
- Control flow (if, for, while)
- Functions
- String formatting
- Lists, tuples, sets, dictionaries
- List/dict/set comprehensions
- File I/O
- Exception handling
"""


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Variables and Data Types
# =============================================================================

section("Variables and Data Types")

# Numbers
integer_num = 42
float_num = 3.14
complex_num = 3 + 4j

print(f"Integer: {integer_num} (type: {type(integer_num).__name__})")
print(f"Float: {float_num} (type: {type(float_num).__name__})")
print(f"Complex: {complex_num} (type: {type(complex_num).__name__})")

# Strings
single_quote = 'Hello'
double_quote = "World"
triple_quote = """Multi-line
string"""

print(f"\nStrings:")
print(f"  Single: {single_quote}")
print(f"  Double: {double_quote}")
print(f"  Triple: {triple_quote}")

# Booleans
is_active = True
is_deleted = False
print(f"\nBooleans: {is_active}, {is_deleted}")

# None
nothing = None
print(f"None: {nothing}")


# =============================================================================
# Operators
# =============================================================================

section("Operators")

# Arithmetic
a, b = 10, 3
print(f"a = {a}, b = {b}")
print(f"  Addition: {a} + {b} = {a + b}")
print(f"  Subtraction: {a} - {b} = {a - b}")
print(f"  Multiplication: {a} * {b} = {a * b}")
print(f"  Division: {a} / {b} = {a / b}")
print(f"  Floor division: {a} // {b} = {a // b}")
print(f"  Modulus: {a} % {b} = {a % b}")
print(f"  Exponentiation: {a} ** {b} = {a ** b}")

# Comparison
print(f"\nComparison:")
print(f"  {a} > {b}: {a > b}")
print(f"  {a} < {b}: {a < b}")
print(f"  {a} == {b}: {a == b}")
print(f"  {a} != {b}: {a != b}")

# Logical
x, y = True, False
print(f"\nLogical (x={x}, y={y}):")
print(f"  x and y: {x and y}")
print(f"  x or y: {x or y}")
print(f"  not x: {not x}")


# =============================================================================
# Control Flow - if/elif/else
# =============================================================================

section("Control Flow - if/elif/else")

age = 25

if age < 18:
    status = "minor"
elif age < 65:
    status = "adult"
else:
    status = "senior"

print(f"Age {age}: {status}")

# Ternary operator
message = "even" if age % 2 == 0 else "odd"
print(f"Age is {message}")


# =============================================================================
# Control Flow - for loop
# =============================================================================

section("Control Flow - for loop")

# Iterate over range
print("Range(5):")
for i in range(5):
    print(f"  {i}", end=" ")
print()

# Iterate over list
fruits = ["apple", "banana", "cherry"]
print("\nFruits:")
for fruit in fruits:
    print(f"  {fruit}")

# Enumerate
print("\nEnumerated:")
for index, fruit in enumerate(fruits):
    print(f"  {index}: {fruit}")

# Iterate over dictionary
person = {"name": "Alice", "age": 30, "city": "NYC"}
print("\nDictionary:")
for key, value in person.items():
    print(f"  {key}: {value}")


# =============================================================================
# Control Flow - while loop
# =============================================================================

section("Control Flow - while loop")

count = 0
print("Counting to 5:")
while count < 5:
    print(f"  Count: {count}")
    count += 1

# break and continue
print("\nWith break and continue:")
for i in range(10):
    if i == 3:
        continue  # Skip 3
    if i == 7:
        break  # Stop at 7
    print(f"  {i}", end=" ")
print()


# =============================================================================
# Functions
# =============================================================================

section("Functions")


def greet(name: str) -> str:
    """Simple function with return value."""
    return f"Hello, {name}!"


def calculate_area(width: float, height: float = 10.0) -> float:
    """Function with default argument."""
    return width * height


def sum_all(*numbers: int) -> int:
    """Variable arguments (*args)."""
    return sum(numbers)


def describe_person(**kwargs: str) -> str:
    """Keyword arguments (**kwargs)."""
    parts = [f"{k}={v}" for k, v in kwargs.items()]
    return ", ".join(parts)


print(greet("Alice"))
print(f"Area: {calculate_area(5)}")
print(f"Area: {calculate_area(5, 20)}")
print(f"Sum: {sum_all(1, 2, 3, 4, 5)}")
print(f"Person: {describe_person(name='Bob', age='30', city='NYC')}")


# Lambda functions
square = lambda x: x ** 2
print(f"\nLambda: square(5) = {square(5)}")


# =============================================================================
# String Formatting
# =============================================================================

section("String Formatting")

name = "Alice"
age = 30
pi = 3.14159

# f-strings (Python 3.6+)
print(f"f-string: {name} is {age} years old")
print(f"Expression: 2 + 2 = {2 + 2}")
print(f"Formatted: pi = {pi:.2f}")

# format() method
print("format(): {} is {} years old".format(name, age))
print("format(): {1} is {0} years old".format(age, name))
print("format(): {name} is {age}".format(name=name, age=age))

# %-formatting (old style)
print("%%: %s is %d years old" % (name, age))


# =============================================================================
# Lists
# =============================================================================

section("Lists")

numbers = [1, 2, 3, 4, 5]
print(f"List: {numbers}")

# Indexing
print(f"First: {numbers[0]}")
print(f"Last: {numbers[-1]}")

# Slicing
print(f"First 3: {numbers[:3]}")
print(f"Last 2: {numbers[-2:]}")
print(f"Every 2nd: {numbers[::2]}")

# Methods
numbers.append(6)
print(f"After append(6): {numbers}")

numbers.insert(0, 0)
print(f"After insert(0, 0): {numbers}")

numbers.pop()
print(f"After pop(): {numbers}")

# List operations
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2
print(f"Combined: {combined}")


# =============================================================================
# List Comprehensions
# =============================================================================

section("List Comprehensions")

# Basic comprehension
squares = [x**2 for x in range(10)]
print(f"Squares: {squares}")

# With condition
evens = [x for x in range(20) if x % 2 == 0]
print(f"Evens: {evens}")

# Nested comprehension
matrix = [[i * j for j in range(3)] for i in range(3)]
print(f"Matrix: {matrix}")


# =============================================================================
# Tuples
# =============================================================================

section("Tuples")

point = (10, 20)
print(f"Tuple: {point}")
print(f"  x={point[0]}, y={point[1]}")

# Tuple unpacking
x, y = point
print(f"Unpacked: x={x}, y={y}")

# Tuples are immutable
try:
    point[0] = 100
except TypeError as e:
    print(f"Cannot modify tuple: {e}")


# =============================================================================
# Sets
# =============================================================================

section("Sets")

set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

print(f"Set1: {set1}")
print(f"Set2: {set2}")

# Set operations
print(f"Union: {set1 | set2}")
print(f"Intersection: {set1 & set2}")
print(f"Difference: {set1 - set2}")
print(f"Symmetric difference: {set1 ^ set2}")

# Set comprehension
squares_set = {x**2 for x in range(10)}
print(f"Squares set: {squares_set}")


# =============================================================================
# Dictionaries
# =============================================================================

section("Dictionaries")

person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}

print(f"Dict: {person}")
print(f"Name: {person['name']}")
print(f"Age: {person.get('age')}")
print(f"Country (default): {person.get('country', 'USA')}")

# Add/modify
person["email"] = "alice@example.com"
person["age"] = 31

print(f"Updated: {person}")

# Dict methods
print(f"Keys: {list(person.keys())}")
print(f"Values: {list(person.values())}")
print(f"Items: {list(person.items())}")

# Dict comprehension
squares_dict = {x: x**2 for x in range(5)}
print(f"Squares dict: {squares_dict}")


# =============================================================================
# File I/O
# =============================================================================

section("File I/O")

import tempfile
import os

# Create temp file
temp_file = os.path.join(tempfile.gettempdir(), "python_basics_demo.txt")

# Writing
with open(temp_file, 'w') as f:
    f.write("Hello, File I/O!\n")
    f.write("Python is awesome.\n")
    lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
    f.writelines(lines)

print(f"Wrote to: {temp_file}")

# Reading
with open(temp_file, 'r') as f:
    content = f.read()
    print(f"Read content:\n{content}")

# Reading lines
with open(temp_file, 'r') as f:
    lines = f.readlines()
    print(f"Lines: {len(lines)}")

# Cleanup
os.remove(temp_file)


# =============================================================================
# Exception Handling
# =============================================================================

section("Exception Handling")


def divide(a: float, b: float) -> float:
    """Divide with exception handling."""
    try:
        result = a / b
        print(f"  {a} / {b} = {result}")
        return result
    except ZeroDivisionError:
        print(f"  Error: Cannot divide by zero")
        return None
    except TypeError:
        print(f"  Error: Invalid types")
        return None
    finally:
        print(f"  Finally block executed")


divide(10, 2)
divide(10, 0)


# Raising exceptions
def validate_age(age: int):
    """Validate age."""
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age too high")
    return True


try:
    validate_age(25)
    print("Age 25: valid")
    validate_age(-5)
except ValueError as e:
    print(f"Invalid age: {e}")


# =============================================================================
# Common Built-in Functions
# =============================================================================

section("Common Built-in Functions")

numbers = [3, 1, 4, 1, 5, 9, 2, 6]

print(f"List: {numbers}")
print(f"len(): {len(numbers)}")
print(f"min(): {min(numbers)}")
print(f"max(): {max(numbers)}")
print(f"sum(): {sum(numbers)}")
print(f"sorted(): {sorted(numbers)}")
print(f"reversed(): {list(reversed(numbers))}")

# map, filter
doubled = list(map(lambda x: x * 2, numbers))
print(f"map(x*2): {doubled}")

evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"filter(even): {evens}")

# zip
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
combined = list(zip(names, ages))
print(f"zip: {combined}")

# any, all
values = [True, True, False]
print(f"any({values}): {any(values)}")
print(f"all({values}): {all(values)}")


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Python basics covered:
1. Variables and types - int, float, str, bool, None
2. Operators - arithmetic, comparison, logical
3. Control flow - if/elif/else, for, while
4. Functions - def, return, *args, **kwargs
5. String formatting - f-strings, format(), %
6. Lists - indexing, slicing, methods
7. Comprehensions - list, dict, set
8. Tuples - immutable sequences
9. Sets - unique elements, set operations
10. Dictionaries - key-value pairs
11. File I/O - open, read, write
12. Exception handling - try/except/finally

Key concepts:
- Python is dynamically typed
- Indentation matters (4 spaces)
- Everything is an object
- Duck typing: "If it walks like a duck..."
- Batteries included philosophy

Next steps:
- Object-oriented programming (classes)
- Modules and packages
- Decorators and generators
- Context managers
- Type hints
- Advanced data structures
""")
