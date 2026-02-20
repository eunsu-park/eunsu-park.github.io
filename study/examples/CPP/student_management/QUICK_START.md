# Student Management System - Quick Start

## Build and Run

```bash
# Build the project
make

# Run the program
./student_manager

# Or combine both
make run
```

## Quick Test with Sample Data

```bash
# The program uses an interactive menu
./student_manager

# Then select option 9 to load sample data:
Choice: 9
Enter filename: sample_students.csv
```

## Sample Workflow

```
1. Load sample data (option 9 → sample_students.csv)
2. List all students (option 5)
3. Sort by GPA (option 6 → choice 3)
4. View statistics (option 7)
5. Find a student (option 3 → ID: 1001)
6. Add new student (option 1)
7. Save database (option 8 → output.csv)
```

## Key Features Demonstrated

| Feature | Menu Option | C++ Concepts |
|---------|-------------|--------------|
| Add Student | 1 | `std::make_shared`, exception handling |
| Remove Student | 2 | `std::find_if`, iterators, exceptions |
| Find by ID | 3 | Lambda expressions, algorithms |
| Find by Name | 4 | `std::copy_if`, substring matching |
| List All | 5 | Range-based for, operator overloading |
| Sort | 6 | `std::sort`, custom comparators |
| Statistics | 7 | `std::accumulate`, `std::map` |
| Save to CSV | 8 | File I/O, serialization |
| Load from CSV | 9 | File parsing, deserialization |

## Modern C++17 Highlights

```cpp
// Smart pointers (automatic memory management)
std::shared_ptr<Student> student = std::make_shared<Student>(...);

// Lambda expressions
std::find_if(students.begin(), students.end(),
    [id](const auto& s) { return s->getId() == id; });

// STL algorithms
std::sort(students.begin(), students.end(),
    [](const auto& a, const auto& b) { return a->getGpa() > b->getGpa(); });

// Custom exceptions
throw StudentNotFoundException(id);

// Operator overloading
std::cout << *student << std::endl;

// Range-based for loops
for (const auto& student : students) { /* ... */ }
```

## File Structure

```
student.h/cpp      → Student class (data model)
database.h/cpp     → StudentDatabase (business logic)
main.cpp           → CLI interface (presentation layer)
Makefile           → Build configuration
sample_students.csv → Sample data (8 students)
```

## Compilation Flags

- `-std=c++17` - Use C++17 standard
- `-Wall -Wextra -Wpedantic` - All warnings enabled
- `-O2` - Optimization level 2

## CSV Format

```csv
id,name,major,gpa
1001,Alice Johnson,Computer Science,3.85
1002,Bob Smith,Mathematics,3.92
```

## Learning Path

1. Start with `student.h/cpp` - Understand basic OOP
2. Study `database.h/cpp` - Learn STL containers and algorithms
3. Explore `main.cpp` - See how everything connects
4. Experiment with the Makefile - Understand build dependencies
5. Try modifying the code - Add new features (sorting by major, search filters, etc.)

## Common Tasks

### Add a new sorting option
Edit `database.cpp` and add a new method like:
```cpp
void StudentDatabase::sortByMajor() {
    std::sort(students.begin(), students.end(),
        [](const auto& a, const auto& b) {
            return a->getMajor() < b->getMajor();
        });
}
```

### Add a new search filter
```cpp
std::vector<std::shared_ptr<Student>> findByGpaRange(double min, double max) const {
    std::vector<std::shared_ptr<Student>> results;
    std::copy_if(students.begin(), students.end(), std::back_inserter(results),
        [min, max](const auto& s) {
            return s->getGpa() >= min && s->getGpa() <= max;
        });
    return results;
}
```

## Troubleshooting

**Build fails:** Ensure you have g++ 7.0+ with C++17 support
```bash
g++ --version
```

**Warnings about unused parameters:** All warnings have been addressed in this code

**File not found:** Make sure you're in the correct directory
```bash
pwd  # Should show .../examples/CPP/student_management
```

## Next Steps

- Add unit tests using Google Test or Catch2
- Implement JSON serialization instead of CSV
- Add a GUI using Qt or GTK
- Create a multi-threaded version with thread-safe operations
- Implement database persistence using SQLite
