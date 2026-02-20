# Student Management System

A comprehensive C++17 educational example demonstrating modern C++ features, STL algorithms, smart pointers, file I/O, and exception handling.

## Features

- **CRUD Operations**: Add, remove, find, and list students
- **Smart Pointers**: Uses `std::shared_ptr` for automatic memory management
- **STL Algorithms**: Demonstrates `find_if`, `sort`, `accumulate`, `count_if`, `copy_if`
- **File Persistence**: Save/load database to/from CSV files
- **Exception Handling**: Custom exceptions and error handling
- **Interactive CLI**: Menu-driven user interface

## File Structure

```
student_management/
├── student.h           # Student class declaration
├── student.cpp         # Student class implementation
├── database.h          # StudentDatabase class declaration
├── database.cpp        # StudentDatabase implementation
├── main.cpp            # Interactive CLI application
├── Makefile            # Build configuration
└── README.md           # This file
```

## Building

### Requirements
- C++17 compatible compiler (g++ 7.0+, clang++ 5.0+)
- Make

### Compile
```bash
make
```

### Clean Build
```bash
make clean
make
```

## Running

```bash
./student_manager
```

Or use the Makefile:
```bash
make run
```

## Usage Examples

### 1. Add Students
```
Choice: 1
Student ID: 1001
Name: Alice Johnson
Major: Computer Science
GPA: 3.85
Student added successfully!
```

### 2. List All Students
```
Choice: 5
================================================================================
STUDENT DATABASE (3 students)
================================================================================
ID:  1001 | Name: Alice Johnson        | Major: Computer Science | GPA: 3.85
ID:  1002 | Name: Bob Smith            | Major: Mathematics      | GPA: 3.92
ID:  1003 | Name: Carol Williams       | Major: Computer Science | GPA: 3.67
================================================================================
```

### 3. Sort by GPA (Descending)
```
Choice: 6
Sort By:
1. ID
2. Name
3. GPA (descending)
Choice: 3
Sorted by GPA (descending).
```

### 4. Statistics
```
Choice: 7
==================================================
STATISTICS
==================================================
Total Students: 3
Average GPA: 3.81

Students by Major:
  Computer Science    : 2 student(s)
  Mathematics         : 1 student(s)
==================================================
```

### 5. Save to File
```
Choice: 8
Enter filename (e.g., students.csv): data.csv
Database saved to data.csv (3 students)
```

### 6. Load from File
```
Choice: 9
Enter filename: data.csv
Database loaded from data.csv (3 students)
```

## C++17 Features Demonstrated

### Smart Pointers
```cpp
std::shared_ptr<Student> student = std::make_shared<Student>(id, name, major, gpa);
std::vector<std::shared_ptr<Student>> students;
```

### Lambda Expressions
```cpp
auto it = std::find_if(students.begin(), students.end(),
    [id](const std::shared_ptr<Student>& s) {
        return s->getId() == id;
    });
```

### STL Algorithms
```cpp
// Sort by GPA (descending)
std::sort(students.begin(), students.end(),
    [](const auto& a, const auto& b) {
        return a->getGpa() > b->getGpa();
    });

// Calculate average GPA
double sum = std::accumulate(students.begin(), students.end(), 0.0,
    [](double total, const auto& s) {
        return total + s->getGpa();
    });
```

### Operator Overloading
```cpp
// Stream insertion operator for pretty printing
friend std::ostream& operator<<(std::ostream& os, const Student& student);

// Comparison operators
bool operator<(const Student& other) const;
bool operator==(const Student& other) const;
```

### Custom Exceptions
```cpp
class StudentNotFoundException : public std::exception {
    const char* what() const noexcept override;
};
```

### File I/O with Exception Safety
```cpp
void saveToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file");
    }
    // Write data...
}
```

## Design Patterns

- **Encapsulation**: Private members with public getters/setters
- **RAII**: Automatic resource management via smart pointers and RAII objects
- **Exception Safety**: Proper error handling throughout
- **Separation of Concerns**: Student, Database, and UI are separate modules

## CSV File Format

```csv
id,name,major,gpa
1001,Alice Johnson,Computer Science,3.85
1002,Bob Smith,Mathematics,3.92
1003,Carol Williams,Computer Science,3.67
```

## Error Handling

The system handles various error conditions:
- Duplicate student IDs
- Invalid GPA values (must be 0.0-4.0)
- Student not found errors
- File I/O errors
- Invalid CSV format

## Educational Goals

This project is designed to teach:

1. **Modern C++ (C++17)**: Smart pointers, lambdas, auto keyword
2. **STL Containers**: `std::vector`, `std::map`
3. **STL Algorithms**: `find_if`, `sort`, `accumulate`, `copy_if`, `count_if`
4. **Object-Oriented Design**: Classes, encapsulation, operator overloading
5. **Exception Handling**: Custom exceptions, RAII
6. **File I/O**: Reading/writing CSV files
7. **Build Systems**: Makefiles with proper dependencies

## License

MIT License - Educational example for the 03_Study project.
