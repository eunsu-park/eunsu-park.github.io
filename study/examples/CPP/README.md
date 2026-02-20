# C++ Examples

This directory contains comprehensive C++ example programs demonstrating modern C++ features and best practices.

## Files

| File | Description | Key Topics |
|------|-------------|------------|
| `01_modern_cpp.cpp` | Modern C++ features (C++17/C++20) | Structured bindings, `std::optional`, `std::variant`, `std::any`, `if constexpr`, fold expressions, `std::filesystem`, concepts |
| `02_stl_containers.cpp` | STL containers and algorithms | `vector`, `map`, `unordered_map`, `set`, algorithms, ranges (C++20), lambdas |
| `03_smart_pointers.cpp` | Smart pointer patterns | `unique_ptr`, `shared_ptr`, `weak_ptr`, custom deleters, polymorphism |
| `04_threading.cpp` | Multithreading and concurrency | `std::thread`, mutex, `std::async`, `std::future`, condition variables, thread-safe queue, `std::jthread` (C++20) |
| `05_design_patterns.cpp` | Common design patterns | Singleton, Observer, Strategy, RAII, CRTP, Factory |
| `06_templates.cpp` | Template metaprogramming | Function/class templates, specialization, variadic templates, SFINAE, concepts, type traits |
| `07_move_semantics.cpp` | Move semantics and value categories | lvalue/rvalue, move constructor/assignment, `std::move`, `std::forward`, perfect forwarding, Rule of Five/Zero |

## Building

### Build all examples
```bash
make all
```

### Build specific example
```bash
make modern      # Build 01_modern_cpp
make stl         # Build 02_stl_containers
make smart       # Build 03_smart_pointers
make threading   # Build 04_threading
make patterns    # Build 05_design_patterns
make templates   # Build 06_templates
make move        # Build 07_move_semantics
```

### Run specific example
```bash
make run-01_modern_cpp
make run-02_stl_containers
# etc.
```

### Clean build artifacts
```bash
make clean
```

### Display help
```bash
make help
```

## Requirements

- **Compiler**: g++ or clang++ with C++20 support
- **Standard**: C++20 (minimum C++17 for some examples)
- **Flags**: `-std=c++20 -Wall -Wextra -O2 -pthread`

### Checking compiler version
```bash
g++ --version
```

Make sure your compiler supports C++20:
- GCC 10+ or Clang 10+

## Running Examples

Each example is a standalone executable:

```bash
./01_modern_cpp
./02_stl_containers
./03_smart_pointers
./04_threading
./05_design_patterns
./06_templates
./07_move_semantics
```

## Example Output

### 01_modern_cpp
```
Modern C++ Features Demo
========================

=== Structured Bindings (C++17) ===
Tuple: 42, hello, 3.14
Alice: 95
Bob: 87
...
```

### 02_stl_containers
```
STL Containers and Algorithms Demo
===================================

=== std::vector ===
Original: 5 2 8 1 9
After push_back(42): size=6
Sorted: 1 2 5 8 9 42
...
```

### 03_smart_pointers
```
Smart Pointers Demo
===================

=== unique_ptr (Exclusive Ownership) ===
  [Resource 'unique-1' created]
  Using resource 'unique-1', data=42
Moving ownership...
...
```

## Key Features Demonstrated

### Modern C++ (C++11/14/17/20)
- Structured bindings (C++17)
- `std::optional`, `std::variant`, `std::any`
- `if constexpr` (C++17)
- Fold expressions (C++17)
- Concepts (C++20)
- Ranges (C++20)
- `std::jthread` (C++20)

### Memory Management
- RAII (Resource Acquisition Is Initialization)
- Smart pointers: `unique_ptr`, `shared_ptr`, `weak_ptr`
- Custom deleters
- Rule of Five / Rule of Zero
- Move semantics

### Concurrency
- `std::thread` basics
- Mutex and lock guards
- `std::async` and `std::future`
- Condition variables
- Thread-safe data structures
- Parallel algorithms

### Design Patterns
- Singleton (thread-safe Meyer's)
- Observer with `std::function`
- Strategy with lambdas
- CRTP (static polymorphism)
- Factory pattern

### Templates
- Function and class templates
- Template specialization (full and partial)
- Variadic templates
- SFINAE (Substitution Failure Is Not An Error)
- Concepts (C++20)
- Template template parameters
- Compile-time computation

### STL
- Containers: `vector`, `map`, `unordered_map`, `set`
- Algorithms: `sort`, `transform`, `accumulate`, `find_if`
- Ranges and views (C++20)
- Lambda expressions
- Iterators

## Learning Path

1. **Start here**: `01_modern_cpp.cpp` - Get familiar with modern C++ syntax
2. **Containers**: `02_stl_containers.cpp` - Learn STL containers and algorithms
3. **Memory**: `03_smart_pointers.cpp` - Understand modern memory management
4. **Concurrency**: `04_threading.cpp` - Explore multithreading
5. **Patterns**: `05_design_patterns.cpp` - Study common design patterns
6. **Templates**: `06_templates.cpp` - Master template metaprogramming
7. **Advanced**: `07_move_semantics.cpp` - Deep dive into move semantics

## Compilation Notes

### C++20 Support
Most examples require C++20. If your compiler doesn't support C++20, you can try C++17:

```bash
g++ -std=c++17 -Wall -Wextra 01_modern_cpp.cpp -o modern_cpp
```

Some features will be unavailable:
- `std::jthread` (C++20)
- Concepts (C++20)
- Ranges (C++20)
- `contains()` for sets (C++20)

### Threading
Threading examples require linking against pthread on Unix systems:

```bash
g++ -std=c++20 -pthread 04_threading.cpp -o threading
```

This is handled automatically by the Makefile.

## Related Resources

- [CPP Learning Guide](/opt/projects/01_Personal/03_Study/content/en/CPP/00_Overview.md)
- [C Programming Examples](/opt/projects/01_Personal/03_Study/examples/C_Programming/)
- [cppreference.com](https://en.cppreference.com/) - Comprehensive C++ reference

## License

MIT License - See project root for details
