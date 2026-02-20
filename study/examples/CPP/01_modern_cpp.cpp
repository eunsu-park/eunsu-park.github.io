/*
 * Modern C++ Features Demo (C++17/C++20)
 *
 * Demonstrates:
 * - Structured bindings (C++17)
 * - std::optional, std::variant, std::any
 * - if constexpr
 * - Fold expressions
 * - std::filesystem
 * - C++20 concepts
 *
 * Compile: g++ -std=c++20 -Wall -Wextra 01_modern_cpp.cpp -o modern_cpp
 */

#include <iostream>
#include <optional>
#include <variant>
#include <any>
#include <vector>
#include <map>
#include <filesystem>
#include <string>
#include <concepts>

namespace fs = std::filesystem;

// ============ C++20 Concepts ============
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<Numeric T>
T add(T a, T b) {
    return a + b;
}

// ============ Structured Bindings ============
void demo_structured_bindings() {
    std::cout << "\n=== Structured Bindings (C++17) ===\n";

    // Tuple decomposition
    auto tuple = std::make_tuple(42, "hello", 3.14);
    auto [num, str, pi] = tuple;
    std::cout << "Tuple: " << num << ", " << str << ", " << pi << "\n";

    // Map iteration
    std::map<std::string, int> scores = {{"Alice", 95}, {"Bob", 87}};
    for (const auto& [name, score] : scores) {
        std::cout << name << ": " << score << "\n";
    }

    // Pair decomposition
    std::pair<int, std::string> pair = {100, "perfect"};
    auto [value, description] = pair;
    std::cout << "Pair: " << value << " = " << description << "\n";
}

// ============ std::optional ============
std::optional<int> safe_divide(int a, int b) {
    if (b == 0) return std::nullopt;
    return a / b;
}

void demo_optional() {
    std::cout << "\n=== std::optional ===\n";

    auto result1 = safe_divide(10, 2);
    if (result1.has_value()) {
        std::cout << "10 / 2 = " << result1.value() << "\n";
    }

    auto result2 = safe_divide(10, 0);
    std::cout << "10 / 0 = " << result2.value_or(-1) << " (using default)\n";

    // Optional chaining with transform (C++23 style shown with manual check)
    std::optional<std::string> name = "Alice";
    if (name) {
        std::cout << "Name length: " << name->length() << "\n";
    }
}

// ============ std::variant ============
void demo_variant() {
    std::cout << "\n=== std::variant ===\n";

    std::variant<int, double, std::string> var;

    var = 42;
    std::cout << "Variant holds int: " << std::get<int>(var) << "\n";

    var = 3.14;
    std::cout << "Variant holds double: " << std::get<double>(var) << "\n";

    var = "hello";
    std::cout << "Variant holds string: " << std::get<std::string>(var) << "\n";

    // Visitor pattern
    std::visit([](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int>)
            std::cout << "Visiting int: " << arg << "\n";
        else if constexpr (std::is_same_v<T, double>)
            std::cout << "Visiting double: " << arg << "\n";
        else if constexpr (std::is_same_v<T, std::string>)
            std::cout << "Visiting string: " << arg << "\n";
    }, var);
}

// ============ std::any ============
void demo_any() {
    std::cout << "\n=== std::any ===\n";

    std::any a = 42;
    std::cout << "Any holds: " << std::any_cast<int>(a) << "\n";

    a = 3.14;
    std::cout << "Any holds: " << std::any_cast<double>(a) << "\n";

    a = std::string("hello");
    std::cout << "Any holds: " << std::any_cast<std::string>(a) << "\n";

    if (a.has_value()) {
        std::cout << "Any has a value of type: " << a.type().name() << "\n";
    }
}

// ============ if constexpr ============
template<typename T>
void print_type(T value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Integer: " << value << "\n";
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Float: " << value << "\n";
    } else {
        std::cout << "Other: " << value << "\n";
    }
}

void demo_if_constexpr() {
    std::cout << "\n=== if constexpr ===\n";
    print_type(42);
    print_type(3.14);
    print_type("hello");
}

// ============ Fold Expressions ============
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);  // Unary right fold
}

template<typename... Args>
void print_all(Args... args) {
    ((std::cout << args << " "), ...);  // Binary left fold
    std::cout << "\n";
}

void demo_fold_expressions() {
    std::cout << "\n=== Fold Expressions ===\n";
    std::cout << "Sum: " << sum(1, 2, 3, 4, 5) << "\n";
    std::cout << "Print all: ";
    print_all(1, "hello", 3.14, "world");
}

// ============ std::filesystem ============
void demo_filesystem() {
    std::cout << "\n=== std::filesystem ===\n";

    fs::path current = fs::current_path();
    std::cout << "Current directory: " << current << "\n";

    // Create temporary directory and file
    fs::path temp_dir = fs::temp_directory_path() / "cpp_demo";
    if (!fs::exists(temp_dir)) {
        fs::create_directory(temp_dir);
        std::cout << "Created: " << temp_dir << "\n";
    }

    fs::path temp_file = temp_dir / "test.txt";
    std::cout << "File exists: " << fs::exists(temp_file) << "\n";
    std::cout << "Is directory: " << fs::is_directory(temp_dir) << "\n";

    // Cleanup
    if (fs::exists(temp_dir)) {
        fs::remove_all(temp_dir);
    }
}

// ============ C++20 Concepts Example ============
void demo_concepts() {
    std::cout << "\n=== C++20 Concepts ===\n";
    std::cout << "add(10, 20) = " << add(10, 20) << "\n";
    std::cout << "add(3.5, 1.5) = " << add(3.5, 1.5) << "\n";
    // add("hello", "world");  // Compile error: doesn't satisfy Numeric
}

// ============ Main ============
int main() {
    std::cout << "Modern C++ Features Demo\n";
    std::cout << "========================\n";

    demo_structured_bindings();
    demo_optional();
    demo_variant();
    demo_any();
    demo_if_constexpr();
    demo_fold_expressions();
    demo_filesystem();
    demo_concepts();

    std::cout << "\nAll demos completed!\n";
    return 0;
}
