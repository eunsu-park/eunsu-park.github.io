/*
 * Template Metaprogramming Demo
 *
 * Demonstrates:
 * - Function templates and class templates
 * - Template specialization
 * - Variadic templates
 * - SFINAE basics
 * - C++20 concepts
 * - Type traits
 *
 * Compile: g++ -std=c++20 -Wall -Wextra 06_templates.cpp -o templates
 */

#include <iostream>
#include <vector>
#include <string>
#include <type_traits>
#include <concepts>

// ============ Function Templates ============
template<typename T>
T my_max(T a, T b) {
    return (a > b) ? a : b;
}

// Template overloading
template<typename T>
T my_max(T a, T b, T c) {
    return my_max(my_max(a, b), c);
}

void demo_function_templates() {
    std::cout << "\n=== Function Templates ===\n";

    std::cout << "my_max(10, 20) = " << my_max(10, 20) << "\n";
    std::cout << "my_max(3.5, 2.1) = " << my_max(3.5, 2.1) << "\n";
    std::cout << "my_max(10, 20, 15) = " << my_max(10, 20, 15) << "\n";

    std::string s1 = "hello", s2 = "world";
    std::cout << "my_max(\"hello\", \"world\") = " << my_max(s1, s2) << "\n";
}

// ============ Class Templates ============
template<typename T>
class Stack {
private:
    std::vector<T> data_;

public:
    void push(const T& value) {
        data_.push_back(value);
    }

    void pop() {
        if (!data_.empty()) {
            data_.pop_back();
        }
    }

    T top() const {
        return data_.back();
    }

    bool empty() const {
        return data_.empty();
    }

    size_t size() const {
        return data_.size();
    }
};

void demo_class_templates() {
    std::cout << "\n=== Class Templates ===\n";

    Stack<int> int_stack;
    int_stack.push(10);
    int_stack.push(20);
    int_stack.push(30);

    std::cout << "Int stack top: " << int_stack.top() << "\n";
    std::cout << "Int stack size: " << int_stack.size() << "\n";

    Stack<std::string> str_stack;
    str_stack.push("hello");
    str_stack.push("world");

    std::cout << "String stack top: " << str_stack.top() << "\n";
}

// ============ Template Specialization ============
template<typename T>
class Printer {
public:
    static void print(const T& value) {
        std::cout << "Generic: " << value << "\n";
    }
};

// Full specialization for bool
template<>
class Printer<bool> {
public:
    static void print(const bool& value) {
        std::cout << "Bool: " << (value ? "true" : "false") << "\n";
    }
};

// Partial specialization for pointers
template<typename T>
class Printer<T*> {
public:
    static void print(T* value) {
        std::cout << "Pointer: " << static_cast<void*>(value);
        if (value) {
            std::cout << " -> " << *value;
        }
        std::cout << "\n";
    }
};

void demo_specialization() {
    std::cout << "\n=== Template Specialization ===\n";

    Printer<int>::print(42);
    Printer<bool>::print(true);

    int x = 100;
    Printer<int*>::print(&x);
}

// ============ Variadic Templates ============
// Base case
void print() {
    std::cout << "\n";
}

// Recursive case
template<typename T, typename... Args>
void print(T first, Args... rest) {
    std::cout << first << " ";
    print(rest...);
}

// Variadic sum using fold expression (C++17)
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);
}

void demo_variadic_templates() {
    std::cout << "\n=== Variadic Templates ===\n";

    std::cout << "print(1, 2, 3, \"hello\", 3.14): ";
    print(1, 2, 3, "hello", 3.14);

    std::cout << "sum(1, 2, 3, 4, 5) = " << sum(1, 2, 3, 4, 5) << "\n";
    std::cout << "sum(1.5, 2.5, 3.0) = " << sum(1.5, 2.5, 3.0) << "\n";
}

// ============ SFINAE (Substitution Failure Is Not An Error) ============
// Enable if T is integral
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
double_value(T value) {
    return value * 2;
}

// Enable if T is floating point
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
double_value(T value) {
    return value * 2.0;
}

void demo_sfinae() {
    std::cout << "\n=== SFINAE ===\n";

    std::cout << "double_value(10) = " << double_value(10) << "\n";
    std::cout << "double_value(3.5) = " << double_value(3.5) << "\n";

    // This would fail to compile:
    // double_value("hello");
}

// ============ C++20 Concepts ============
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
};

template<Numeric T>
T triple(T value) {
    return value * 3;
}

template<Addable T>
T add_three(T a, T b, T c) {
    return a + b + c;
}

void demo_concepts() {
    std::cout << "\n=== C++20 Concepts ===\n";

    std::cout << "triple(10) = " << triple(10) << "\n";
    std::cout << "triple(3.5) = " << triple(3.5) << "\n";

    std::cout << "add_three(1, 2, 3) = " << add_three(1, 2, 3) << "\n";
    std::cout << "add_three(1.5, 2.5, 3.0) = " << add_three(1.5, 2.5, 3.0) << "\n";

    // These would fail to compile:
    // triple("hello");
    // add_three("a", "b", "c");  // strings are Addable but behave differently
}

// ============ Type Traits ============
template<typename T>
void analyze_type(T value) {
    std::cout << "Type analysis for value: " << value << "\n";
    std::cout << "  is_integral: " << std::is_integral_v<T> << "\n";
    std::cout << "  is_floating_point: " << std::is_floating_point_v<T> << "\n";
    std::cout << "  is_pointer: " << std::is_pointer_v<T> << "\n";
    std::cout << "  is_const: " << std::is_const_v<T> << "\n";
    std::cout << "  is_arithmetic: " << std::is_arithmetic_v<T> << "\n";
}

void demo_type_traits() {
    std::cout << "\n=== Type Traits ===\n";

    analyze_type(42);
    analyze_type(3.14);

    int x = 10;
    analyze_type(&x);
}

// ============ Template Template Parameters ============
template<typename T, template<typename> class Container>
class Wrapper {
private:
    Container<T> data_;

public:
    void add(const T& value) {
        data_.push_back(value);
    }

    void print() const {
        std::cout << "  Container contents: ";
        for (const auto& item : data_) {
            std::cout << item << " ";
        }
        std::cout << "\n";
    }
};

void demo_template_template() {
    std::cout << "\n=== Template Template Parameters ===\n";

    Wrapper<int, std::vector> wrapper;
    wrapper.add(10);
    wrapper.add(20);
    wrapper.add(30);
    wrapper.print();
}

// ============ Compile-Time Computation ============
template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N - 1>::value;
};

template<>
struct Factorial<0> {
    static constexpr int value = 1;
};

// C++11 constexpr function
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

void demo_compile_time() {
    std::cout << "\n=== Compile-Time Computation ===\n";

    std::cout << "Factorial<5>::value = " << Factorial<5>::value << "\n";
    std::cout << "factorial(5) = " << factorial(5) << "\n";

    // These are computed at compile time
    constexpr int fact10 = factorial(10);
    std::cout << "factorial(10) = " << fact10 << "\n";
}

// ============ Main ============
int main() {
    std::cout << "Template Metaprogramming Demo\n";
    std::cout << "==============================\n";

    demo_function_templates();
    demo_class_templates();
    demo_specialization();
    demo_variadic_templates();
    demo_sfinae();
    demo_concepts();
    demo_type_traits();
    demo_template_template();
    demo_compile_time();

    std::cout << "\nAll demos completed!\n";
    return 0;
}
