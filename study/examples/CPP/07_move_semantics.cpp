/*
 * Move Semantics and Value Categories Demo
 *
 * Demonstrates:
 * - lvalue vs rvalue
 * - Move constructor and move assignment
 * - std::move and std::forward
 * - Perfect forwarding
 * - Rule of five / rule of zero
 * - Return value optimization (RVO)
 *
 * Compile: g++ -std=c++20 -Wall -Wextra 07_move_semantics.cpp -o move_semantics
 */

#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>

// ============ Value Categories ============
void demo_value_categories() {
    std::cout << "\n=== Value Categories ===\n";

    int x = 10;        // x is an lvalue
    int y = 20;        // y is an lvalue

    int z = x + y;     // x + y is an rvalue (temporary)

    std::cout << "lvalue x = " << x << "\n";
    std::cout << "lvalue y = " << y << "\n";
    std::cout << "rvalue (x + y) stored in z = " << z << "\n";

    // int& ref1 = x + y;        // Error: can't bind lvalue ref to rvalue
    int&& ref2 = x + y;          // OK: rvalue reference
    std::cout << "rvalue reference ref2 = " << ref2 << "\n";

    const int& ref3 = x + y;     // OK: const lvalue ref can bind to rvalue
    std::cout << "const lvalue reference ref3 = " << ref3 << "\n";
}

// ============ Rule of Five ============
class Buffer {
private:
    size_t size_;
    int* data_;
    std::string name_;

public:
    // Constructor
    Buffer(size_t size, const std::string& name)
        : size_(size), data_(new int[size]), name_(name) {
        std::cout << "  [Constructor: " << name_ << ", size=" << size_ << "]\n";
        for (size_t i = 0; i < size_; i++) {
            data_[i] = i;
        }
    }

    // Destructor
    ~Buffer() {
        std::cout << "  [Destructor: " << name_ << "]\n";
        delete[] data_;
    }

    // Copy constructor
    Buffer(const Buffer& other)
        : size_(other.size_), data_(new int[other.size_]),
          name_(other.name_ + "_copy") {
        std::cout << "  [Copy constructor: " << name_ << "]\n";
        std::copy(other.data_, other.data_ + size_, data_);
    }

    // Copy assignment
    Buffer& operator=(const Buffer& other) {
        std::cout << "  [Copy assignment: " << other.name_ << " -> " << name_ << "]\n";
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = new int[size_];
            std::copy(other.data_, other.data_ + size_, data_);
        }
        return *this;
    }

    // Move constructor
    Buffer(Buffer&& other) noexcept
        : size_(other.size_), data_(other.data_),
          name_(std::move(other.name_) + "_moved") {
        std::cout << "  [Move constructor: " << name_ << "]\n";
        other.size_ = 0;
        other.data_ = nullptr;
    }

    // Move assignment
    Buffer& operator=(Buffer&& other) noexcept {
        std::cout << "  [Move assignment: " << other.name_ << " -> " << name_ << "]\n";
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = other.data_;
            name_ = std::move(other.name_) + "_moved";

            other.size_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    void print() const {
        std::cout << "  Buffer '" << name_ << "' [size=" << size_ << "]: ";
        if (data_) {
            for (size_t i = 0; i < std::min(size_, size_t(5)); i++) {
                std::cout << data_[i] << " ";
            }
            if (size_ > 5) std::cout << "...";
        } else {
            std::cout << "(empty)";
        }
        std::cout << "\n";
    }
};

void demo_rule_of_five() {
    std::cout << "\n=== Rule of Five ===\n";

    Buffer buf1(10, "buf1");
    buf1.print();

    // Copy constructor
    Buffer buf2 = buf1;
    buf2.print();

    // Move constructor
    Buffer buf3 = std::move(buf1);
    buf3.print();
    buf1.print();  // buf1 is now empty

    // Copy assignment
    Buffer buf4(5, "buf4");
    buf4 = buf2;
    buf4.print();

    // Move assignment
    Buffer buf5(3, "buf5");
    buf5 = std::move(buf3);
    buf5.print();
}

// ============ std::move ============
void demo_std_move() {
    std::cout << "\n=== std::move ===\n";

    std::string str1 = "Hello, World!";
    std::cout << "str1 before move: '" << str1 << "'\n";

    std::string str2 = std::move(str1);  // str1 is moved from
    std::cout << "str1 after move: '" << str1 << "'\n";
    std::cout << "str2 after move: '" << str2 << "'\n";

    // Move into vector
    std::vector<std::string> vec;
    std::string str3 = "Move me!";
    vec.push_back(std::move(str3));
    std::cout << "str3 after push_back(move): '" << str3 << "'\n";
    std::cout << "vec[0]: '" << vec[0] << "'\n";
}

// ============ Perfect Forwarding ============
void process_lvalue(int& x) {
    std::cout << "  Processing lvalue: " << x << "\n";
}

void process_rvalue(int&& x) {
    std::cout << "  Processing rvalue: " << x << "\n";
}

// Universal reference with perfect forwarding
template<typename T>
void forward_to_process(T&& arg) {
    if constexpr (std::is_lvalue_reference_v<T>) {
        process_lvalue(arg);
    } else {
        process_rvalue(std::forward<T>(arg));
    }
}

void demo_perfect_forwarding() {
    std::cout << "\n=== Perfect Forwarding ===\n";

    int x = 42;

    forward_to_process(x);         // lvalue
    forward_to_process(100);       // rvalue
    forward_to_process(x + 10);    // rvalue
}

// ============ Rule of Zero ============
class SimpleBuffer {
private:
    std::vector<int> data_;   // STL handles memory
    std::string name_;

public:
    SimpleBuffer(size_t size, const std::string& name)
        : data_(size), name_(name) {
        std::cout << "  [SimpleBuffer constructor: " << name_ << "]\n";
        for (size_t i = 0; i < size; i++) {
            data_[i] = i;
        }
    }

    // No destructor, copy/move constructors, or assignment operators needed!
    // Compiler-generated versions work correctly

    void print() const {
        std::cout << "  SimpleBuffer '" << name_ << "' [size=" << data_.size() << "]\n";
    }
};

void demo_rule_of_zero() {
    std::cout << "\n=== Rule of Zero ===\n";

    SimpleBuffer buf1(10, "simple1");
    buf1.print();

    // Copy (compiler-generated)
    SimpleBuffer buf2 = buf1;
    buf2.print();

    // Move (compiler-generated)
    SimpleBuffer buf3 = std::move(buf1);
    buf3.print();

    std::cout << "  No manual memory management needed!\n";
}

// ============ Return Value Optimization (RVO) ============
Buffer create_buffer(size_t size, const std::string& name) {
    return Buffer(size, name);  // RVO: no move/copy
}

void demo_rvo() {
    std::cout << "\n=== Return Value Optimization (RVO) ===\n";

    std::cout << "Creating buffer via function...\n";
    Buffer buf = create_buffer(5, "rvo_buffer");
    buf.print();
    std::cout << "Note: No move/copy constructor called (RVO)!\n";
}

// ============ Move-Only Types ============
class MoveOnly {
private:
    std::unique_ptr<int> data_;
    std::string name_;

public:
    MoveOnly(int value, const std::string& name)
        : data_(std::make_unique<int>(value)), name_(name) {
        std::cout << "  [MoveOnly constructor: " << name_ << "]\n";
    }

    // Delete copy
    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;

    // Default move
    MoveOnly(MoveOnly&&) = default;
    MoveOnly& operator=(MoveOnly&&) = default;

    void print() const {
        std::cout << "  MoveOnly '" << name_ << "': ";
        if (data_) {
            std::cout << *data_;
        } else {
            std::cout << "(empty)";
        }
        std::cout << "\n";
    }
};

void demo_move_only() {
    std::cout << "\n=== Move-Only Types ===\n";

    MoveOnly obj1(42, "obj1");
    obj1.print();

    // MoveOnly obj2 = obj1;  // Error: copy deleted
    MoveOnly obj2 = std::move(obj1);  // OK: move
    obj2.print();
    obj1.print();  // obj1 is now empty

    // Store in vector
    std::vector<MoveOnly> vec;
    vec.push_back(std::move(obj2));
    std::cout << "  Moved into vector\n";
}

// ============ Main ============
int main() {
    std::cout << "Move Semantics and Value Categories Demo\n";
    std::cout << "==========================================\n";

    demo_value_categories();
    demo_rule_of_five();
    demo_std_move();
    demo_perfect_forwarding();
    demo_rule_of_zero();
    demo_rvo();
    demo_move_only();

    std::cout << "\nAll demos completed!\n";
    return 0;
}
