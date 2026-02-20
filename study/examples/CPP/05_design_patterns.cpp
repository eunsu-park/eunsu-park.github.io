/*
 * Design Patterns in C++ Demo
 *
 * Demonstrates:
 * - Singleton (thread-safe Meyer's)
 * - Observer pattern with std::function
 * - Strategy pattern with lambdas
 * - RAII / Resource management
 * - CRTP (Curiously Recurring Template Pattern)
 *
 * Compile: g++ -std=c++20 -Wall -Wextra 05_design_patterns.cpp -o design_patterns
 */

#include <iostream>
#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <algorithm>
#include <fstream>

// ============ Singleton Pattern (Thread-Safe Meyer's) ============
class Logger {
private:
    Logger() {
        std::cout << "[Logger initialized]\n";
    }

public:
    // Delete copy/move
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

    static Logger& instance() {
        static Logger instance;  // Thread-safe in C++11+
        return instance;
    }

    void log(const std::string& message) {
        std::cout << "[LOG] " << message << "\n";
    }
};

void demo_singleton() {
    std::cout << "\n=== Singleton Pattern ===\n";

    Logger::instance().log("First message");
    Logger::instance().log("Second message");

    Logger& logger = Logger::instance();
    logger.log("Third message");
}

// ============ Observer Pattern ============
class Subject {
private:
    std::vector<std::function<void(const std::string&)>> observers_;

public:
    void attach(std::function<void(const std::string&)> observer) {
        observers_.push_back(observer);
    }

    void notify(const std::string& event) {
        for (auto& observer : observers_) {
            observer(event);
        }
    }
};

void demo_observer() {
    std::cout << "\n=== Observer Pattern ===\n";

    Subject subject;

    // Attach observers using lambdas
    subject.attach([](const std::string& event) {
        std::cout << "Observer 1 received: " << event << "\n";
    });

    subject.attach([](const std::string& event) {
        std::cout << "Observer 2 received: " << event << "\n";
    });

    // Capture state in observer
    int count = 0;
    subject.attach([&count](const std::string& event) {
        count++;
        std::cout << "Observer 3 received (count=" << count << "): " << event << "\n";
    });

    // Notify all observers
    subject.notify("Button clicked");
    subject.notify("Data updated");
}

// ============ Strategy Pattern ============
class SortStrategy {
public:
    virtual ~SortStrategy() = default;
    virtual void sort(std::vector<int>& data) = 0;
};

class BubbleSort : public SortStrategy {
public:
    void sort(std::vector<int>& data) override {
        std::cout << "  Using bubble sort\n";
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data.size() - i - 1; j++) {
                if (data[j] > data[j + 1]) {
                    std::swap(data[j], data[j + 1]);
                }
            }
        }
    }
};

class QuickSort : public SortStrategy {
public:
    void sort(std::vector<int>& data) override {
        std::cout << "  Using quick sort (std::sort)\n";
        std::sort(data.begin(), data.end());
    }
};

class Sorter {
private:
    std::unique_ptr<SortStrategy> strategy_;

public:
    void set_strategy(std::unique_ptr<SortStrategy> strategy) {
        strategy_ = std::move(strategy);
    }

    void sort(std::vector<int>& data) {
        if (strategy_) {
            strategy_->sort(data);
        }
    }
};

// Strategy with lambdas (modern approach)
using SortFunc = std::function<void(std::vector<int>&)>;

void demo_strategy() {
    std::cout << "\n=== Strategy Pattern ===\n";

    std::vector<int> data1 = {5, 2, 8, 1, 9};
    std::vector<int> data2 = {5, 2, 8, 1, 9};

    // Classic approach
    Sorter sorter;
    sorter.set_strategy(std::make_unique<BubbleSort>());
    sorter.sort(data1);

    sorter.set_strategy(std::make_unique<QuickSort>());
    sorter.sort(data2);

    // Lambda-based approach
    std::cout << "\nLambda-based strategy:\n";
    std::vector<int> data3 = {5, 2, 8, 1, 9};

    SortFunc reverse_sort = [](std::vector<int>& data) {
        std::cout << "  Using reverse sort\n";
        std::sort(data.begin(), data.end(), std::greater<int>());
    };

    reverse_sort(data3);

    std::cout << "  Sorted (descending): ";
    for (int n : data3) {
        std::cout << n << " ";
    }
    std::cout << "\n";
}

// ============ RAII (Resource Acquisition Is Initialization) ============
class FileHandler {
private:
    std::ofstream file_;
    std::string filename_;

public:
    FileHandler(const std::string& filename) : filename_(filename) {
        file_.open(filename);
        std::cout << "  [File '" << filename_ << "' opened]\n";
    }

    ~FileHandler() {
        if (file_.is_open()) {
            file_.close();
            std::cout << "  [File '" << filename_ << "' closed]\n";
        }
    }

    // Delete copy, allow move
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;

    FileHandler(FileHandler&& other) noexcept
        : file_(std::move(other.file_)), filename_(std::move(other.filename_)) {
    }

    void write(const std::string& data) {
        if (file_.is_open()) {
            file_ << data;
        }
    }
};

void demo_raii() {
    std::cout << "\n=== RAII Pattern ===\n";

    {
        FileHandler file("/tmp/raii_test.txt");
        file.write("Hello RAII!\n");
        file.write("Resource automatically managed.\n");

        std::cout << "  Using file...\n";
        // File automatically closed when leaving scope
    }

    std::cout << "  File closed automatically\n";
}

// ============ CRTP (Curiously Recurring Template Pattern) ============
template<typename Derived>
class Shape {
public:
    void draw() const {
        static_cast<const Derived*>(this)->draw_impl();
    }

    double area() const {
        return static_cast<const Derived*>(this)->area_impl();
    }
};

class Circle : public Shape<Circle> {
private:
    double radius_;

public:
    Circle(double r) : radius_(r) {}

    void draw_impl() const {
        std::cout << "  Drawing Circle (radius=" << radius_ << ")\n";
    }

    double area_impl() const {
        return 3.14159 * radius_ * radius_;
    }
};

class Rectangle : public Shape<Rectangle> {
private:
    double width_, height_;

public:
    Rectangle(double w, double h) : width_(w), height_(h) {}

    void draw_impl() const {
        std::cout << "  Drawing Rectangle (" << width_ << "x" << height_ << ")\n";
    }

    double area_impl() const {
        return width_ * height_;
    }
};

template<typename T>
void process_shape(const Shape<T>& shape) {
    shape.draw();
    std::cout << "    Area: " << shape.area() << "\n";
}

void demo_crtp() {
    std::cout << "\n=== CRTP (Curiously Recurring Template Pattern) ===\n";

    Circle circle(5.0);
    Rectangle rect(4.0, 6.0);

    process_shape(circle);
    process_shape(rect);

    std::cout << "  CRTP enables static polymorphism (no virtual calls)\n";
}

// ============ Factory Pattern ============
class Animal {
public:
    virtual ~Animal() = default;
    virtual void make_sound() const = 0;
};

class Dog : public Animal {
public:
    void make_sound() const override {
        std::cout << "  Woof!\n";
    }
};

class Cat : public Animal {
public:
    void make_sound() const override {
        std::cout << "  Meow!\n";
    }
};

class AnimalFactory {
public:
    static std::unique_ptr<Animal> create(const std::string& type) {
        if (type == "dog") {
            return std::make_unique<Dog>();
        } else if (type == "cat") {
            return std::make_unique<Cat>();
        }
        return nullptr;
    }
};

void demo_factory() {
    std::cout << "\n=== Factory Pattern ===\n";

    auto dog = AnimalFactory::create("dog");
    auto cat = AnimalFactory::create("cat");

    if (dog) {
        std::cout << "Dog says: ";
        dog->make_sound();
    }

    if (cat) {
        std::cout << "Cat says: ";
        cat->make_sound();
    }
}

// ============ Main ============
int main() {
    std::cout << "Design Patterns in C++ Demo\n";
    std::cout << "============================\n";

    demo_singleton();
    demo_observer();
    demo_strategy();
    demo_raii();
    demo_crtp();
    demo_factory();

    std::cout << "\nAll demos completed!\n";
    return 0;
}
