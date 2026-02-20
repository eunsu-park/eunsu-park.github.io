/*
 * Smart Pointers Demo
 *
 * Demonstrates:
 * - unique_ptr: exclusive ownership
 * - shared_ptr: shared ownership with reference counting
 * - weak_ptr: breaking circular references
 * - Custom deleters
 * - make_unique / make_shared
 *
 * Compile: g++ -std=c++20 -Wall -Wextra 03_smart_pointers.cpp -o smart_pointers
 */

#include <iostream>
#include <memory>
#include <vector>
#include <string>

// ============ Helper Class ============
class Resource {
private:
    std::string name_;
    int* data_;

public:
    Resource(const std::string& name) : name_(name) {
        data_ = new int(42);
        std::cout << "  [Resource '" << name_ << "' created]\n";
    }

    ~Resource() {
        delete data_;
        std::cout << "  [Resource '" << name_ << "' destroyed]\n";
    }

    void use() const {
        std::cout << "  Using resource '" << name_ << "', data=" << *data_ << "\n";
    }

    std::string get_name() const { return name_; }
};

// ============ unique_ptr ============
void demo_unique_ptr() {
    std::cout << "\n=== unique_ptr (Exclusive Ownership) ===\n";

    // Create unique_ptr with make_unique
    std::unique_ptr<Resource> ptr1 = std::make_unique<Resource>("unique-1");
    ptr1->use();

    // Move semantics (transfer ownership)
    std::cout << "Moving ownership...\n";
    std::unique_ptr<Resource> ptr2 = std::move(ptr1);

    if (!ptr1) {
        std::cout << "  ptr1 is now nullptr\n";
    }

    if (ptr2) {
        std::cout << "  ptr2 owns the resource\n";
        ptr2->use();
    }

    // Array with unique_ptr
    std::cout << "\nUnique array:\n";
    std::unique_ptr<int[]> arr = std::make_unique<int[]>(5);
    for (int i = 0; i < 5; i++) {
        arr[i] = i * 10;
    }
    std::cout << "  Array: ";
    for (int i = 0; i < 5; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Exiting demo_unique_ptr scope...\n";
}

// ============ shared_ptr ============
void demo_shared_ptr() {
    std::cout << "\n=== shared_ptr (Shared Ownership) ===\n";

    // Create shared_ptr with make_shared
    std::shared_ptr<Resource> ptr1 = std::make_shared<Resource>("shared-1");
    std::cout << "  ptr1 use_count: " << ptr1.use_count() << "\n";

    // Copy (share ownership)
    {
        std::shared_ptr<Resource> ptr2 = ptr1;
        std::cout << "  ptr1 use_count: " << ptr1.use_count() << "\n";
        std::cout << "  ptr2 use_count: " << ptr2.use_count() << "\n";

        std::shared_ptr<Resource> ptr3 = ptr1;
        std::cout << "  ptr1 use_count: " << ptr1.use_count() << "\n";

        std::cout << "  Exiting inner scope...\n";
    }

    std::cout << "  ptr1 use_count after scope: " << ptr1.use_count() << "\n";
    ptr1->use();

    std::cout << "Exiting demo_shared_ptr scope...\n";
}

// ============ weak_ptr (Breaking Cycles) ============
class Node {
public:
    std::string name;
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;  // weak_ptr to avoid cycle

    Node(const std::string& n) : name(n) {
        std::cout << "  [Node '" << name << "' created]\n";
    }

    ~Node() {
        std::cout << "  [Node '" << name << "' destroyed]\n";
    }
};

void demo_weak_ptr() {
    std::cout << "\n=== weak_ptr (Breaking Circular References) ===\n";

    std::shared_ptr<Node> node1 = std::make_shared<Node>("A");
    std::shared_ptr<Node> node2 = std::make_shared<Node>("B");

    // Create doubly-linked list
    node1->next = node2;
    node2->prev = node1;  // weak_ptr: doesn't increment ref count

    std::cout << "  node1 use_count: " << node1.use_count() << "\n";
    std::cout << "  node2 use_count: " << node2.use_count() << "\n";

    // Access weak_ptr through lock()
    if (auto prev = node2->prev.lock()) {
        std::cout << "  node2's prev: " << prev->name << "\n";
    }

    std::cout << "Exiting demo_weak_ptr scope...\n";
}

// ============ Custom Deleter ============
struct FileCloser {
    void operator()(FILE* fp) const {
        if (fp) {
            std::cout << "  [Custom deleter: closing file]\n";
            fclose(fp);
        }
    }
};

void demo_custom_deleter() {
    std::cout << "\n=== Custom Deleter ===\n";

    // unique_ptr with custom deleter
    {
        std::unique_ptr<FILE, FileCloser> file(fopen("/tmp/test.txt", "w"));
        if (file) {
            fprintf(file.get(), "Hello from custom deleter!\n");
            std::cout << "  File written\n";
        }
        std::cout << "  Exiting scope...\n";
    }
    std::cout << "  File automatically closed by custom deleter\n";

    // shared_ptr with custom deleter (lambda)
    {
        std::shared_ptr<Resource> ptr(
            new Resource("custom-delete"),
            [](Resource* r) {
                std::cout << "  [Lambda deleter called]\n";
                delete r;
            }
        );
        ptr->use();
        std::cout << "  Exiting scope...\n";
    }
}

// ============ Container of Smart Pointers ============
void demo_container() {
    std::cout << "\n=== Container of Smart Pointers ===\n";

    std::vector<std::unique_ptr<Resource>> resources;

    resources.push_back(std::make_unique<Resource>("resource-1"));
    resources.push_back(std::make_unique<Resource>("resource-2"));
    resources.push_back(std::make_unique<Resource>("resource-3"));

    std::cout << "Using all resources:\n";
    for (const auto& res : resources) {
        res->use();
    }

    std::cout << "Clearing container...\n";
    resources.clear();
    std::cout << "All resources destroyed\n";
}

// ============ Polymorphism with Smart Pointers ============
class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0;
};

class Circle : public Shape {
public:
    void draw() const override {
        std::cout << "  Drawing Circle\n";
    }
};

class Rectangle : public Shape {
public:
    void draw() const override {
        std::cout << "  Drawing Rectangle\n";
    }
};

void demo_polymorphism() {
    std::cout << "\n=== Polymorphism with Smart Pointers ===\n";

    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>());
    shapes.push_back(std::make_unique<Rectangle>());
    shapes.push_back(std::make_unique<Circle>());

    std::cout << "Drawing all shapes:\n";
    for (const auto& shape : shapes) {
        shape->draw();
    }
}

// ============ Main ============
int main() {
    std::cout << "Smart Pointers Demo\n";
    std::cout << "===================\n";

    demo_unique_ptr();
    demo_shared_ptr();
    demo_weak_ptr();
    demo_custom_deleter();
    demo_container();
    demo_polymorphism();

    std::cout << "\nAll demos completed!\n";
    return 0;
}
