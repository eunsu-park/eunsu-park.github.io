/*
 * STL Containers and Algorithms Demo
 *
 * Demonstrates:
 * - vector, map, unordered_map, set
 * - Range-based for loops
 * - STL algorithms: sort, transform, accumulate, find_if
 * - std::ranges (C++20)
 * - Lambda expressions with STL
 *
 * Compile: g++ -std=c++20 -Wall -Wextra 02_stl_containers.cpp -o stl_containers
 */

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <numeric>
#include <string>
#include <ranges>

// ============ std::vector ============
void demo_vector() {
    std::cout << "\n=== std::vector ===\n";

    std::vector<int> nums = {5, 2, 8, 1, 9};

    // Range-based for loop
    std::cout << "Original: ";
    for (const auto& n : nums) {
        std::cout << n << " ";
    }
    std::cout << "\n";

    // Push back
    nums.push_back(42);
    std::cout << "After push_back(42): size=" << nums.size() << "\n";

    // Sort
    std::sort(nums.begin(), nums.end());
    std::cout << "Sorted: ";
    for (const auto& n : nums) {
        std::cout << n << " ";
    }
    std::cout << "\n";

    // Erase
    nums.erase(nums.begin() + 2);  // Remove 3rd element
    std::cout << "After erase(index 2): ";
    for (const auto& n : nums) {
        std::cout << n << " ";
    }
    std::cout << "\n";
}

// ============ std::map (ordered) ============
void demo_map() {
    std::cout << "\n=== std::map ===\n";

    std::map<std::string, int> scores;
    scores["Alice"] = 95;
    scores["Bob"] = 87;
    scores["Charlie"] = 92;

    // Iterate (ordered by key)
    std::cout << "Scores (sorted by name):\n";
    for (const auto& [name, score] : scores) {
        std::cout << "  " << name << ": " << score << "\n";
    }

    // Find
    if (scores.find("Bob") != scores.end()) {
        std::cout << "Bob's score: " << scores["Bob"] << "\n";
    }

    // Insert or update
    scores.insert_or_assign("Alice", 98);
    std::cout << "Updated Alice's score: " << scores["Alice"] << "\n";
}

// ============ std::unordered_map (hash table) ============
void demo_unordered_map() {
    std::cout << "\n=== std::unordered_map ===\n";

    std::unordered_map<std::string, int> word_count;
    std::vector<std::string> words = {"apple", "banana", "apple", "cherry", "banana", "apple"};

    for (const auto& word : words) {
        word_count[word]++;
    }

    std::cout << "Word frequencies:\n";
    for (const auto& [word, count] : word_count) {
        std::cout << "  " << word << ": " << count << "\n";
    }
}

// ============ std::set ============
void demo_set() {
    std::cout << "\n=== std::set ===\n";

    std::set<int> unique_nums = {5, 2, 8, 2, 1, 5, 9};

    std::cout << "Unique numbers (sorted): ";
    for (const auto& n : unique_nums) {
        std::cout << n << " ";
    }
    std::cout << "\n";

    // Insert
    auto [it, inserted] = unique_nums.insert(3);
    std::cout << "Insert 3: " << (inserted ? "success" : "already exists") << "\n";

    auto [it2, inserted2] = unique_nums.insert(5);
    std::cout << "Insert 5: " << (inserted2 ? "success" : "already exists") << "\n";

    // Contains (C++20)
    std::cout << "Contains 8: " << unique_nums.contains(8) << "\n";
    std::cout << "Contains 100: " << unique_nums.contains(100) << "\n";
}

// ============ STL Algorithms ============
void demo_algorithms() {
    std::cout << "\n=== STL Algorithms ===\n";

    std::vector<int> nums = {1, 2, 3, 4, 5};

    // std::transform with lambda
    std::vector<int> squared;
    std::transform(nums.begin(), nums.end(), std::back_inserter(squared),
                   [](int x) { return x * x; });

    std::cout << "Squared: ";
    for (const auto& n : squared) {
        std::cout << n << " ";
    }
    std::cout << "\n";

    // std::accumulate (sum)
    int sum = std::accumulate(nums.begin(), nums.end(), 0);
    std::cout << "Sum: " << sum << "\n";

    // std::accumulate (product)
    int product = std::accumulate(nums.begin(), nums.end(), 1, std::multiplies<int>());
    std::cout << "Product: " << product << "\n";

    // std::find_if with lambda
    auto it = std::find_if(nums.begin(), nums.end(), [](int x) { return x > 3; });
    if (it != nums.end()) {
        std::cout << "First element > 3: " << *it << "\n";
    }

    // std::count_if
    int count_even = std::count_if(nums.begin(), nums.end(), [](int x) { return x % 2 == 0; });
    std::cout << "Count even numbers: " << count_even << "\n";

    // std::all_of, std::any_of, std::none_of
    bool all_positive = std::all_of(nums.begin(), nums.end(), [](int x) { return x > 0; });
    std::cout << "All positive: " << all_positive << "\n";
}

// ============ std::ranges (C++20) ============
void demo_ranges() {
    std::cout << "\n=== std::ranges (C++20) ===\n";

    std::vector<int> nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Filter and transform using views
    auto even_squared = nums
        | std::views::filter([](int x) { return x % 2 == 0; })
        | std::views::transform([](int x) { return x * x; });

    std::cout << "Even numbers squared: ";
    for (const auto& n : even_squared) {
        std::cout << n << " ";
    }
    std::cout << "\n";

    // Take first 5 elements
    auto first_five = nums | std::views::take(5);
    std::cout << "First 5: ";
    for (const auto& n : first_five) {
        std::cout << n << " ";
    }
    std::cout << "\n";

    // Drop first 5, take next 3
    auto middle = nums | std::views::drop(5) | std::views::take(3);
    std::cout << "Middle (drop 5, take 3): ";
    for (const auto& n : middle) {
        std::cout << n << " ";
    }
    std::cout << "\n";

    // Reverse view
    auto reversed = nums | std::views::reverse;
    std::cout << "Reversed: ";
    for (const auto& n : reversed) {
        std::cout << n << " ";
    }
    std::cout << "\n";
}

// ============ Lambda Expressions ============
void demo_lambdas() {
    std::cout << "\n=== Lambda Expressions ===\n";

    // Basic lambda
    auto add = [](int a, int b) { return a + b; };
    std::cout << "add(3, 4) = " << add(3, 4) << "\n";

    // Capture by value
    int x = 10;
    auto add_x = [x](int y) { return x + y; };
    std::cout << "add_x(5) = " << add_x(5) << "\n";

    // Capture by reference
    int count = 0;
    auto increment = [&count]() { count++; };
    increment();
    increment();
    std::cout << "Count after 2 increments: " << count << "\n";

    // Generic lambda (C++14)
    auto print = [](const auto& value) {
        std::cout << "Value: " << value << "\n";
    };
    print(42);
    print(3.14);
    print("hello");
}

// ============ Main ============
int main() {
    std::cout << "STL Containers and Algorithms Demo\n";
    std::cout << "===================================\n";

    demo_vector();
    demo_map();
    demo_unordered_map();
    demo_set();
    demo_algorithms();
    demo_ranges();
    demo_lambdas();

    std::cout << "\nAll demos completed!\n";
    return 0;
}
