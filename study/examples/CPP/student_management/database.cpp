#include "database.h"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <iomanip>

/**
 * @brief Add a new student to the database
 */
void StudentDatabase::addStudent(std::shared_ptr<Student> student) {
    // Check for duplicate ID
    auto it = std::find_if(students.begin(), students.end(),
        [&student](const std::shared_ptr<Student>& s) {
            return s->getId() == student->getId();
        });

    if (it != students.end()) {
        throw std::invalid_argument("Student with ID " + std::to_string(student->getId()) + " already exists");
    }

    students.push_back(student);
}

/**
 * @brief Remove a student by ID
 */
void StudentDatabase::removeStudent(int id) {
    auto it = std::find_if(students.begin(), students.end(),
        [id](const std::shared_ptr<Student>& s) {
            return s->getId() == id;
        });

    if (it == students.end()) {
        throw StudentNotFoundException(id);
    }

    students.erase(it);
}

/**
 * @brief Find a student by ID
 * @return Shared pointer to student
 * @throws StudentNotFoundException if not found
 */
std::shared_ptr<Student> StudentDatabase::findById(int id) const {
    auto it = std::find_if(students.begin(), students.end(),
        [id](const std::shared_ptr<Student>& s) {
            return s->getId() == id;
        });

    if (it == students.end()) {
        throw StudentNotFoundException(id);
    }

    return *it;
}

/**
 * @brief Find all students with matching name (case-sensitive substring)
 */
std::vector<std::shared_ptr<Student>> StudentDatabase::findByName(const std::string& name) const {
    std::vector<std::shared_ptr<Student>> results;

    std::copy_if(students.begin(), students.end(), std::back_inserter(results),
        [&name](const std::shared_ptr<Student>& s) {
            return s->getName().find(name) != std::string::npos;
        });

    return results;
}

/**
 * @brief Display all students
 */
void StudentDatabase::listAll() const {
    if (students.empty()) {
        std::cout << "No students in database.\n";
        return;
    }

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "STUDENT DATABASE (" << students.size() << " students)\n";
    std::cout << std::string(80, '=') << "\n";

    for (const auto& student : students) {
        std::cout << *student << "\n";
    }

    std::cout << std::string(80, '=') << "\n\n";
}

/**
 * @brief Sort students by ID (ascending)
 */
void StudentDatabase::sortById() {
    std::sort(students.begin(), students.end(),
        [](const std::shared_ptr<Student>& a, const std::shared_ptr<Student>& b) {
            return a->getId() < b->getId();
        });
}

/**
 * @brief Sort students by name (alphabetical)
 */
void StudentDatabase::sortByName() {
    std::sort(students.begin(), students.end(),
        [](const std::shared_ptr<Student>& a, const std::shared_ptr<Student>& b) {
            return a->getName() < b->getName();
        });
}

/**
 * @brief Sort students by GPA (descending - highest first)
 */
void StudentDatabase::sortByGpa() {
    std::sort(students.begin(), students.end(),
        [](const std::shared_ptr<Student>& a, const std::shared_ptr<Student>& b) {
            return a->getGpa() > b->getGpa();
        });
}

/**
 * @brief Calculate average GPA of all students
 */
double StudentDatabase::calculateAverageGpa() const {
    if (students.empty()) {
        return 0.0;
    }

    double sum = std::accumulate(students.begin(), students.end(), 0.0,
        [](double total, const std::shared_ptr<Student>& s) {
            return total + s->getGpa();
        });

    return sum / students.size();
}

/**
 * @brief Count students by major
 * @return Map of major -> student count
 */
std::map<std::string, int> StudentDatabase::countByMajor() const {
    std::map<std::string, int> majorCount;

    for (const auto& student : students) {
        majorCount[student->getMajor()]++;
    }

    return majorCount;
}

/**
 * @brief Save database to CSV file
 */
void StudentDatabase::saveToFile(const std::string& filename) const {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Write header
    file << "id,name,major,gpa\n";

    // Write student data
    for (const auto& student : students) {
        file << student->toCSV() << "\n";
    }

    file.close();
    std::cout << "Database saved to " << filename << " (" << students.size() << " students)\n";
}

/**
 * @brief Load database from CSV file
 */
void StudentDatabase::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    students.clear();
    std::string line;

    // Skip header
    std::getline(file, line);

    // Read student data
    int count = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        try {
            auto student = std::make_shared<Student>(Student::fromCSV(line));
            students.push_back(student);
            count++;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to parse line: " << line << " (" << e.what() << ")\n";
        }
    }

    file.close();
    std::cout << "Database loaded from " << filename << " (" << count << " students)\n";
}
