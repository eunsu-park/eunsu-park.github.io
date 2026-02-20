#ifndef DATABASE_H
#define DATABASE_H

#include "student.h"
#include <vector>
#include <memory>
#include <string>
#include <map>

/**
 * @brief Custom exception for student not found errors
 */
class StudentNotFoundException : public std::exception {
private:
    std::string message;
public:
    explicit StudentNotFoundException(int id)
        : message("Student with ID " + std::to_string(id) + " not found") {}
    explicit StudentNotFoundException(const std::string& name)
        : message("Student with name '" + name + "' not found") {}
    const char* what() const noexcept override { return message.c_str(); }
};

/**
 * @brief Student database management system
 *
 * This class demonstrates:
 * - Smart pointers (shared_ptr) for memory management
 * - STL containers (vector, map)
 * - STL algorithms (find_if, sort, accumulate, count_if)
 * - File I/O with CSV format
 * - Exception handling
 */
class StudentDatabase {
private:
    std::vector<std::shared_ptr<Student>> students;

public:
    // CRUD operations
    void addStudent(std::shared_ptr<Student> student);
    void removeStudent(int id);
    std::shared_ptr<Student> findById(int id) const;
    std::vector<std::shared_ptr<Student>> findByName(const std::string& name) const;

    // List and sort operations
    void listAll() const;
    void sortById();
    void sortByName();
    void sortByGpa();

    // Statistics
    double calculateAverageGpa() const;
    std::map<std::string, int> countByMajor() const;
    size_t size() const { return students.size(); }

    // File I/O
    void saveToFile(const std::string& filename) const;
    void loadFromFile(const std::string& filename);
};

#endif // DATABASE_H
