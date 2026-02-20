#ifndef STUDENT_H
#define STUDENT_H

#include <string>
#include <iostream>

/**
 * @brief Represents a student with basic academic information
 *
 * This class demonstrates:
 * - Encapsulation with private members and public accessors
 * - Operator overloading for comparison and output
 * - String serialization for persistence
 */
class Student {
private:
    int id;
    std::string name;
    std::string major;
    double gpa;

public:
    // Constructor
    Student(int id, const std::string& name, const std::string& major, double gpa);

    // Default constructor
    Student() : id(0), name(""), major(""), gpa(0.0) {}

    // Getters
    int getId() const { return id; }
    std::string getName() const { return name; }
    std::string getMajor() const { return major; }
    double getGpa() const { return gpa; }

    // Setters
    void setName(const std::string& newName) { name = newName; }
    void setMajor(const std::string& newMajor) { major = newMajor; }
    void setGpa(double newGpa) { gpa = newGpa; }

    // Comparison operators (for sorting)
    bool operator<(const Student& other) const { return id < other.id; }
    bool operator==(const Student& other) const { return id == other.id; }

    // Serialization
    std::string toCSV() const;
    static Student fromCSV(const std::string& csvLine);

    // Stream insertion operator
    friend std::ostream& operator<<(std::ostream& os, const Student& student);
};

#endif // STUDENT_H
