#include "student.h"
#include <sstream>
#include <iomanip>
#include <stdexcept>

/**
 * @brief Construct a new Student object
 */
Student::Student(int id, const std::string& name, const std::string& major, double gpa)
    : id(id), name(name), major(major), gpa(gpa) {
    if (gpa < 0.0 || gpa > 4.0) {
        throw std::invalid_argument("GPA must be between 0.0 and 4.0");
    }
}

/**
 * @brief Convert student data to CSV format
 * @return CSV string representation
 */
std::string Student::toCSV() const {
    std::ostringstream oss;
    oss << id << "," << name << "," << major << "," << std::fixed << std::setprecision(2) << gpa;
    return oss.str();
}

/**
 * @brief Create Student object from CSV string
 * @param csvLine CSV formatted string (id,name,major,gpa)
 * @return Student object
 */
Student Student::fromCSV(const std::string& csvLine) {
    std::istringstream iss(csvLine);
    std::string token;
    int id;
    std::string name, major;
    double gpa;

    // Parse CSV (simple implementation - assumes no commas in fields)
    std::getline(iss, token, ',');
    id = std::stoi(token);

    std::getline(iss, name, ',');
    std::getline(iss, major, ',');

    std::getline(iss, token, ',');
    gpa = std::stod(token);

    return Student(id, name, major, gpa);
}

/**
 * @brief Stream insertion operator for pretty printing
 */
std::ostream& operator<<(std::ostream& os, const Student& student) {
    os << "ID: " << std::setw(5) << student.id
       << " | Name: " << std::setw(20) << std::left << student.name
       << " | Major: " << std::setw(15) << student.major
       << " | GPA: " << std::fixed << std::setprecision(2) << student.gpa;
    return os;
}
