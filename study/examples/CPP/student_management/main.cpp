#include "database.h"
#include <iostream>
#include <limits>
#include <iomanip>

/**
 * @brief Clear input buffer after invalid input
 */
void clearInput() {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

/**
 * @brief Display main menu
 */
void displayMenu() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════╗\n";
    std::cout << "║   STUDENT MANAGEMENT SYSTEM            ║\n";
    std::cout << "╠════════════════════════════════════════╣\n";
    std::cout << "║ 1. Add Student                         ║\n";
    std::cout << "║ 2. Remove Student                      ║\n";
    std::cout << "║ 3. Find Student by ID                  ║\n";
    std::cout << "║ 4. Find Students by Name               ║\n";
    std::cout << "║ 5. List All Students                   ║\n";
    std::cout << "║ 6. Sort Students                       ║\n";
    std::cout << "║ 7. Statistics                          ║\n";
    std::cout << "║ 8. Save to File                        ║\n";
    std::cout << "║ 9. Load from File                      ║\n";
    std::cout << "║ 0. Exit                                ║\n";
    std::cout << "╚════════════════════════════════════════╝\n";
    std::cout << "Choice: ";
}

/**
 * @brief Add a new student interactively
 */
void addStudentInteractive(StudentDatabase& db) {
    int id;
    std::string name, major;
    double gpa;

    std::cout << "\nAdd New Student\n";
    std::cout << "---------------\n";
    std::cout << "Student ID: ";
    std::cin >> id;
    clearInput();

    std::cout << "Name: ";
    std::getline(std::cin, name);

    std::cout << "Major: ";
    std::getline(std::cin, major);

    std::cout << "GPA (0.0-4.0): ";
    std::cin >> gpa;

    try {
        auto student = std::make_shared<Student>(id, name, major, gpa);
        db.addStudent(student);
        std::cout << "Student added successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

/**
 * @brief Display sort submenu
 */
void sortSubmenu(StudentDatabase& db) {
    std::cout << "\nSort By:\n";
    std::cout << "1. ID\n";
    std::cout << "2. Name\n";
    std::cout << "3. GPA (descending)\n";
    std::cout << "Choice: ";

    int choice;
    std::cin >> choice;

    switch (choice) {
        case 1:
            db.sortById();
            std::cout << "Sorted by ID.\n";
            break;
        case 2:
            db.sortByName();
            std::cout << "Sorted by Name.\n";
            break;
        case 3:
            db.sortByGpa();
            std::cout << "Sorted by GPA (descending).\n";
            break;
        default:
            std::cout << "Invalid choice.\n";
    }
}

/**
 * @brief Display statistics
 */
void showStatistics(const StudentDatabase& db) {
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "STATISTICS\n";
    std::cout << std::string(50, '=') << "\n";

    std::cout << "Total Students: " << db.size() << "\n";

    if (db.size() > 0) {
        std::cout << "Average GPA: " << std::fixed << std::setprecision(2)
                  << db.calculateAverageGpa() << "\n\n";

        std::cout << "Students by Major:\n";
        auto majorCounts = db.countByMajor();
        for (const auto& [major, count] : majorCounts) {
            std::cout << "  " << std::setw(20) << std::left << major
                      << ": " << count << " student(s)\n";
        }
    }

    std::cout << std::string(50, '=') << "\n";
}

/**
 * @brief Main application entry point
 */
int main() {
    StudentDatabase db;
    int choice;

    std::cout << "Welcome to Student Management System (C++17)\n";
    std::cout << "Demonstrates: STL, Smart Pointers, File I/O, Exception Handling\n";

    while (true) {
        displayMenu();
        std::cin >> choice;

        if (std::cin.fail()) {
            clearInput();
            std::cout << "Invalid input. Please enter a number.\n";
            continue;
        }

        try {
            switch (choice) {
                case 0:
                    std::cout << "Goodbye!\n";
                    return 0;

                case 1:
                    addStudentInteractive(db);
                    break;

                case 2: {
                    int id;
                    std::cout << "Enter Student ID to remove: ";
                    std::cin >> id;
                    db.removeStudent(id);
                    std::cout << "Student removed successfully.\n";
                    break;
                }

                case 3: {
                    int id;
                    std::cout << "Enter Student ID: ";
                    std::cin >> id;
                    auto student = db.findById(id);
                    std::cout << "\nFound: " << *student << "\n";
                    break;
                }

                case 4: {
                    std::string name;
                    std::cout << "Enter name (or partial name): ";
                    clearInput();
                    std::getline(std::cin, name);
                    auto results = db.findByName(name);

                    if (results.empty()) {
                        std::cout << "No students found matching '" << name << "'\n";
                    } else {
                        std::cout << "\nFound " << results.size() << " student(s):\n";
                        for (const auto& s : results) {
                            std::cout << *s << "\n";
                        }
                    }
                    break;
                }

                case 5:
                    db.listAll();
                    break;

                case 6:
                    sortSubmenu(db);
                    break;

                case 7:
                    showStatistics(db);
                    break;

                case 8: {
                    std::string filename;
                    std::cout << "Enter filename (e.g., students.csv): ";
                    std::cin >> filename;
                    db.saveToFile(filename);
                    break;
                }

                case 9: {
                    std::string filename;
                    std::cout << "Enter filename: ";
                    std::cin >> filename;
                    db.loadFromFile(filename);
                    break;
                }

                default:
                    std::cout << "Invalid choice. Please try again.\n";
            }

        } catch (const StudentNotFoundException& e) {
            std::cerr << "Error: " << e.what() << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
        }
    }

    return 0;
}
