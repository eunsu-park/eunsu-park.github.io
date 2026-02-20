# Makefile for Student Management System
# Demonstrates: Modern C++17, proper dependency tracking, clean builds

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Wpedantic -O2
TARGET = student_manager
OBJS = main.o student.o database.o

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
	@echo "Build complete: $(TARGET)"

# Compile main.cpp
main.o: main.cpp database.h student.h
	$(CXX) $(CXXFLAGS) -c main.cpp

# Compile database.cpp
database.o: database.cpp database.h student.h
	$(CXX) $(CXXFLAGS) -c database.cpp

# Compile student.cpp
student.o: student.cpp student.h
	$(CXX) $(CXXFLAGS) -c student.cpp

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)
	@echo "Clean complete"

# Phony targets (not actual files)
.PHONY: all clean run
