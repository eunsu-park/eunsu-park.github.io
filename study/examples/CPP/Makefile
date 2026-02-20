# Makefile for C++ Examples
# Compile with C++20 standard and common flags

CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2
LDFLAGS = -pthread

# Source files
SOURCES = 01_modern_cpp.cpp \
          02_stl_containers.cpp \
          03_smart_pointers.cpp \
          04_threading.cpp \
          05_design_patterns.cpp \
          06_templates.cpp \
          07_move_semantics.cpp

# Object files (replace .cpp with .o)
OBJECTS = $(SOURCES:.cpp=.o)

# Executables (replace .cpp with no extension)
EXECUTABLES = $(SOURCES:.cpp=)

# Default target: build all
.PHONY: all
all: $(EXECUTABLES)

# Pattern rule: compile .cpp to executable
%: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Individual targets (for convenience)
.PHONY: modern stl smart threading patterns templates move
modern: 01_modern_cpp
stl: 02_stl_containers
smart: 03_smart_pointers
threading: 04_threading
patterns: 05_design_patterns
templates: 06_templates
move: 07_move_semantics

# Run all examples
.PHONY: run
run: all
	@echo "\n=== Running all examples ===\n"
	@for exe in $(EXECUTABLES); do \
		echo "\n--- Running $$exe ---\n"; \
		./$$exe; \
		echo "\n"; \
	done

# Run specific example
.PHONY: run-%
run-%: %
	@echo "\n=== Running $< ===\n"
	./$<

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(EXECUTABLES) $(OBJECTS) *.out core /tmp/raii_test.txt /tmp/test.txt

# Display help
.PHONY: help
help:
	@echo "C++ Examples Makefile"
	@echo "====================="
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build all examples (default)"
	@echo "  clean        - Remove all build artifacts"
	@echo "  run          - Build and run all examples"
	@echo "  run-<name>   - Build and run specific example"
	@echo ""
	@echo "Individual examples:"
	@echo "  modern       - Build 01_modern_cpp"
	@echo "  stl          - Build 02_stl_containers"
	@echo "  smart        - Build 03_smart_pointers"
	@echo "  threading    - Build 04_threading"
	@echo "  patterns     - Build 05_design_patterns"
	@echo "  templates    - Build 06_templates"
	@echo "  move         - Build 07_move_semantics"
	@echo ""
	@echo "Examples:"
	@echo "  make                  # Build all"
	@echo "  make modern           # Build modern C++ demo"
	@echo "  make run-threading    # Build and run threading demo"
	@echo "  make clean            # Clean all build artifacts"

# Compiler and flags info
.PHONY: info
info:
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS)"
	@echo "Linker flags: $(LDFLAGS)"
	@$(CXX) --version
