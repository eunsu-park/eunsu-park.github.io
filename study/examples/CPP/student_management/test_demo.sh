#!/bin/bash
# Automated test demonstration of the Student Management System

echo "==================================="
echo "Student Management System Demo"
echo "==================================="
echo ""

# Create a temporary input file for automated testing
cat > test_input.txt <<EOF
9
sample_students.csv
5
7
6
3
5
3
1001
4
Alice
6
2
5
8
output.csv
1
1009
Test Student
Chemistry
3.55
5
0
EOF

# Run the program with the test input
echo "Running automated test sequence..."
echo ""
./student_manager < test_input.txt

echo ""
echo "==================================="
echo "Test Complete!"
echo "==================================="
echo ""
echo "Generated files:"
ls -lh output.csv 2>/dev/null && echo "  - output.csv (saved database)"
echo ""

# Clean up
rm -f test_input.txt

echo "To run the interactive version, execute:"
echo "  ./student_manager"
