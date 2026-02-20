#!/usr/bin/env bash
set -euo pipefail

# Declare/Typeset Demonstrations
# Shows various uses of the declare builtin for variable attributes

echo "=== Declare/Typeset Demonstrations ==="
echo

# 1. Integer arithmetic with declare -i
echo "1. Integer Arithmetic (declare -i)"
echo "-----------------------------------"
declare -i num1=10
declare -i num2=20

echo "num1=$num1, num2=$num2"
num1=num1+5  # With -i, string assignment is evaluated as arithmetic
echo "After 'num1=num1+5': num1=$num1"

num2=num1*2
echo "After 'num2=num1*2': num2=$num2"

# Attempting non-integer assignment
num1="hello"  # Will be evaluated as 0
echo "After 'num1=\"hello\"': num1=$num1(non-integer evaluates to 0)"

echo
echo

# 2. Readonly variables with declare -r
echo "2. Readonly Variables (declare -r)"
echo "----------------------------------"
declare -r READONLY_VAR="This cannot be changed"
echo "READONLY_VAR=$READONLY_VAR"
echo "Attempting to change readonly variable..."

if ! READONLY_VAR="New value" 2>/dev/null; then
    echo "  ✗ Failed as expected (readonly variable)"
fi

echo
echo

# 3. Case conversion with declare -l/-u
echo "3. Case Conversion (declare -l/-u)"
echo "----------------------------------"

# Lowercase variable
declare -l lowercase_var
lowercase_var="HELLO WORLD"
echo "Set to 'HELLO WORLD', stored as: $lowercase_var"

lowercase_var="MiXeD CaSe"
echo "Set to 'MiXeD CaSe', stored as: $lowercase_var"

echo

# Uppercase variable
declare -u uppercase_var
uppercase_var="hello world"
echo "Set to 'hello world', stored as: $uppercase_var"

uppercase_var="MiXeD CaSe"
echo "Set to 'MiXeD CaSe', stored as: $uppercase_var"

echo
echo

# 4. Nameref with declare -n
echo "4. Name References (declare -n)"
echo "-------------------------------"

original_var="Original Value"
echo "original_var=$original_var"

# Create a nameref (pointer to another variable)
declare -n ref=original_var
echo "Created nameref 'ref' pointing to 'original_var'"
echo "ref=$ref"

echo
echo "Modifying through nameref..."
ref="Modified through reference"
echo "ref=$ref"
echo "original_var=$original_var (also changed!)"

echo
echo

# 5. Array declaration
echo "5. Array Declaration"
echo "--------------------"

# Indexed array
declare -a indexed_array=("apple" "banana" "cherry")
echo "Indexed array: ${indexed_array[@]}"
echo "Element 0: ${indexed_array[0]}"
echo "Element 1: ${indexed_array[1]}"

echo

# Associative array
declare -A assoc_array=(
    [name]="John Doe"
    [age]="30"
    [city]="New York"
)
echo "Associative array:"
for key in "${!assoc_array[@]}"; do
    echo "  $key = ${assoc_array[$key]}"
done

echo
echo

# 6. Export variables
echo "6. Export Variables (declare -x)"
echo "--------------------------------"
declare -x EXPORTED_VAR="I am exported to child processes"
echo "EXPORTED_VAR=$EXPORTED_VAR"

# Demonstrate in subshell
bash -c 'echo "In subshell: EXPORTED_VAR=$EXPORTED_VAR"'

# Non-exported variable
NON_EXPORTED="I stay in this shell"
bash -c 'echo "In subshell: NON_EXPORTED=$NON_EXPORTED" || echo "  ✗ Variable not available in subshell"'

echo
echo

# 7. List all declared variables with attributes
echo "7. Viewing Variable Attributes"
echo "------------------------------"
echo "Integer variables in this script:"
declare -i | grep -E "(num1|num2)" || echo "  (showing only num1, num2)"

echo
echo "Readonly variables in this script:"
declare -r | grep READONLY_VAR || echo "  (showing only READONLY_VAR)"

echo
echo "Exported variables (sample):"
declare -x | grep EXPORTED_VAR || echo "  (showing only EXPORTED_VAR)"

echo
echo

# 8. Combining attributes
echo "8. Combining Attributes"
echo "-----------------------"
declare -ir READONLY_INT=42
echo "Readonly integer: READONLY_INT=$READONLY_INT"

declare -lx lowercase_export="THIS WILL BE LOWERCASE AND EXPORTED"
echo "Lowercase exported: lowercase_export=$lowercase_export"
bash -c 'echo "In subshell: lowercase_export=$lowercase_export"'

echo
echo

# 9. Function-local variables
echo "9. Function-local Variables"
echo "---------------------------"

global_var="I am global"

demo_function() {
    local local_var="I am local to the function"
    global_var="Modified in function"

    echo "Inside function:"
    echo "  local_var=$local_var"
    echo "  global_var=$global_var"
}

echo "Before function call:"
echo "  global_var=$global_var"

demo_function

echo "After function call:"
echo "  global_var=$global_var"
echo "  local_var=${local_var:-<not defined>}"

echo
echo

# 10. Printing variable information
echo "10. Variable Information"
echo "------------------------"
declare -p num1 2>/dev/null || echo "num1 info not available"
declare -p READONLY_VAR 2>/dev/null || echo "READONLY_VAR info not available"
declare -p lowercase_var 2>/dev/null || echo "lowercase_var info not available"
declare -p assoc_array 2>/dev/null || echo "assoc_array info not available"

echo
echo "=== Demo Complete ==="
