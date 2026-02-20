#!/usr/bin/env bash
set -euo pipefail

# Math Library for Testing
# A collection of mathematical functions demonstrating testable bash code
# Can be sourced by other scripts or run standalone for demonstration

# ============================================================================
# Basic Arithmetic Operations
# ============================================================================

# Add two numbers
# Args: $1 - first number, $2 - second number
# Returns: sum of the two numbers
add() {
    local a=$1
    local b=$2
    echo $((a + b))
}

# Subtract two numbers
# Args: $1 - first number, $2 - second number
# Returns: difference (first - second)
subtract() {
    local a=$1
    local b=$2
    echo $((a - b))
}

# Multiply two numbers
# Args: $1 - first number, $2 - second number
# Returns: product of the two numbers
multiply() {
    local a=$1
    local b=$2
    echo $((a * b))
}

# Divide two numbers
# Args: $1 - dividend, $2 - divisor
# Returns: quotient (integer division)
# Exits with error if divisor is zero
divide() {
    local a=$1
    local b=$2

    if [[ $b -eq 0 ]]; then
        echo "Error: Division by zero" >&2
        return 1
    fi

    echo $((a / b))
}

# ============================================================================
# Advanced Mathematical Functions
# ============================================================================

# Calculate factorial of a number
# Args: $1 - non-negative integer
# Returns: factorial of the number
factorial() {
    local n=$1

    if [[ $n -lt 0 ]]; then
        echo "Error: Factorial not defined for negative numbers" >&2
        return 1
    fi

    if [[ $n -eq 0 ]] || [[ $n -eq 1 ]]; then
        echo 1
        return 0
    fi

    local result=1
    local i
    for ((i = 2; i <= n; i++)); do
        result=$((result * i))
    done

    echo $result
}

# Check if a number is prime
# Args: $1 - positive integer
# Returns: 0 (true) if prime, 1 (false) otherwise
is_prime() {
    local n=$1

    if [[ $n -lt 2 ]]; then
        return 1
    fi

    if [[ $n -eq 2 ]]; then
        return 0
    fi

    if [[ $((n % 2)) -eq 0 ]]; then
        return 1
    fi

    local i
    local sqrt_n=$(awk "BEGIN {print int(sqrt($n))}")

    for ((i = 3; i <= sqrt_n; i += 2)); do
        if [[ $((n % i)) -eq 0 ]]; then
            return 1
        fi
    done

    return 0
}

# Calculate greatest common divisor (GCD)
# Args: $1 - first number, $2 - second number
# Returns: GCD of the two numbers
gcd() {
    local a=${1#-}  # Remove negative sign if present
    local b=${2#-}

    while [[ $b -ne 0 ]]; do
        local temp=$b
        b=$((a % b))
        a=$temp
    done

    echo $a
}

# Calculate least common multiple (LCM)
# Args: $1 - first number, $2 - second number
# Returns: LCM of the two numbers
lcm() {
    local a=${1#-}  # Remove negative sign if present
    local b=${2#-}

    if [[ $a -eq 0 ]] || [[ $b -eq 0 ]]; then
        echo 0
        return 0
    fi

    local gcd_val
    gcd_val=$(gcd "$a" "$b")
    echo $(( (a * b) / gcd_val ))
}

# ============================================================================
# Demo Functions (only run when executed, not sourced)
# ============================================================================

demo_basic_operations() {
    echo "=== Basic Arithmetic Operations ==="
    echo "add 10 5 = $(add 10 5)"
    echo "subtract 10 5 = $(subtract 10 5)"
    echo "multiply 10 5 = $(multiply 10 5)"
    echo "divide 10 5 = $(divide 10 5)"
    echo
}

demo_advanced_functions() {
    echo "=== Advanced Mathematical Functions ==="
    echo "factorial 5 = $(factorial 5)"
    echo "factorial 0 = $(factorial 0)"

    if is_prime 17; then
        echo "17 is prime"
    else
        echo "17 is not prime"
    fi

    if is_prime 20; then
        echo "20 is prime"
    else
        echo "20 is not prime"
    fi

    echo "gcd 48 18 = $(gcd 48 18)"
    echo "lcm 12 18 = $(lcm 12 18)"
    echo
}

demo_error_handling() {
    echo "=== Error Handling ==="
    echo "Testing division by zero:"
    if ! divide 10 0; then
        echo "Correctly caught division by zero error"
    fi

    echo
    echo "Testing negative factorial:"
    if ! factorial -5; then
        echo "Correctly caught negative factorial error"
    fi
    echo
}

# Main execution (only if script is run directly, not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Math Library Demonstration"
    echo "=========================="
    echo

    demo_basic_operations
    demo_advanced_functions
    demo_error_handling

    echo "To use this library in other scripts, source it:"
    echo "  source math_lib.sh"
fi
