#!/usr/bin/env bats

# Bats Tests for Math Library
# Tests all functions in math_lib.sh for correctness and error handling

# Setup - run before each test
setup() {
    # Source the library to test
    load '../13_testing/math_lib'

    # Create temp directory for test files if needed
    TEST_TEMP_DIR="${BATS_TEST_TMPDIR}/math_lib_tests"
    mkdir -p "$TEST_TEMP_DIR"
}

# Teardown - run after each test
teardown() {
    # Clean up temp directory
    rm -rf "$TEST_TEMP_DIR"
}

# ============================================================================
# Tests for Basic Arithmetic Operations
# ============================================================================

@test "add: positive numbers" {
    result=$(add 5 3)
    [ "$result" -eq 8 ]
}

@test "add: negative numbers" {
    result=$(add -5 -3)
    [ "$result" -eq -8 ]
}

@test "add: mixed positive and negative" {
    result=$(add 10 -3)
    [ "$result" -eq 7 ]
}

@test "add: zero" {
    result=$(add 5 0)
    [ "$result" -eq 5 ]
}

@test "subtract: positive numbers" {
    result=$(subtract 10 3)
    [ "$result" -eq 7 ]
}

@test "subtract: negative result" {
    result=$(subtract 3 10)
    [ "$result" -eq -7 ]
}

@test "subtract: negative numbers" {
    result=$(subtract -5 -3)
    [ "$result" -eq -2 ]
}

@test "multiply: positive numbers" {
    result=$(multiply 6 7)
    [ "$result" -eq 42 ]
}

@test "multiply: by zero" {
    result=$(multiply 5 0)
    [ "$result" -eq 0 ]
}

@test "multiply: negative numbers" {
    result=$(multiply -4 -5)
    [ "$result" -eq 20 ]
}

@test "multiply: mixed signs" {
    result=$(multiply -4 5)
    [ "$result" -eq -20 ]
}

@test "divide: positive numbers" {
    result=$(divide 20 4)
    [ "$result" -eq 5 ]
}

@test "divide: integer division truncates" {
    result=$(divide 10 3)
    [ "$result" -eq 3 ]
}

@test "divide: negative numbers" {
    result=$(divide -20 4)
    [ "$result" -eq -5 ]
}

@test "divide: division by zero returns error" {
    run divide 10 0
    [ "$status" -eq 1 ]
    [[ "$output" =~ "Division by zero" ]]
}

# ============================================================================
# Tests for Factorial Function
# ============================================================================

@test "factorial: zero" {
    result=$(factorial 0)
    [ "$result" -eq 1 ]
}

@test "factorial: one" {
    result=$(factorial 1)
    [ "$result" -eq 1 ]
}

@test "factorial: small number" {
    result=$(factorial 5)
    [ "$result" -eq 120 ]
}

@test "factorial: larger number" {
    result=$(factorial 10)
    [ "$result" -eq 3628800 ]
}

@test "factorial: negative number returns error" {
    run factorial -5
    [ "$status" -eq 1 ]
    [[ "$output" =~ "negative numbers" ]]
}

# ============================================================================
# Tests for Prime Number Check
# ============================================================================

@test "is_prime: 2 is prime" {
    is_prime 2
    [ "$?" -eq 0 ]
}

@test "is_prime: 3 is prime" {
    is_prime 3
    [ "$?" -eq 0 ]
}

@test "is_prime: 17 is prime" {
    is_prime 17
    [ "$?" -eq 0 ]
}

@test "is_prime: 97 is prime" {
    is_prime 97
    [ "$?" -eq 0 ]
}

@test "is_prime: 1 is not prime" {
    run is_prime 1
    [ "$status" -eq 1 ]
}

@test "is_prime: 4 is not prime" {
    run is_prime 4
    [ "$status" -eq 1 ]
}

@test "is_prime: 20 is not prime" {
    run is_prime 20
    [ "$status" -eq 1 ]
}

@test "is_prime: 100 is not prime" {
    run is_prime 100
    [ "$status" -eq 1 ]
}

@test "is_prime: 0 is not prime" {
    run is_prime 0
    [ "$status" -eq 1 ]
}

# ============================================================================
# Tests for GCD (Greatest Common Divisor)
# ============================================================================

@test "gcd: simple case" {
    result=$(gcd 48 18)
    [ "$result" -eq 6 ]
}

@test "gcd: coprime numbers" {
    result=$(gcd 17 19)
    [ "$result" -eq 1 ]
}

@test "gcd: one number is multiple of other" {
    result=$(gcd 24 12)
    [ "$result" -eq 12 ]
}

@test "gcd: same numbers" {
    result=$(gcd 42 42)
    [ "$result" -eq 42 ]
}

@test "gcd: with zero" {
    result=$(gcd 15 0)
    [ "$result" -eq 15 ]
}

@test "gcd: negative numbers" {
    result=$(gcd -48 18)
    [ "$result" -eq 6 ]
}

# ============================================================================
# Tests for LCM (Least Common Multiple)
# ============================================================================

@test "lcm: simple case" {
    result=$(lcm 12 18)
    [ "$result" -eq 36 ]
}

@test "lcm: coprime numbers" {
    result=$(lcm 7 13)
    [ "$result" -eq 91 ]
}

@test "lcm: one number is multiple of other" {
    result=$(lcm 5 15)
    [ "$result" -eq 15 ]
}

@test "lcm: same numbers" {
    result=$(lcm 8 8)
    [ "$result" -eq 8 ]
}

@test "lcm: with zero" {
    result=$(lcm 5 0)
    [ "$result" -eq 0 ]
}

@test "lcm: negative numbers" {
    result=$(lcm -12 18)
    [ "$result" -eq 36 ]
}

# ============================================================================
# Integration Tests
# ============================================================================

@test "integration: factorial of prime" {
    # 7 is prime, factorial(7) = 5040
    is_prime 7
    [ "$?" -eq 0 ]

    result=$(factorial 7)
    [ "$result" -eq 5040 ]
}

@test "integration: gcd and lcm relationship" {
    # For any two numbers a and b: a * b = gcd(a, b) * lcm(a, b)
    a=24
    b=36

    gcd_val=$(gcd $a $b)
    lcm_val=$(lcm $a $b)

    product=$((a * b))
    gcd_lcm_product=$((gcd_val * lcm_val))

    [ "$product" -eq "$gcd_lcm_product" ]
}
