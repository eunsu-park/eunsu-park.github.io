#!/usr/bin/env bash
set -euo pipefail

# Demo script for logging and validation libraries

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the libraries
source "$SCRIPT_DIR/lib/logging.sh"
source "$SCRIPT_DIR/lib/validation.sh"

# Main demo
main() {
    log_section "Library Demo: Logging & Validation"

    # --- Logging Demo ---
    log_section "Logging Demo"

    log_info "This is an informational message"
    log_warn "This is a warning message"
    log_error "This is an error message"
    log_debug "This debug message won't show (default log level is INFO)"

    echo
    log_info "Changing log level to DEBUG..."
    set_log_level DEBUG

    log_debug "Now debug messages are visible!"
    log_info "Info still shows"

    echo
    log_info "Enabling file logging..."
    local log_file="/tmp/demo.log"
    enable_file_logging "$log_file"

    log_info "This message goes to both console and file"
    log_warn "Warnings are also logged to file"

    echo
    log_info "Log file contents:"
    log_separator
    cat "$log_file"
    log_separator

    disable_file_logging
    rm -f "$log_file"

    # Reset to INFO level
    set_log_level INFO

    echo
    log_section "Validation Demo"

    # Email validation
    log_info "Testing email validation:"
    local -a emails=(
        "user@example.com"
        "invalid.email"
        "test@domain.co.uk"
        "@invalid.com"
    )

    for email in "${emails[@]}"; do
        if validate_email "$email"; then
            log_info "  ✓ Valid email: $email"
        else
            log_warn "  ✗ Invalid email: $email"
        fi
    done

    echo
    log_info "Testing IP address validation:"
    local -a ips=(
        "192.168.1.1"
        "256.1.1.1"
        "10.0.0.1"
        "invalid.ip"
    )

    for ip in "${ips[@]}"; do
        if validate_ip "$ip"; then
            log_info "  ✓ Valid IP: $ip"
        else
            log_warn "  ✗ Invalid IP: $ip"
        fi
    done

    echo
    log_info "Testing port validation:"
    local -a ports=(
        "80"
        "8080"
        "65536"
        "-1"
        "abc"
    )

    for port in "${ports[@]}"; do
        if validate_port "$port"; then
            log_info "  ✓ Valid port: $port"
        else
            log_warn "  ✗ Invalid port: $port"
        fi
    done

    echo
    log_info "Testing integer checks:"
    local -a numbers=(
        "42"
        "-10"
        "0"
        "3.14"
        "abc"
    )

    for num in "${numbers[@]}"; do
        local checks=""
        is_integer "$num" && checks+="integer " || checks+="not-integer "
        is_positive "$num" && checks+="positive " || true
        is_non_negative "$num" && checks+="non-negative " || true

        log_info "  $num: $checks"
    done

    echo
    log_info "Testing range validation (0-100):"
    local -a range_tests=(
        "50"
        "0"
        "100"
        "101"
        "-1"
    )

    for num in "${range_tests[@]}"; do
        if is_in_range "$num" 0 100; then
            log_info "  ✓ $num is in range [0, 100]"
        else
            log_warn "  ✗ $num is out of range [0, 100]"
        fi
    done

    echo
    log_info "Testing URL validation:"
    local -a urls=(
        "https://www.example.com"
        "http://localhost:8080/path"
        "ftp://invalid.protocol"
        "not a url"
    )

    for url in "${urls[@]}"; do
        if validate_url "$url"; then
            log_info "  ✓ Valid URL: $url"
        else
            log_warn "  ✗ Invalid URL: $url"
        fi
    done

    echo
    log_info "Testing hostname validation:"
    local -a hostnames=(
        "example.com"
        "sub.domain.example.com"
        "invalid..hostname"
        "-invalid.com"
    )

    for hostname in "${hostnames[@]}"; do
        if validate_hostname "$hostname"; then
            log_info "  ✓ Valid hostname: $hostname"
        else
            log_warn "  ✗ Invalid hostname: $hostname"
        fi
    done

    echo
    log_info "Testing MAC address validation:"
    local -a macs=(
        "00:1A:2B:3C:4D:5E"
        "00-1A-2B-3C-4D-5E"
        "invalid:mac"
        "00:1A:2B:3C:4D"
    )

    for mac in "${macs[@]}"; do
        if validate_mac "$mac"; then
            log_info "  ✓ Valid MAC: $mac"
        else
            log_warn "  ✗ Invalid MAC: $mac"
        fi
    done

    echo
    log_info "Testing semantic version validation:"
    local -a versions=(
        "1.2.3"
        "2.0.0-beta"
        "1.0.0+build.123"
        "invalid.version"
    )

    for version in "${versions[@]}"; do
        if validate_semver "$version"; then
            log_info "  ✓ Valid semver: $version"
        else
            log_warn "  ✗ Invalid semver: $version"
        fi
    done

    echo
    log_info "Testing hex color validation:"
    local -a colors=(
        "#FF5733"
        "#F57"
        "FF5733"
        "#GGGGGG"
    )

    for color in "${colors[@]}"; do
        if validate_hex_color "$color"; then
            log_info "  ✓ Valid hex color: $color"
        else
            log_warn "  ✗ Invalid hex color: $color"
        fi
    done

    echo
    log_info "Testing command existence:"
    local -a commands=(
        "bash"
        "ls"
        "nonexistent_command_xyz"
    )

    for cmd in "${commands[@]}"; do
        if command_exists "$cmd"; then
            log_info "  ✓ Command exists: $cmd"
        else
            log_warn "  ✗ Command not found: $cmd"
        fi
    done

    echo
    log_info "Testing path validation:"
    local -a paths=(
        "/tmp"
        "/etc/passwd"
        "/nonexistent/path"
    )

    for path in "${paths[@]}"; do
        if validate_path "$path" "any"; then
            if validate_path "$path" "dir"; then
                log_info "  ✓ Directory exists: $path"
            else
                log_info "  ✓ File exists: $path"
            fi
        else
            log_warn "  ✗ Path not found: $path"
        fi
    done

    echo
    log_section "Demo Complete"
    log_info "Libraries are now loaded and ready to use"
    log_info "Available logging functions: log_debug, log_info, log_warn, log_error"
    log_info "Available validation functions: run 'declare -F | grep validate' to see all"
}

# Run the demo
main "$@"
