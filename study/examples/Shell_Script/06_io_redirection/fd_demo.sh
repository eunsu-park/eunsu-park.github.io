#!/usr/bin/env bash
set -euo pipefail

# File Descriptor Demonstrations
# Shows advanced usage of file descriptors for I/O redirection

echo "=== File Descriptor Demonstrations ==="
echo

# --- Demo 1: Custom file descriptors ---
demo_custom_fd() {
    echo "1. Custom File Descriptors"
    echo "--------------------------"

    local output_file="/tmp/fd_demo_output.txt"
    local error_file="/tmp/fd_demo_error.txt"

    # Open custom file descriptors
    exec 3>"$output_file"  # FD 3 for normal output
    exec 4>"$error_file"   # FD 4 for error output

    echo "Writing to FD 3 (output file)" >&3
    echo "Writing to FD 4 (error file)" >&4

    echo "More output data" >&3
    echo "More error data" >&4

    # Close custom file descriptors
    exec 3>&-
    exec 4>&-

    echo "Files created:"
    echo "  Output file: $output_file"
    cat "$output_file"

    echo
    echo "  Error file: $error_file"
    cat "$error_file"

    # Cleanup
    rm -f "$output_file" "$error_file"

    echo
}

# --- Demo 2: Redirecting stderr separately ---
demo_separate_stderr() {
    echo "2. Redirecting Stderr Separately"
    echo "---------------------------------"

    local stdout_file="/tmp/stdout.txt"
    local stderr_file="/tmp/stderr.txt"

    # Function that outputs to both stdout and stderr
    mixed_output() {
        echo "This goes to stdout"
        echo "This goes to stderr" >&2
        echo "More stdout"
        echo "More stderr" >&2
    }

    echo "Calling function with mixed output..."
    mixed_output 1>"$stdout_file" 2>"$stderr_file"

    echo
    echo "STDOUT contents:"
    cat "$stdout_file"

    echo
    echo "STDERR contents:"
    cat "$stderr_file"

    # Cleanup
    rm -f "$stdout_file" "$stderr_file"

    echo
}

# --- Demo 3: Swapping stdout and stderr ---
demo_swap_stdout_stderr() {
    echo "3. Swapping Stdout and Stderr"
    echo "-----------------------------"

    # Function with mixed output
    test_output() {
        echo "STDOUT message"
        echo "STDERR message" >&2
    }

    echo "Normal output:"
    test_output 2>&1 | sed 's/^/  /'

    echo
    echo "With stdout and stderr swapped (3>&1 1>&2 2>&3):"
    (test_output 3>&1 1>&2 2>&3) 2>&1 | sed 's/^/  /'

    echo
}

# --- Demo 4: Logging to both console and file ---
demo_tee_with_fd() {
    echo "4. Logging to Console and File Simultaneously"
    echo "----------------------------------------------"

    local log_file="/tmp/dual_output.log"

    # Open log file on FD 3
    exec 3>"$log_file"

    # Function to log to both console and file
    dual_log() {
        local message="$*"
        echo "$message" | tee /dev/fd/3
    }

    dual_log "This message appears on console and in log file"
    dual_log "Second message"
    dual_log "Third message"

    # Close log file
    exec 3>&-

    echo
    echo "Log file contents:"
    cat "$log_file"

    # Cleanup
    rm -f "$log_file"

    echo
}

# --- Demo 5: Reading from and writing to same file descriptor ---
demo_readwrite_fd() {
    echo "5. Read/Write File Descriptor"
    echo "-----------------------------"

    local data_file="/tmp/rw_data.txt"

    # Create initial data
    cat > "$data_file" << EOF
Line 1: Original
Line 2: Original
Line 3: Original
EOF

    echo "Original file:"
    cat "$data_file"

    echo
    echo "Opening file for read/write on FD 3..."

    # Open file for reading and writing
    exec 3<> "$data_file"

    # Read first line
    read -r line <&3
    echo "Read: $line"

    # Write to file (appends)
    echo "Line 4: Added via FD 3" >&3

    # Close FD
    exec 3>&-

    echo
    echo "Modified file:"
    cat "$data_file"

    # Cleanup
    rm -f "$data_file"

    echo
}

# --- Demo 6: Saving and restoring stdout ---
demo_save_restore_stdout() {
    echo "6. Saving and Restoring Stdout"
    echo "-------------------------------"

    local temp_file="/tmp/redirected_output.txt"

    echo "This goes to original stdout"

    # Save original stdout to FD 6
    exec 6>&1

    # Redirect stdout to file
    exec 1>"$temp_file"

    echo "This goes to the file (not visible on console)"
    echo "Another line to the file"

    # Restore original stdout
    exec 1>&6

    # Close FD 6
    exec 6>&-

    echo "This goes to original stdout again"

    echo
    echo "File contents:"
    cat "$temp_file"

    # Cleanup
    rm -f "$temp_file"

    echo
}

# --- Demo 7: Here-document with file descriptor ---
demo_heredoc_fd() {
    echo "7. Here-document with File Descriptor"
    echo "--------------------------------------"

    # Assign here-document to FD 3
    exec 3<<EOF
First line from here-doc
Second line from here-doc
Third line from here-doc
EOF

    echo "Reading from here-document via FD 3:"
    while IFS= read -r line <&3; do
        echo "  > $line"
    done

    # Close FD 3
    exec 3>&-

    echo
}

# --- Demo 8: Noclobber and forcing overwrite ---
demo_noclobber() {
    echo "8. Noclobber and Forcing Overwrite"
    echo "-----------------------------------"

    local test_file="/tmp/noclobber_test.txt"

    echo "Original content" > "$test_file"

    # Enable noclobber
    set -C

    echo "Noclobber enabled (set -C)"
    echo "Trying to overwrite existing file..."

    if echo "Overwrite attempt" > "$test_file" 2>/dev/null; then
        echo "  ✗ Should have failed!"
    else
        echo "  ✓ Prevented overwrite (as expected)"
    fi

    echo "Forcing overwrite with >|..."
    echo "Forced overwrite" >| "$test_file"
    echo "  ✓ Success"

    # Disable noclobber
    set +C

    echo "Noclobber disabled (set +C)"
    echo "Normal overwrite works again" > "$test_file"
    echo "  ✓ Overwrite successful"

    # Cleanup
    rm -f "$test_file"

    echo
}

# Main execution
main() {
    demo_custom_fd
    demo_separate_stderr
    demo_swap_stdout_stderr
    demo_tee_with_fd
    demo_readwrite_fd
    demo_save_restore_stdout
    demo_heredoc_fd
    demo_noclobber

    echo "=== All Demos Complete ==="
}

main "$@"
