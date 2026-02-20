#!/usr/bin/env bash
set -euo pipefail

# Process Substitution Examples
# Demonstrates <(...) and >(...) for advanced I/O patterns

echo "=== Process Substitution Demonstrations ==="
echo

# --- Demo 1: Comparing output of two commands ---
demo_diff_commands() {
    echo "1. Comparing Output of Two Commands"
    echo "------------------------------------"

    # Create two sample files
    cat > /tmp/file1.txt << EOF
apple
banana
cherry
date
elderberry
EOF

    cat > /tmp/file2.txt << EOF
apple
blueberry
cherry
date
fig
EOF

    echo "Comparing two files with diff:"
    echo

    # Traditional approach (requires temp files)
    echo "File 1 contents:"
    cat /tmp/file1.txt
    echo
    echo "File 2 contents:"
    cat /tmp/file2.txt
    echo

    # Using process substitution - compare sorted versions
    echo "Diff output (process substitution):"
    if diff <(sort /tmp/file1.txt) <(sort /tmp/file2.txt); then
        echo "Files are identical"
    else
        echo "(showing differences above)"
    fi

    echo

    # Another example: compare directory listings
    echo "Comparing directory contents:"
    mkdir -p /tmp/dir1 /tmp/dir2
    touch /tmp/dir1/{a,b,c}.txt
    touch /tmp/dir2/{b,c,d}.txt

    echo "Files only in dir1:"
    comm -23 <(ls /tmp/dir1) <(ls /tmp/dir2)

    echo "Files only in dir2:"
    comm -13 <(ls /tmp/dir1) <(ls /tmp/dir2)

    echo "Files in both:"
    comm -12 <(ls /tmp/dir1) <(ls /tmp/dir2)

    # Cleanup
    rm -rf /tmp/file1.txt /tmp/file2.txt /tmp/dir1 /tmp/dir2

    echo
}

# --- Demo 2: Multiple inputs to a command ---
demo_multiple_inputs() {
    echo "2. Feeding Multiple Inputs to a Command"
    echo "----------------------------------------"

    # Use paste to combine output from multiple commands
    echo "Combining outputs side-by-side with paste:"
    paste <(echo -e "1\n2\n3\n4\n5") \
          <(echo -e "one\ntwo\nthree\nfour\nfive") \
          <(echo -e "I\nII\nIII\nIV\nV")

    echo

    # Use join to merge related data
    echo "Joining related data:"

    # Process 1: user IDs and names
    # Process 2: user IDs and emails
    join <(echo -e "1 Alice\n2 Bob\n3 Charlie") \
         <(echo -e "1 alice@example.com\n2 bob@example.com\n3 charlie@example.com")

    echo
}

# --- Demo 3: Avoiding subshell variable issues ---
demo_avoid_subshell() {
    echo "3. Avoiding Subshell Variable Issues"
    echo "-------------------------------------"

    # Problem: Variables set in pipeline are lost (subshell)
    echo "Problem - variable in pipeline (lost):"
    count=0
    echo -e "a\nb\nc" | while read -r line; do
        ((count++))
    done
    echo "  Count after pipeline: $count (still 0!)"

    echo

    # Solution: Use process substitution (no subshell)
    echo "Solution - process substitution (preserved):"
    count=0
    while read -r line; do
        ((count++))
    done < <(echo -e "a\nb\nc")
    echo "  Count after process sub: $count (correct!)"

    echo
}

# --- Demo 4: Reading multiple streams simultaneously ---
demo_multiple_streams() {
    echo "4. Reading Multiple Streams Simultaneously"
    echo "------------------------------------------"

    # Read from two sources in parallel
    echo "Reading two files line-by-line in parallel:"

    # Create sample files
    seq 1 5 > /tmp/numbers.txt
    echo -e "one\ntwo\nthree\nfour\nfive" > /tmp/words.txt

    while IFS= read -r num <&3 && IFS= read -r word <&4; do
        echo "  $num: $word"
    done 3< /tmp/numbers.txt 4< /tmp/words.txt

    # Cleanup
    rm -f /tmp/numbers.txt /tmp/words.txt

    echo
}

# --- Demo 5: Process substitution for writing ---
demo_write_process_sub() {
    echo "5. Process Substitution for Writing >(cmd)"
    echo "-------------------------------------------"

    echo "Splitting output to multiple destinations:"

    # Generate data and send to multiple processors
    {
        echo "apple"
        echo "banana"
        echo "cherry"
        echo "date"
    } | tee >(grep 'a' > /tmp/has_a.txt) \
            >(grep 'e' > /tmp/has_e.txt) \
            >(wc -l > /tmp/count.txt) \
            > /dev/null

    # Small delay to ensure all processes complete
    sleep 0.1

    echo "Words containing 'a':"
    cat /tmp/has_a.txt

    echo
    echo "Words containing 'e':"
    cat /tmp/has_e.txt

    echo
    echo "Total word count:"
    cat /tmp/count.txt

    # Cleanup
    rm -f /tmp/has_a.txt /tmp/has_e.txt /tmp/count.txt

    echo
}

# --- Demo 6: Complex data processing pipeline ---
demo_complex_pipeline() {
    echo "6. Complex Data Processing Pipeline"
    echo "------------------------------------"

    # Simulate log file
    cat > /tmp/access.log << EOF
192.168.1.1 - - [01/Jan/2024:10:00:00] "GET /index.html HTTP/1.1" 200 1234
192.168.1.2 - - [01/Jan/2024:10:00:01] "GET /about.html HTTP/1.1" 200 2345
192.168.1.1 - - [01/Jan/2024:10:00:02] "GET /contact.html HTTP/1.1" 404 0
192.168.1.3 - - [01/Jan/2024:10:00:03] "POST /api/data HTTP/1.1" 200 5678
192.168.1.2 - - [01/Jan/2024:10:00:04] "GET /index.html HTTP/1.1" 200 1234
EOF

    echo "Analyzing log file with process substitution:"
    echo

    # Compare successful vs failed requests
    echo "Status code comparison:"
    paste <(echo "Success (200):") <(grep " 200 " /tmp/access.log | wc -l)
    paste <(echo "Not Found (404):") <(grep " 404 " /tmp/access.log | wc -l)

    echo
    echo "Unique IP addresses:"
    awk '{print $1}' /tmp/access.log | sort -u

    echo
    echo "Request methods:"
    awk '{print $6}' /tmp/access.log | tr -d '"' | sort | uniq -c

    # Cleanup
    rm -f /tmp/access.log

    echo
}

# --- Demo 7: Named pipe vs process substitution ---
demo_comparison() {
    echo "7. Named Pipe vs Process Substitution"
    echo "--------------------------------------"

    echo "Process substitution creates temporary named pipes:"

    # Show what process substitution looks like
    echo "Process substitution expands to:"
    echo "  <(cmd) might expand to: /dev/fd/63"

    # Demonstrate by echoing the expansion
    bash -c 'echo "Expansion: " <(echo test)'

    echo
    echo "This is equivalent to creating a named pipe, but automatic!"

    echo
}

# --- Demo 8: Practical example - log analysis ---
demo_log_analysis() {
    echo "8. Practical Example: Log Analysis"
    echo "-----------------------------------"

    # Generate sample log data
    {
        echo "2024-01-01 10:00:00 ERROR Database connection failed"
        echo "2024-01-01 10:00:05 INFO User login: alice"
        echo "2024-01-01 10:00:10 WARN High memory usage: 85%"
        echo "2024-01-01 10:00:15 ERROR Timeout on API call"
        echo "2024-01-01 10:00:20 INFO User login: bob"
        echo "2024-01-01 10:00:25 ERROR File not found: data.csv"
    } > /tmp/application.log

    echo "Log file analysis:"
    echo

    # Count by log level using process substitution
    echo "Log level counts:"
    while IFS= read -r level; do
        count=$(grep "$level" /tmp/application.log | wc -l)
        printf "  %-10s %d\n" "$level" "$count"
    done < <(awk '{print $3}' /tmp/application.log | sort -u)

    echo
    echo "Error messages only:"
    grep "ERROR" /tmp/application.log | sed 's/^/  /'

    # Cleanup
    rm -f /tmp/application.log

    echo
}

# Main execution
main() {
    demo_diff_commands
    demo_multiple_inputs
    demo_avoid_subshell
    demo_multiple_streams
    demo_write_process_sub
    demo_complex_pipeline
    demo_comparison
    demo_log_analysis

    echo "=== All Demos Complete ==="
}

main "$@"
