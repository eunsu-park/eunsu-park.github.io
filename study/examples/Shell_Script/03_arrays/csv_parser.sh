#!/usr/bin/env bash
set -euo pipefail

# CSV Parser
# Reads CSV files, supports querying, and pretty-prints as a table

# Global arrays to store CSV data
declare -a headers
declare -a rows

# Parse CSV file
parse_csv() {
    local file="$1"
    local line_num=0

    if [[ ! -f "$file" ]]; then
        echo "Error: File '$file' not found" >&2
        return 1
    fi

    while IFS= read -r line; do
        ((line_num++))

        # Parse CSV line handling quoted fields
        local -a fields
        parse_csv_line "$line" fields

        if [[ $line_num -eq 1 ]]; then
            # First line is headers
            headers=("${fields[@]}")
        else
            # Store row data as comma-separated string
            # (bash arrays of arrays are tricky, so we serialize)
            local row_data
            printf -v row_data '%s,' "${fields[@]}"
            row_data="${row_data%,}"  # Remove trailing comma
            rows+=("$row_data")
        fi
    done < "$file"

    echo "Parsed $((line_num - 1)) data rows with ${#headers[@]} columns"
}

# Parse a single CSV line into an array
# Handles quoted fields with commas inside
parse_csv_line() {
    local line="$1"
    local -n result_array="$2"

    result_array=()
    local field=""
    local in_quotes=false
    local i

    for ((i=0; i<${#line}; i++)); do
        local char="${line:i:1}"

        if [[ "$char" == '"' ]]; then
            if $in_quotes; then
                # Check for escaped quote ("")
                if [[ "${line:i+1:1}" == '"' ]]; then
                    field+="\""
                    ((i++))
                else
                    in_quotes=false
                fi
            else
                in_quotes=true
            fi
        elif [[ "$char" == ',' ]] && ! $in_quotes; then
            result_array+=("$field")
            field=""
        else
            field+="$char"
        fi
    done

    # Add last field
    result_array+=("$field")
}

# Get column index by header name
get_column_index() {
    local column_name="$1"
    local i

    for i in "${!headers[@]}"; do
        if [[ "${headers[i]}" == "$column_name" ]]; then
            echo "$i"
            return 0
        fi
    done

    echo "Error: Column '$column_name' not found" >&2
    return 1
}

# Query rows by column value
query_by_column() {
    local column_name="$1"
    local search_value="$2"

    local col_index
    col_index=$(get_column_index "$column_name") || return 1

    echo "Searching for '$search_value' in column '$column_name':"
    echo

    local found=0
    for row_data in "${rows[@]}"; do
        IFS=',' read -ra fields <<< "$row_data"

        if [[ "${fields[col_index]}" == *"$search_value"* ]]; then
            ((found++))
            for i in "${!headers[@]}"; do
                printf "  %-15s: %s\n" "${headers[i]}" "${fields[i]}"
            done
            echo
        fi
    done

    if [[ $found -eq 0 ]]; then
        echo "  No matching rows found"
    else
        echo "Found $found matching row(s)"
    fi
}

# Pretty-print entire table
print_table() {
    # Calculate column widths
    local -a col_widths
    local i

    # Initialize with header lengths
    for i in "${!headers[@]}"; do
        col_widths[i]=${#headers[i]}
    done

    # Check data rows for max widths
    for row_data in "${rows[@]}"; do
        IFS=',' read -ra fields <<< "$row_data"
        for i in "${!fields[@]}"; do
            local field_len=${#fields[i]}
            if [[ $field_len -gt ${col_widths[i]} ]]; then
                col_widths[i]=$field_len
            fi
        done
    done

    # Print header
    echo
    for i in "${!headers[@]}"; do
        printf "| %-${col_widths[i]}s " "${headers[i]}"
    done
    echo "|"

    # Print separator
    for i in "${!headers[@]}"; do
        printf "|"
        printf -- '-%.0s' $(seq 1 $((col_widths[i] + 2)))
    done
    echo "|"

    # Print data rows
    for row_data in "${rows[@]}"; do
        IFS=',' read -ra fields <<< "$row_data"
        for i in "${!fields[@]}"; do
            printf "| %-${col_widths[i]}s " "${fields[i]}"
        done
        echo "|"
    done
    echo
}

# Get column values as array
get_column() {
    local column_name="$1"
    local col_index
    col_index=$(get_column_index "$column_name") || return 1

    echo "Values in column '$column_name':"
    for row_data in "${rows[@]}"; do
        IFS=',' read -ra fields <<< "$row_data"
        echo "  ${fields[col_index]}"
    done
}

# Create sample CSV file
create_sample_csv() {
    local file="$1"

    cat > "$file" << 'EOF'
Name,Age,City,Email
Alice Smith,30,New York,alice@example.com
Bob Jones,25,Los Angeles,bob@example.com
Charlie Brown,35,Chicago,charlie@example.com
"Diana, Princess",28,"London, UK",diana@example.com
Eve Wilson,42,Boston,eve@example.com
Frank Lee,31,Seattle,frank@example.com
EOF

    echo "Created sample CSV: $file"
}

# Main demo
main() {
    echo "=== CSV Parser Demo ==="
    echo

    # Create sample CSV
    local csv_file="/tmp/sample_data.csv"
    create_sample_csv "$csv_file"

    echo
    echo "--- Parsing CSV ---"
    parse_csv "$csv_file"

    echo
    echo "--- Pretty Print Table ---"
    print_table

    echo "--- Query by Column ---"
    echo
    query_by_column "City" "Los Angeles"

    echo
    echo "--- Get Column Values ---"
    echo
    get_column "Name"

    echo
    echo "--- Query with Quoted Field ---"
    echo
    query_by_column "Name" "Diana"

    # Cleanup
    rm -f "$csv_file"

    echo
    echo "=== Demo Complete ==="
    echo
    echo "Usage:"
    echo "  source $0  # To load functions"
    echo "  parse_csv <file.csv>"
    echo "  print_table"
    echo "  query_by_column <column_name> <search_value>"
    echo "  get_column <column_name>"
}

# Run demo if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
