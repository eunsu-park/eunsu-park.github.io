#!/usr/bin/env bash
set -euo pipefail

# Associative Arrays Demonstrations
# Shows practical uses of associative arrays (hash maps/dictionaries)

# --- Word Frequency Counter ---
word_frequency() {
    echo "=== Word Frequency Counter ==="
    echo

    local input_file="${1:-}"
    declare -A word_count

    # Read from file or stdin
    local text
    if [[ -n "$input_file" && -f "$input_file" ]]; then
        text=$(cat "$input_file")
        echo "Reading from file: $input_file"
    else
        if [[ -t 0 ]]; then
            # Terminal input, use sample text
            text="the quick brown fox jumps over the lazy dog the fox is quick"
            echo "Using sample text:"
            echo "\"$text\""
        else
            # Pipe/redirect input
            text=$(cat)
            echo "Reading from stdin"
        fi
    fi

    echo

    # Convert to lowercase and split into words
    for word in $text; do
        # Remove punctuation and convert to lowercase
        word=$(echo "$word" | tr -d '.,!?;:' | tr '[:upper:]' '[:lower:]')
        [[ -z "$word" ]] && continue

        # Increment count
        ((word_count[$word]++)) || word_count[$word]=1
    done

    # Display results sorted by frequency
    echo "Word frequencies:"
    for word in "${!word_count[@]}"; do
        echo "${word_count[$word]} $word"
    done | sort -rn | awk '{printf "  %-15s %d\n", $2, $1}'

    echo
}

# --- Config File Loader ---
load_config() {
    echo "=== Config File Loader ==="
    echo

    declare -gA config

    # Create a sample config file
    local config_file="/tmp/sample_config.ini"
    cat > "$config_file" << 'EOF'
# Database Configuration
database.host=localhost
database.port=5432
database.name=myapp
database.user=admin

# Application Settings
app.name=MyApplication
app.version=1.2.3
app.debug=true
app.max_connections=100

# Cache Settings
cache.enabled=true
cache.ttl=3600
EOF

    echo "Loading config from: $config_file"
    echo

    # Parse config file
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue

        # Trim whitespace
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)

        config["$key"]="$value"
    done < "$config_file"

    # Display loaded config
    echo "Loaded configuration:"
    for key in "${!config[@]}"; do
        echo "  $key = ${config[$key]}"
    done | sort

    echo
    echo "Accessing specific values:"
    echo "  Database: ${config[database.name]} at ${config[database.host]}:${config[database.port]}"
    echo "  App: ${config[app.name]} v${config[app.version]}"
    echo "  Debug mode: ${config[app.debug]}"

    echo

    # Cleanup
    rm -f "$config_file"
}

# --- Simple Phonebook ---
phonebook() {
    echo "=== Simple Phonebook ==="
    echo

    declare -A phonebook

    # Add entries
    add_contact() {
        local name="$1"
        local number="$2"
        phonebook["$name"]="$number"
        echo "  ✓ Added: $name -> $number"
    }

    # Lookup entry
    lookup_contact() {
        local name="$1"
        if [[ -n "${phonebook[$name]+x}" ]]; then
            echo "  $name: ${phonebook[$name]}"
        else
            echo "  ✗ Contact '$name' not found"
        fi
    }

    # Delete entry
    delete_contact() {
        local name="$1"
        if [[ -n "${phonebook[$name]+x}" ]]; then
            unset 'phonebook[$name]'
            echo "  ✓ Deleted: $name"
        else
            echo "  ✗ Contact '$name' not found"
        fi
    }

    # List all entries
    list_contacts() {
        if [[ ${#phonebook[@]} -eq 0 ]]; then
            echo "  (phonebook is empty)"
            return
        fi

        echo "  Phonebook entries:"
        for name in "${!phonebook[@]}"; do
            printf "    %-20s %s\n" "$name" "${phonebook[$name]}"
        done | sort
    }

    # Demo operations
    echo "Adding contacts..."
    add_contact "Alice Smith" "555-0101"
    add_contact "Bob Jones" "555-0102"
    add_contact "Charlie Brown" "555-0103"
    add_contact "Diana Prince" "555-0104"

    echo
    list_contacts

    echo
    echo "Looking up contacts..."
    lookup_contact "Alice Smith"
    lookup_contact "Bob Jones"
    lookup_contact "Eve (Unknown)"

    echo
    echo "Deleting a contact..."
    delete_contact "Bob Jones"

    echo
    list_contacts

    echo
    echo "Total contacts: ${#phonebook[@]}"

    echo
}

# --- Multi-level Dictionary ---
nested_example() {
    echo "=== Nested Data Structure (Simulated) ==="
    echo

    declare -A user_data

    # Store structured data using key prefixes
    user_data["user1.name"]="Alice"
    user_data["user1.email"]="alice@example.com"
    user_data["user1.age"]="30"

    user_data["user2.name"]="Bob"
    user_data["user2.email"]="bob@example.com"
    user_data["user2.age"]="25"

    user_data["user3.name"]="Charlie"
    user_data["user3.email"]="charlie@example.com"
    user_data["user3.age"]="35"

    echo "Stored user data:"
    for key in "${!user_data[@]}"; do
        echo "  $key = ${user_data[$key]}"
    done | sort

    echo
    echo "Query users by prefix:"

    query_user() {
        local user_id="$1"
        echo "  $user_id:"
        for key in "${!user_data[@]}"; do
            if [[ "$key" == "$user_id."* ]]; then
                local field="${key#*.}"
                printf "    %-10s %s\n" "$field" "${user_data[$key]}"
            fi
        done
    }

    query_user "user1"
    echo
    query_user "user2"

    echo
}

# --- Main execution ---
main() {
    word_frequency
    echo
    echo "----------------------------------------"
    echo

    load_config
    echo "----------------------------------------"
    echo

    phonebook
    echo "----------------------------------------"
    echo

    nested_example

    echo "=== All Demos Complete ==="
}

main "$@"
