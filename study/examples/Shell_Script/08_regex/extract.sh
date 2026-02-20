#!/usr/bin/env bash
set -euo pipefail

# Data Extraction Using Regex
# Demonstrates extracting structured data from text using bash regex patterns

# ============================================================================
# Color definitions for output
# ============================================================================

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# ============================================================================
# Extraction Functions
# ============================================================================

# Extract all URLs from text
extract_urls() {
    local text="$1"

    echo -e "${CYAN}Extracting URLs...${NC}"

    # URL pattern: http(s)://...
    local url_pattern='https?://[a-zA-Z0-9./?=_%:-]*'

    local count=0
    while [[ "$text" =~ $url_pattern ]]; do
        local url="${BASH_REMATCH[0]}"
        echo -e "  ${GREEN}→${NC} $url"

        # Remove matched URL from text to find next one
        text="${text#*"$url"}"
        ((count++))
    done

    if [[ $count -eq 0 ]]; then
        echo -e "  ${YELLOW}(no URLs found)${NC}"
    else
        echo -e "  Found ${GREEN}$count${NC} URL(s)"
    fi
    echo
}

# Extract all email addresses from text
extract_emails() {
    local text="$1"

    echo -e "${CYAN}Extracting email addresses...${NC}"

    # Email pattern
    local email_pattern='[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

    local count=0
    while [[ "$text" =~ $email_pattern ]]; do
        local email="${BASH_REMATCH[0]}"
        echo -e "  ${GREEN}→${NC} $email"

        # Remove matched email to find next one
        text="${text#*"$email"}"
        ((count++))
    done

    if [[ $count -eq 0 ]]; then
        echo -e "  ${YELLOW}(no emails found)${NC}"
    else
        echo -e "  Found ${GREEN}$count${NC} email(s)"
    fi
    echo
}

# Parse structured log line into components
parse_log_line() {
    local log_line="$1"

    # Log pattern: [timestamp] level: message
    # Example: [2024-01-15 10:30:45] ERROR: Connection failed
    local log_pattern='^\[([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})\] ([A-Z]+): (.*)$'

    if [[ "$log_line" =~ $log_pattern ]]; then
        local timestamp="${BASH_REMATCH[1]}"
        local level="${BASH_REMATCH[2]}"
        local message="${BASH_REMATCH[3]}"

        echo -e "${GREEN}✓${NC} Successfully parsed log line:"
        echo -e "  Timestamp: ${CYAN}$timestamp${NC}"
        echo -e "  Level:     ${YELLOW}$level${NC}"
        echo -e "  Message:   $message"
        return 0
    else
        echo -e "${RED}✗${NC} Failed to parse log line: $log_line"
        return 1
    fi
}

# Parse CSV line handling quoted fields
parse_csv_line() {
    local csv_line="$1"

    echo -e "${CYAN}Parsing CSV line:${NC} $csv_line"

    # Simple CSV parser for demonstration
    # Handles: field1,"field with, comma","field with ""quotes"""

    local -a fields=()
    local field=""
    local in_quotes=false
    local i

    for ((i=0; i<${#csv_line}; i++)); do
        local char="${csv_line:$i:1}"

        if [[ "$char" == '"' ]]; then
            if [[ "$in_quotes" == true ]]; then
                # Check for escaped quote ""
                if [[ "${csv_line:$((i+1)):1}" == '"' ]]; then
                    field+="\""
                    ((i++))
                else
                    in_quotes=false
                fi
            else
                in_quotes=true
            fi
        elif [[ "$char" == ',' && "$in_quotes" == false ]]; then
            fields+=("$field")
            field=""
        else
            field+="$char"
        fi
    done

    # Add last field
    fields+=("$field")

    echo -e "  ${GREEN}Extracted ${#fields[@]} field(s):${NC}"
    local idx=1
    for field in "${fields[@]}"; do
        echo -e "    [$idx] $field"
        ((idx++))
    done
    echo
}

# Extract key-value pairs from configuration-style text
extract_key_values() {
    local text="$1"

    echo -e "${CYAN}Extracting key-value pairs...${NC}"

    # Pattern: key = value (or key=value)
    local kv_pattern='([a-zA-Z_][a-zA-Z0-9_]*)[[:space:]]*=[[:space:]]*([^[:space:]].*)'

    local count=0
    while IFS= read -r line; do
        if [[ "$line" =~ $kv_pattern ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"
            echo -e "  ${GREEN}$key${NC} = $value"
            ((count++))
        fi
    done <<< "$text"

    if [[ $count -eq 0 ]]; then
        echo -e "  ${YELLOW}(no key-value pairs found)${NC}"
    fi
    echo
}

# Extract phone numbers (US format)
extract_phone_numbers() {
    local text="$1"

    echo -e "${CYAN}Extracting phone numbers...${NC}"

    # US phone pattern: (123) 456-7890 or 123-456-7890 or 1234567890
    local phone_pattern='(\([0-9]{3}\) ?[0-9]{3}-[0-9]{4}|[0-9]{3}-[0-9]{3}-[0-9]{4}|[0-9]{10})'

    local count=0
    while [[ "$text" =~ $phone_pattern ]]; do
        local phone="${BASH_REMATCH[1]}"
        echo -e "  ${GREEN}→${NC} $phone"

        text="${text#*"$phone"}"
        ((count++))
    done

    if [[ $count -eq 0 ]]; then
        echo -e "  ${YELLOW}(no phone numbers found)${NC}"
    else
        echo -e "  Found ${GREEN}$count${NC} phone number(s)"
    fi
    echo
}

# ============================================================================
# Demo Section
# ============================================================================

demo_url_extraction() {
    echo -e "\n${BLUE}=== URL Extraction Demo ===${NC}\n"

    local sample_text="Check out https://example.com and http://api.example.org/v1/data
Also visit https://github.com/user/repo for more info.
FTP not supported: ftp://old.server.com"

    echo "Sample text:"
    echo "$sample_text"
    echo

    extract_urls "$sample_text"
}

demo_email_extraction() {
    echo -e "${BLUE}=== Email Extraction Demo ===${NC}\n"

    local sample_text="Contact john.doe@example.com or support@company.co.uk
For sales, reach out to sales@example.com
Admin email: admin@localhost"

    echo "Sample text:"
    echo "$sample_text"
    echo

    extract_emails "$sample_text"
}

demo_log_parsing() {
    echo -e "${BLUE}=== Log Line Parsing Demo ===${NC}\n"

    local -a log_lines=(
        "[2024-01-15 10:30:45] ERROR: Connection timeout"
        "[2024-01-15 10:31:12] INFO: Retrying connection"
        "[2024-01-15 10:31:15] WARN: High memory usage detected"
        "Invalid log format without timestamp"
    )

    for log in "${log_lines[@]}"; do
        parse_log_line "$log"
        echo
    done
}

demo_csv_parsing() {
    echo -e "${BLUE}=== CSV Parsing Demo ===${NC}\n"

    local -a csv_lines=(
        'John,Doe,30,Engineer'
        '"Smith, Jane",Manager,"San Francisco, CA",45'
        'Bob,"Quote ""test"" here",Developer,35'
    )

    for csv in "${csv_lines[@]}"; do
        parse_csv_line "$csv"
    done
}

demo_key_value_extraction() {
    echo -e "${BLUE}=== Key-Value Extraction Demo ===${NC}\n"

    local config_text="# Configuration file
database_host = localhost
database_port = 5432
max_connections=100
timeout = 30
debug_mode = true"

    echo "Sample configuration:"
    echo "$config_text"
    echo

    extract_key_values "$config_text"
}

demo_phone_extraction() {
    echo -e "${BLUE}=== Phone Number Extraction Demo ===${NC}\n"

    local sample_text="Call us at (555) 123-4567 or 555-987-6543
Mobile: 5551234567
International numbers not supported: +1-555-123-4567"

    echo "Sample text:"
    echo "$sample_text"
    echo

    extract_phone_numbers "$sample_text"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║    Data Extraction with Regex Demo        ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"

    demo_url_extraction
    demo_email_extraction
    demo_log_parsing
    demo_csv_parsing
    demo_key_value_extraction
    demo_phone_extraction

    echo -e "${GREEN}=== All Extraction Demos Complete ===${NC}\n"
}

main "$@"
