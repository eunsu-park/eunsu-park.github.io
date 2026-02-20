#!/usr/bin/env bash
set -euo pipefail

# String Operations with Parameter Expansion
# Demonstrates various string manipulation techniques using bash parameter expansion

echo "=== Parameter Expansion String Operations ==="
echo

# 1. Extract filename, extension, directory from a path
echo "1. Path Component Extraction"
echo "----------------------------"
filepath="/home/user/documents/report.tar.gz"
echo "Full path: $filepath"
echo

# Remove directory path (shortest match from left)
filename="${filepath##*/}"
echo "Filename: $filename"

# Extract directory (longest match from right)
directory="${filepath%/*}"
echo "Directory: $directory"

# Extract basename (remove extension)
basename="${filename%%.*}"
echo "Basename: $basename"

# Extract extension (remove everything before last dot)
extension="${filename##*.}"
echo "Extension: $extension"

# Extract full extension (.tar.gz)
full_extension="${filename#*.}"
echo "Full extension: $full_extension"

echo
echo

# 2. Batch rename files using parameter expansion
echo "2. Batch Rename Simulation"
echo "--------------------------"
files=("photo_2023.jpg" "photo_2024.jpg" "photo_2025.png")

echo "Original files:"
printf "  %s\n" "${files[@]}"
echo

echo "Renamed files (photo -> image):"
for file in "${files[@]}"; do
    new_name="${file/photo/image}"
    echo "  $file -> $new_name"
done

echo
echo "Change all .jpg to .jpeg:"
for file in "${files[@]}"; do
    new_name="${file%.jpg}.jpeg"
    # Only show if it actually changed
    if [[ "$new_name" != "$file" ]]; then
        echo "  $file -> $new_name"
    fi
done

echo
echo

# 3. Config value parsing (key=value format)
echo "3. Config Value Parsing"
echo "-----------------------"
config_lines=(
    "database_host=localhost"
    "database_port=5432"
    "database_name=myapp"
    "# This is a comment"
    "max_connections=100"
)

declare -A config
for line in "${config_lines[@]}"; do
    # Skip comments and empty lines
    [[ "$line" =~ ^#.*$ ]] && continue
    [[ -z "$line" ]] && continue

    # Extract key and value
    key="${line%%=*}"
    value="${line#*=}"
    config["$key"]="$value"
    echo "  $key = $value"
done

echo
echo "Access config values:"
echo "  Database: ${config[database_name]} at ${config[database_host]}:${config[database_port]}"

echo
echo

# 4. URL parsing (protocol, host, path)
echo "4. URL Parsing"
echo "--------------"
url="https://api.example.com:8443/v1/users?page=1"
echo "URL: $url"
echo

# Extract protocol
protocol="${url%%://*}"
echo "Protocol: $protocol"

# Remove protocol to get the rest
rest="${url#*://}"

# Extract host:port and path
host_port="${rest%%/*}"
path="${rest#*/}"

# Separate host and port
if [[ "$host_port" == *:* ]]; then
    host="${host_port%:*}"
    port="${host_port#*:}"
else
    host="$host_port"
    port=""
fi

echo "Host: $host"
echo "Port: ${port:-default}"
echo "Path: /$path"

# Extract query parameters
if [[ "$path" == *\?* ]]; then
    query="${path#*\?}"
    path="${path%%\?*}"
    echo "Path (no query): /$path"
    echo "Query string: $query"
fi

echo
echo

# 5. Default values and substitution
echo "5. Default Values and Substitution"
echo "-----------------------------------"
unset optional_var
required_var="I exist"

echo "Use default if unset: ${optional_var:-default_value}"
echo "Use default if unset or empty: ${optional_var-default_value}"
echo "Variable doesn't change: optional_var=${optional_var:-}"

# Assign default if unset
echo "Assign default: ${optional_var:=assigned_default}"
echo "Now optional_var=$optional_var"

echo
echo "Use alternate value if set:"
echo "  required_var is set: ${required_var:+yes, it is set}"
echo "  optional_var is set: ${optional_var:+yes, it is set}"

echo
echo

# 6. String length and substring
echo "6. String Length and Substring"
echo "------------------------------"
text="Hello, World!"
echo "Text: $text"
echo "Length: ${#text}"
echo "Substring (0-5): ${text:0:5}"
echo "Substring (7-5): ${text:7:5}"
echo "Last 6 chars: ${text: -6}"

echo
echo

# 7. Case conversion (bash 4.0+)
echo "7. Case Conversion"
echo "------------------"
mixed="Hello World"
echo "Original: $mixed"
echo "Lowercase: ${mixed,,}"
echo "Uppercase: ${mixed^^}"
echo "First char uppercase: ${mixed^}"
echo "First char of each word uppercase: ${mixed^^[hw]}"

echo
echo "=== Demo Complete ==="
