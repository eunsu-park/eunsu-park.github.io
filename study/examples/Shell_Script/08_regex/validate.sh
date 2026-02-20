#!/usr/bin/env bash
set -euo pipefail

# Input Validation Using Bash Regex
# Demonstrates various validation patterns using bash's =~ operator and BASH_REMATCH

# ============================================================================
# Color definitions for output
# ============================================================================

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# ============================================================================
# Validation Functions
# ============================================================================

# Validate email address format
validate_email() {
    local email="$1"

    # Email regex: local-part@domain.tld
    # Simplified pattern for demonstration
    local email_regex='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if [[ "$email" =~ $email_regex ]]; then
        # BASH_REMATCH[0] contains the entire match
        echo -e "${GREEN}✓${NC} Valid email: $email"
        return 0
    else
        echo -e "${RED}✗${NC} Invalid email: $email"
        return 1
    fi
}

# Validate IPv4 address
validate_ipv4() {
    local ip="$1"

    # IPv4 regex with capturing groups for each octet
    local ip_regex='^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$'

    if [[ ! "$ip" =~ $ip_regex ]]; then
        echo -e "${RED}✗${NC} Invalid IPv4 format: $ip"
        return 1
    fi

    # Check each octet is in valid range (0-255)
    # BASH_REMATCH[1-4] contain the four octets
    local octet
    for i in 1 2 3 4; do
        octet="${BASH_REMATCH[$i]}"
        if [[ $octet -gt 255 ]]; then
            echo -e "${RED}✗${NC} Invalid IPv4 (octet $octet > 255): $ip"
            return 1
        fi
    done

    echo -e "${GREEN}✓${NC} Valid IPv4: $ip (octets: ${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.${BASH_REMATCH[3]}.${BASH_REMATCH[4]})"
    return 0
}

# Validate date in YYYY-MM-DD format with range checks
validate_date() {
    local date_str="$1"

    # Date regex with capturing groups for year, month, day
    local date_regex='^([0-9]{4})-([0-9]{2})-([0-9]{2})$'

    if [[ ! "$date_str" =~ $date_regex ]]; then
        echo -e "${RED}✗${NC} Invalid date format: $date_str (expected YYYY-MM-DD)"
        return 1
    fi

    local year="${BASH_REMATCH[1]}"
    local month="${BASH_REMATCH[2]}"
    local day="${BASH_REMATCH[3]}"

    # Basic range validation
    if [[ $month -lt 1 || $month -gt 12 ]]; then
        echo -e "${RED}✗${NC} Invalid month: $month (must be 01-12)"
        return 1
    fi

    if [[ $day -lt 1 || $day -gt 31 ]]; then
        echo -e "${RED}✗${NC} Invalid day: $day (must be 01-31)"
        return 1
    fi

    if [[ $year -lt 1900 || $year -gt 2100 ]]; then
        echo -e "${YELLOW}⚠${NC} Date year out of typical range: $year"
    fi

    echo -e "${GREEN}✓${NC} Valid date: $date_str (Y=$year, M=$month, D=$day)"
    return 0
}

# Validate semantic version (e.g., 1.2.3, 2.0.0-beta, 1.0.0+build123)
validate_semver() {
    local version="$1"

    # Semantic versioning regex: major.minor.patch[-prerelease][+buildmetadata]
    local semver_regex='^([0-9]+)\.([0-9]+)\.([0-9]+)(-([a-zA-Z0-9.-]+))?(\+([a-zA-Z0-9.-]+))?$'

    if [[ ! "$version" =~ $semver_regex ]]; then
        echo -e "${RED}✗${NC} Invalid semver: $version"
        return 1
    fi

    local major="${BASH_REMATCH[1]}"
    local minor="${BASH_REMATCH[2]}"
    local patch="${BASH_REMATCH[3]}"
    local prerelease="${BASH_REMATCH[5]}"
    local build="${BASH_REMATCH[7]}"

    echo -e "${GREEN}✓${NC} Valid semver: $version"
    echo "    Major: $major, Minor: $minor, Patch: $patch"
    [[ -n "$prerelease" ]] && echo "    Prerelease: $prerelease"
    [[ -n "$build" ]] && echo "    Build: $build"

    return 0
}

# Validate URL (http/https)
validate_url() {
    local url="$1"

    # URL regex with protocol, host, optional port, optional path
    local url_regex='^(https?)://([a-zA-Z0-9.-]+)(:[0-9]+)?(/[^[:space:]]*)?$'

    if [[ ! "$url" =~ $url_regex ]]; then
        echo -e "${RED}✗${NC} Invalid URL: $url"
        return 1
    fi

    local protocol="${BASH_REMATCH[1]}"
    local host="${BASH_REMATCH[2]}"
    local port="${BASH_REMATCH[3]}"
    local path="${BASH_REMATCH[4]}"

    echo -e "${GREEN}✓${NC} Valid URL: $url"
    echo "    Protocol: $protocol"
    echo "    Host: $host"
    [[ -n "$port" ]] && echo "    Port: ${port:1}"  # Remove leading colon
    [[ -n "$path" ]] && echo "    Path: $path"

    return 0
}

# ============================================================================
# Demo Section
# ============================================================================

demo_email_validation() {
    echo -e "\n${BLUE}=== Email Validation ===${NC}\n"

    local -a test_emails=(
        "user@example.com"
        "john.doe+tag@company.co.uk"
        "invalid@"
        "@example.com"
        "no-at-sign.com"
        "test@domain"
        "valid.email@subdomain.example.com"
    )

    for email in "${test_emails[@]}"; do
        validate_email "$email"
    done
}

demo_ipv4_validation() {
    echo -e "\n${BLUE}=== IPv4 Validation ===${NC}\n"

    local -a test_ips=(
        "192.168.1.1"
        "10.0.0.1"
        "255.255.255.255"
        "256.1.1.1"
        "192.168.1"
        "192.168.1.1.1"
        "0.0.0.0"
    )

    for ip in "${test_ips[@]}"; do
        validate_ipv4 "$ip"
    done
}

demo_date_validation() {
    echo -e "\n${BLUE}=== Date Validation ===${NC}\n"

    local -a test_dates=(
        "2024-01-15"
        "2024-12-31"
        "2024-13-01"
        "2024-01-32"
        "24-01-15"
        "2024/01/15"
        "1850-06-15"
    )

    for date in "${test_dates[@]}"; do
        validate_date "$date"
    done
}

demo_semver_validation() {
    echo -e "\n${BLUE}=== Semantic Version Validation ===${NC}\n"

    local -a test_versions=(
        "1.0.0"
        "2.5.13"
        "1.0.0-alpha"
        "1.0.0-beta.1"
        "1.0.0+20240115"
        "1.0.0-rc.1+build.456"
        "v1.0.0"
        "1.0"
    )

    for version in "${test_versions[@]}"; do
        validate_semver "$version"
    done
}

demo_url_validation() {
    echo -e "\n${BLUE}=== URL Validation ===${NC}\n"

    local -a test_urls=(
        "https://example.com"
        "http://localhost:8080"
        "https://api.example.com/v1/users"
        "https://sub.domain.example.com:443/path?query=1"
        "ftp://example.com"
        "example.com"
        "http://"
    )

    for url in "${test_urls[@]}"; do
        validate_url "$url"
    done
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Input Validation with Bash Regex Demo    ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"

    demo_email_validation
    demo_ipv4_validation
    demo_date_validation
    demo_semver_validation
    demo_url_validation

    echo -e "\n${GREEN}=== All Validation Demos Complete ===${NC}\n"
}

main "$@"
