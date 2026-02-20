#!/usr/bin/env bash
#
# Performance Bottleneck Finder
#
# This script diagnoses system performance bottlenecks by analyzing:
# - CPU usage and load
# - Memory usage and swap
# - Disk I/O
# - Network I/O
# - Process resource usage
#
# Features:
# - Color-coded output with warning/critical thresholds
# - Summary report generation
# - Export to file
# - Multiple analysis modes

set -euo pipefail

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

readonly SCRIPT_NAME="$(basename "$0")"

# Thresholds
readonly CPU_WARNING=70
readonly CPU_CRITICAL=90
readonly MEM_WARNING=75
readonly MEM_CRITICAL=90
readonly DISK_IO_WARNING=80
readonly DISK_IO_CRITICAL=95
readonly LOAD_WARNING=2.0
readonly LOAD_CRITICAL=4.0

# Colors
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly GREEN='\033[0;32m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Output settings
OUTPUT_FILE=""
VERBOSE=false

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS]

Diagnose system performance bottlenecks.

OPTIONS:
    -h, --help              Show this help message
    -o, --output FILE       Export report to file
    -v, --verbose           Verbose output
    -a, --analyze TYPE      Analyze specific type: cpu|memory|disk|network|process|all

EXAMPLES:
    $SCRIPT_NAME                        # Full analysis
    $SCRIPT_NAME -a cpu                 # CPU analysis only
    $SCRIPT_NAME -o report.txt          # Export to file

EOF
}

print_header() {
    local title="$1"
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}${title}${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_status() {
    local value="$1"
    local threshold_warn="$2"
    local threshold_crit="$3"
    local unit="${4:-%}"

    if (( $(echo "$value >= $threshold_crit" | bc -l) )); then
        echo -e "${RED}${value}${unit} [CRITICAL]${NC}"
    elif (( $(echo "$value >= $threshold_warn" | bc -l) )); then
        echo -e "${YELLOW}${value}${unit} [WARNING]${NC}"
    else
        echo -e "${GREEN}${value}${unit} [OK]${NC}"
    fi
}

analyze_cpu() {
    print_header "CPU Analysis"

    echo "CPU Information:"
    if command -v lscpu &>/dev/null; then
        lscpu | grep -E "^(Model name|CPU\(s\)|Thread|Core|Socket)"
    fi

    echo ""
    echo "Current CPU Usage:"

    # Get CPU usage (requires top or mpstat)
    if command -v mpstat &>/dev/null; then
        local cpu_usage
        cpu_usage=$(mpstat 1 1 | awk '/Average:/ {print 100 - $NF}')
        echo -n "  Overall CPU: "
        print_status "$cpu_usage" "$CPU_WARNING" "$CPU_CRITICAL"

        # Per-CPU stats
        if [[ "$VERBOSE" == "true" ]]; then
            echo ""
            echo "  Per-CPU usage:"
            mpstat -P ALL 1 1 | tail -n +4
        fi
    else
        # Fallback to top
        local cpu_usage
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
        echo -n "  CPU Usage: "
        print_status "$cpu_usage" "$CPU_WARNING" "$CPU_CRITICAL"
    fi

    echo ""
    echo "Load Average:"
    local load_1min load_5min load_15min
    read -r load_1min load_5min load_15min _ _ < /proc/loadavg

    echo -n "  1-min:  "
    print_status "$load_1min" "$LOAD_WARNING" "$LOAD_CRITICAL" ""

    echo -n "  5-min:  "
    print_status "$load_5min" "$LOAD_WARNING" "$LOAD_CRITICAL" ""

    echo -n "  15-min: "
    print_status "$load_15min" "$LOAD_WARNING" "$LOAD_CRITICAL" ""

    echo ""
    echo "Top CPU Consumers:"
    ps aux --sort=-%cpu | head -6 | awk 'NR==1 {print "  " $0} NR>1 {printf "  %-10s %5s %5s %s\n", $1, $3, $4, $11}'
}

analyze_memory() {
    print_header "Memory Analysis"

    echo "Memory Usage:"

    if [[ -f /proc/meminfo ]]; then
        local mem_total mem_available mem_free mem_used mem_percent

        mem_total=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
        mem_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        mem_free=$(awk '/^MemFree:/ {print $2}' /proc/meminfo)

        mem_used=$((mem_total - mem_available))
        mem_percent=$(awk "BEGIN {printf \"%.1f\", ($mem_used / $mem_total) * 100}")

        echo "  Total:     $((mem_total / 1024)) MB"
        echo "  Used:      $((mem_used / 1024)) MB"
        echo "  Available: $((mem_available / 1024)) MB"
        echo -n "  Usage:     "
        print_status "$mem_percent" "$MEM_WARNING" "$MEM_CRITICAL"
    fi

    echo ""
    echo "Swap Usage:"

    if [[ -f /proc/meminfo ]]; then
        local swap_total swap_free swap_used swap_percent

        swap_total=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
        swap_free=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)

        if [[ $swap_total -gt 0 ]]; then
            swap_used=$((swap_total - swap_free))
            swap_percent=$(awk "BEGIN {printf \"%.1f\", ($swap_used / $swap_total) * 100}")

            echo "  Total:     $((swap_total / 1024)) MB"
            echo "  Used:      $((swap_used / 1024)) MB"
            echo -n "  Usage:     "
            print_status "$swap_percent" "$MEM_WARNING" "$MEM_CRITICAL"
        else
            echo "  No swap configured"
        fi
    fi

    echo ""
    echo "Top Memory Consumers:"
    ps aux --sort=-%mem | head -6 | awk 'NR==1 {print "  " $0} NR>1 {printf "  %-10s %5s %5s %s\n", $1, $3, $4, $11}'
}

analyze_disk() {
    print_header "Disk I/O Analysis"

    echo "Disk Usage:"
    df -h | awk 'NR==1 {print "  " $0} NR>1 && $1 ~ /^\/dev\// {print "  " $0}'

    echo ""
    echo "Disk I/O Statistics:"

    if command -v iostat &>/dev/null; then
        iostat -x 1 2 | tail -n +4 | grep -v "^$"
    else
        echo "  iostat not available"

        if [[ -f /proc/diskstats ]]; then
            echo "  Raw disk stats from /proc/diskstats:"
            awk '{if ($4 > 0) print "  " $3, "reads:", $4, "writes:", $8}' /proc/diskstats
        fi
    fi

    echo ""
    echo "Disk Space Warnings:"
    df -h | awk 'NR>1 && $1 ~ /^\/dev\// {
        usage = substr($5, 1, length($5)-1);
        if (usage >= 90) {
            print "  [CRITICAL] " $6 " is " usage "% full"
        } else if (usage >= 75) {
            print "  [WARNING] " $6 " is " usage "% full"
        }
    }'
}

analyze_network() {
    print_header "Network Analysis"

    echo "Network Interfaces:"
    if command -v ip &>/dev/null; then
        ip -s link | grep -E "^[0-9]+:|RX:|TX:" | awk '{
            if ($0 ~ /^[0-9]+:/) {
                printf "  %-20s", $2
            } else if ($0 ~ /RX:/) {
                getline; printf "RX: %s bytes  ", $1
            } else if ($0 ~ /TX:/) {
                getline; printf "TX: %s bytes\n", $1
            }
        }'
    else
        ifconfig | grep -E "^[a-z]|RX packets|TX packets"
    fi

    echo ""
    echo "Network Connections:"
    if command -v ss &>/dev/null; then
        echo "  Active connections by state:"
        ss -s | grep -v "^Total:"
    else
        echo "  Active connections:"
        netstat -an | awk '/^tcp/ {print $6}' | sort | uniq -c | sort -rn | awk '{print "  " $2 ": " $1}'
    fi

    echo ""
    echo "Top Network Processes:"
    if command -v ss &>/dev/null; then
        ss -tunap | grep -v "State" | awk '{print $NF}' | grep -o '".*"' | tr -d '"' | sort | uniq -c | sort -rn | head -5 | awk '{print "  " $2 " (" $1 " connections)"}'
    else
        echo "  Detailed process info not available (requires ss)"
    fi
}

analyze_processes() {
    print_header "Process Analysis"

    echo "Process Count:"
    local total_procs running_procs sleeping_procs zombie_procs

    total_procs=$(ps aux | wc -l)
    running_procs=$(ps aux | awk '$8 ~ /R/ {count++} END {print count+0}')
    sleeping_procs=$(ps aux | awk '$8 ~ /S/ {count++} END {print count+0}')
    zombie_procs=$(ps aux | awk '$8 ~ /Z/ {count++} END {print count+0}')

    echo "  Total:    $total_procs"
    echo "  Running:  $running_procs"
    echo "  Sleeping: $sleeping_procs"

    if [[ $zombie_procs -gt 0 ]]; then
        echo -e "  ${RED}Zombies:  $zombie_procs [WARNING]${NC}"
    else
        echo "  Zombies:  $zombie_procs"
    fi

    echo ""
    echo "Top Processes by CPU:"
    ps aux --sort=-%cpu | head -6 | awk '{printf "  %-10s %5s%% CPU  %5s%% MEM  %s\n", $1, $3, $4, $11}'

    echo ""
    echo "Top Processes by Memory:"
    ps aux --sort=-%mem | head -6 | awk '{printf "  %-10s %5s%% CPU  %5s%% MEM  %s\n", $1, $3, $4, $11}'
}

generate_summary() {
    print_header "Summary & Recommendations"

    echo "Potential Bottlenecks:"

    local issues_found=false

    # CPU check
    if command -v mpstat &>/dev/null; then
        local cpu_usage
        cpu_usage=$(mpstat 1 1 | awk '/Average:/ {print 100 - $NF}')
        if (( $(echo "$cpu_usage >= $CPU_CRITICAL" | bc -l) )); then
            echo -e "  ${RED}[CRITICAL]${NC} CPU usage is very high ($cpu_usage%)"
            echo "    - Check top CPU consumers"
            echo "    - Consider scaling up CPU resources"
            issues_found=true
        fi
    fi

    # Memory check
    local mem_percent
    mem_percent=$(free | awk '/^Mem:/ {printf "%.1f", ($3 / $2) * 100}')
    if (( $(echo "$mem_percent >= $MEM_CRITICAL" | bc -l) )); then
        echo -e "  ${RED}[CRITICAL]${NC} Memory usage is very high ($mem_percent%)"
        echo "    - Check for memory leaks"
        echo "    - Review top memory consumers"
        issues_found=true
    fi

    # Swap check
    local swap_percent
    swap_percent=$(free | awk '/^Swap:/ {if ($2 > 0) printf "%.1f", ($3 / $2) * 100; else print "0"}')
    if (( $(echo "$swap_percent >= $MEM_WARNING" | bc -l) )); then
        echo -e "  ${YELLOW}[WARNING]${NC} High swap usage ($swap_percent%)"
        echo "    - System may be thrashing"
        echo "    - Consider adding more RAM"
        issues_found=true
    fi

    # Zombie processes
    local zombie_count
    zombie_count=$(ps aux | awk '$8 ~ /Z/ {count++} END {print count+0}')
    if [[ $zombie_count -gt 0 ]]; then
        echo -e "  ${YELLOW}[WARNING]${NC} Zombie processes detected ($zombie_count)"
        echo "    - Check parent processes"
        issues_found=true
    fi

    if [[ "$issues_found" == "false" ]]; then
        echo -e "  ${GREEN}No critical issues detected${NC}"
    fi
}

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

main() {
    local analyze_type="all"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            -o|--output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -a|--analyze)
                analyze_type="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Redirect output if file specified
    if [[ -n "$OUTPUT_FILE" ]]; then
        exec > >(tee "$OUTPUT_FILE")
    fi

    echo "System Performance Bottleneck Analysis"
    echo "Generated: $(date)"
    echo "Hostname: $(hostname)"

    case "$analyze_type" in
        all)
            analyze_cpu
            analyze_memory
            analyze_disk
            analyze_network
            analyze_processes
            generate_summary
            ;;
        cpu)
            analyze_cpu
            ;;
        memory)
            analyze_memory
            ;;
        disk)
            analyze_disk
            ;;
        network)
            analyze_network
            ;;
        process)
            analyze_processes
            ;;
        *)
            echo "Unknown analysis type: $analyze_type"
            usage
            exit 1
            ;;
    esac

    if [[ -n "$OUTPUT_FILE" ]]; then
        echo ""
        echo "Report saved to: $OUTPUT_FILE"
    fi
}

main "$@"
