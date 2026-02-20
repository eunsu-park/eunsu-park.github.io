#!/usr/bin/env bash
# =============================================================================
# performance_diagnostics.sh - System Performance Bottleneck Diagnosis
#
# PURPOSE: Identifies CPU, memory, disk I/O, and network bottlenecks using
#          standard Linux (and macOS-compatible) tools. Teaches which metrics
#          matter and how to interpret them.
#
# USAGE:
#   ./performance_diagnostics.sh [--cpu|--memory|--disk|--network|--all]
#
# MODES:
#   --cpu       Analyze CPU load and top consumers
#   --memory    Analyze RAM and swap usage
#   --disk      Analyze disk I/O throughput and utilization
#   --network   Analyze connections and listening ports
#   --all       Run all analyses and print a bottleneck summary (default)
#
# PREREQUISITES:
#   Linux: ps, top, df, iostat (sysstat), ss or netstat, free
#   macOS: ps, top, df, vm_stat, netstat (iostat available via brew install sysstat)
#
# CROSS-PLATFORM NOTES:
#   This script detects the OS and adjusts commands accordingly.
#   On macOS, some Linux-specific flags are not available; fallbacks are used.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Color and formatting
# ---------------------------------------------------------------------------
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Bottleneck severity levels
SEV_OK="OK"
SEV_WARN="WARN"
SEV_CRIT="CRIT"

# Accumulated bottleneck findings for the summary report
BOTTLENECKS=()

# ---------------------------------------------------------------------------
# OS detection — commands differ between Linux and macOS (Darwin)
# ---------------------------------------------------------------------------
OS="$(uname -s)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
section() {
    echo ""
    echo -e "${BOLD}${CYAN}>>> $1${RESET}"
    echo -e "${CYAN}$(printf '%0.s-' {1..60})${RESET}"
}

flag_bottleneck() {
    local severity="$1"
    local message="$2"
    BOTTLENECKS+=("[$severity] $message")
}

sev_color() {
    case "$1" in
        "$SEV_CRIT") echo -e "${RED}[CRIT]${RESET}" ;;
        "$SEV_WARN") echo -e "${YELLOW}[WARN]${RESET}" ;;
        *)           echo -e "${GREEN}[ OK ]${RESET}" ;;
    esac
}

# Check if a command exists; return 1 silently if missing
has_cmd() { command -v "$1" &>/dev/null; }

# ---------------------------------------------------------------------------
# SYSTEM OVERVIEW
# Metric: uptime, kernel, CPU core count
# Why it matters: baseline context before diving into specific metrics
# ---------------------------------------------------------------------------
system_overview() {
    section "System Overview"

    echo -e "  Hostname   : $(hostname)"
    echo -e "  OS         : $OS $(uname -r)"
    echo -e "  Date       : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo -e "  Uptime     : $(uptime | sed 's/.*up //' | sed 's/,  [0-9]* user.*//')"

    # CPU core count (logical processors)
    local cores
    if [[ "$OS" == "Darwin" ]]; then
        cores=$(sysctl -n hw.logicalcpu)
    else
        cores=$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo)
    fi
    echo -e "  CPU Cores  : $cores logical processor(s)"
}

# ---------------------------------------------------------------------------
# CPU ANALYSIS
# Metrics: load average (1/5/15 min), per-process CPU%
#
# Rule of thumb:
#   load avg / cores < 1.0 → healthy
#   load avg / cores 1.0–2.0 → elevated, monitor
#   load avg / cores > 2.0 → saturated, investigate
# ---------------------------------------------------------------------------
analyze_cpu() {
    section "CPU Analysis"

    # Load average is the number of processes waiting for CPU time
    local load1 load5 load15 cores
    if [[ "$OS" == "Darwin" ]]; then
        read -r load1 load5 load15 <<< "$(sysctl -n vm.loadavg | tr -d '{}' | awk '{print $1,$2,$3}')"
        cores=$(sysctl -n hw.logicalcpu)
    else
        read -r load1 load5 load15 <<< "$(awk '{print $1,$2,$3}' /proc/loadavg)"
        cores=$(nproc)
    fi

    echo -e "  Load Average (1/5/15 min): ${BOLD}$load1  $load5  $load15${RESET}"
    echo -e "  Logical CPU Cores        : $cores"

    # Compare 1-min load to core count to detect saturation
    local ratio
    ratio=$(awk "BEGIN {printf \"%.2f\", $load1 / $cores}")
    echo -e "  Load/Core Ratio (1 min)  : $ratio"

    local sev="$SEV_OK"
    if awk "BEGIN {exit !($load1 / $cores >= 2.0)}"; then
        sev="$SEV_CRIT"; flag_bottleneck "$sev" "CPU saturated: load/core ratio $ratio"
    elif awk "BEGIN {exit !($load1 / $cores >= 1.0)}"; then
        sev="$SEV_WARN"; flag_bottleneck "$sev" "CPU elevated: load/core ratio $ratio"
    fi
    echo -e "  CPU Pressure             : $(sev_color "$sev")"

    echo ""
    echo -e "  ${BOLD}Top 5 CPU-consuming processes:${RESET}"
    # ps output: %cpu pid comm — sorted descending by CPU
    # -A selects all processes; -o customizes output columns
    ps -A -o pcpu,pid,comm 2>/dev/null \
        | sort -rn \
        | head -5 \
        | awk '{printf "    %6s%%  PID %-6s  %s\n", $1, $2, $3}'

    # Detect zombie processes — processes that have exited but whose parent
    # has not yet called wait(). Large numbers indicate a parent bug.
    local zombies
    zombies=$(ps -A -o stat 2>/dev/null | grep -c '^Z' || echo 0)
    echo ""
    if [[ $zombies -gt 0 ]]; then
        echo -e "  ${YELLOW}Zombie processes: $zombies (parent process may not be reaping children)${RESET}"
        flag_bottleneck "$SEV_WARN" "Zombie processes detected: $zombies"
    else
        echo -e "  Zombie processes: ${GREEN}0${RESET}"
    fi
}

# ---------------------------------------------------------------------------
# MEMORY ANALYSIS
# Metrics: total/used/free/cached RAM, swap in/out
#
# Key insight: Linux aggressively uses free RAM for disk cache (page cache).
# "Available" memory (not just "free") is the correct metric for headroom.
# High swap usage with low available RAM = memory pressure bottleneck.
# ---------------------------------------------------------------------------
analyze_memory() {
    section "Memory Analysis"

    if [[ "$OS" == "Darwin" ]]; then
        # macOS uses vm_stat; parse page counts (each page = 4096 bytes)
        local page_size=4096
        local vm
        vm=$(vm_stat)
        local free_pages wired active inactive
        free_pages=$(echo "$vm" | awk '/Pages free/{gsub(/\./,"",$3); print $3}')
        wired=$(echo "$vm"      | awk '/Pages wired/{gsub(/\./,"",$4); print $4}')
        active=$(echo "$vm"     | awk '/Pages active/{gsub(/\./,"",$3); print $3}')
        inactive=$(echo "$vm"   | awk '/Pages inactive/{gsub(/\./,"",$3); print $3}')

        local total_pages=$(( (${free_pages:-0} + ${wired:-0} + ${active:-0} + ${inactive:-0}) ))
        local total_mb=$(( total_pages * page_size / 1024 / 1024 ))
        local free_mb=$(( ${free_pages:-0} * page_size / 1024 / 1024 ))
        local used_mb=$(( total_mb - free_mb ))

        echo -e "  Total RAM : ${total_mb} MB"
        echo -e "  Used      : ${used_mb} MB"
        echo -e "  Free      : ${free_mb} MB"
        echo -e "  (macOS caches aggressively; use 'Memory Pressure' in Activity Monitor)"

        # Swap on macOS via sysctl
        local swap_used
        swap_used=$(sysctl -n vm.swapusage 2>/dev/null | awk '{print $6}' | tr -d 'M' || echo 0)
        echo -e "  Swap Used : ${swap_used} MB"
        if awk "BEGIN {exit !(${swap_used:-0} > 1024)}"; then
            flag_bottleneck "$SEV_WARN" "Swap usage elevated: ${swap_used} MB"
        fi
    else
        # Linux: 'free' provides clear columns
        echo -e "  ${BOLD}Memory (MB):${RESET}"
        free -m | awk '
            NR==1 {printf "  %-12s %8s %8s %8s %8s\n", "", $1, $2, $3, $6}
            NR==2 {printf "  %-12s %8s %8s %8s %8s\n", "RAM", $2, $3, $4, $7}
            NR==3 {printf "  %-12s %8s %8s %8s\n", "Swap", $2, $3, $4}
        '

        # Parse available memory to calculate utilization percentage
        local avail_mb total_mb
        avail_mb=$(free -m | awk '/^Mem:/{print $7}')
        total_mb=$(free -m | awk '/^Mem:/{print $2}')
        local used_pct=$(( (total_mb - avail_mb) * 100 / total_mb ))

        echo ""
        echo -e "  RAM Utilization : ${used_pct}%"

        local sev="$SEV_OK"
        if [[ $used_pct -ge 90 ]]; then
            sev="$SEV_CRIT"; flag_bottleneck "$sev" "Memory critical: ${used_pct}% used"
        elif [[ $used_pct -ge 75 ]]; then
            sev="$SEV_WARN"; flag_bottleneck "$sev" "Memory elevated: ${used_pct}% used"
        fi
        echo -e "  Memory Pressure : $(sev_color "$sev")"
    fi

    echo ""
    echo -e "  ${BOLD}Top 5 memory-consuming processes:${RESET}"
    ps -A -o pmem,pid,comm 2>/dev/null \
        | sort -rn \
        | head -5 \
        | awk '{printf "    %6s%%  PID %-6s  %s\n", $1, $2, $3}'
}

# ---------------------------------------------------------------------------
# DISK I/O ANALYSIS
# Metrics: disk utilization %, read/write throughput
#
# iostat's %util field: percentage of time the device was busy.
# Values near 100% indicate the disk is saturated (I/O bottleneck).
# High await (average I/O wait time) with high %util = disk is the bottleneck.
# ---------------------------------------------------------------------------
analyze_disk() {
    section "Disk I/O Analysis"

    # Disk space — always available
    echo -e "  ${BOLD}Filesystem Usage:${RESET}"
    df -h | awk 'NR==1 || /^\// {printf "  %-30s %6s %6s %6s %5s  %s\n", $1, $2, $3, $4, $5, $6}' \
        | head -8

    # Check for critically full filesystems
    df -h | awk 'NR>1 && /^\// {gsub(/%/,"",$5); if ($5+0 >= 90) print $6, $5}' \
    | while read -r mount pct; do
        flag_bottleneck "$SEV_CRIT" "Filesystem $mount at ${pct}% — nearly full"
        echo -e "  ${RED}[CRIT]${RESET} Filesystem $mount at ${pct}%"
    done

    echo ""
    echo -e "  ${BOLD}Disk I/O Throughput:${RESET}"

    if has_cmd iostat; then
        if [[ "$OS" == "Darwin" ]]; then
            # macOS iostat: columns are different from Linux
            echo -e "  (macOS iostat — KB/t=KB per transfer, tps=transfers/sec)"
            iostat -d 1 2 2>/dev/null | tail -n +4 | head -6 \
                | awk '{printf "  %-12s KB/t=%8s  tps=%8s\n", $1, $2, $3}' || true
        else
            # Linux iostat -x: extended stats including %util and await
            echo -e "  (Linux iostat -x: %util=disk busy%, await=avg I/O wait ms)"
            iostat -dx 1 2 2>/dev/null \
                | awk '/^[svhm]d|^nvme/ {printf "  %-12s  %util=%6s%%  await=%6s ms  r/s=%6s  w/s=%6s\n", $1, $NF, $10, $4, $5}' \
                | head -6 || true

            # Flag disks with high utilization
            iostat -dx 1 2 2>/dev/null \
                | awk '/^[svhm]d|^nvme/ {util=$NF+0; dev=$1; if (util>=80) print dev, util}' \
                | while read -r dev util; do
                    flag_bottleneck "$SEV_WARN" "Disk $dev I/O utilization ${util}%"
                done
        fi
    else
        echo -e "  ${YELLOW}iostat not found.${RESET} Install sysstat (Linux) or brew install sysstat (macOS)"
        echo -e "  Showing /proc/diskstats snapshot instead (Linux only):"
        if [[ -f /proc/diskstats ]]; then
            awk 'NF>=14 && $3~/^[svhm]d|^nvme/ {printf "  %-10s reads=%s writes=%s\n", $3, $6, $10}' \
                /proc/diskstats | head -5
        fi
    fi
}

# ---------------------------------------------------------------------------
# NETWORK ANALYSIS
# Metrics: active connections by state, listening ports, interface stats
#
# TIME_WAIT: normal; connections waiting for duplicate packets to expire (2*MSL)
# CLOSE_WAIT: may indicate app not closing sockets — potential resource leak
# High connection counts on a single port may indicate a traffic spike or DoS
# ---------------------------------------------------------------------------
analyze_network() {
    section "Network Analysis"

    echo -e "  ${BOLD}Connection State Summary:${RESET}"

    # Prefer 'ss' (socket statistics, modern Linux) over netstat
    if has_cmd ss; then
        ss -tan 2>/dev/null | awk 'NR>1 {states[$1]++} END {for (s in states) printf "  %-15s : %d\n", s, states[s]}' | sort
    elif has_cmd netstat; then
        netstat -an 2>/dev/null | awk '/^tcp/ {states[$6]++} END {for (s in states) printf "  %-15s : %d\n", s, states[s]}' | sort
    else
        echo -e "  ${YELLOW}Neither ss nor netstat found${RESET}"
    fi

    echo ""
    echo -e "  ${BOLD}Listening Ports (TCP):${RESET}"
    if has_cmd ss; then
        ss -tlnp 2>/dev/null | awk 'NR>1 {printf "  %-25s %s\n", $4, $6}' | head -10
    elif has_cmd netstat; then
        netstat -tlnp 2>/dev/null | awk 'NR>2 && /LISTEN/ {printf "  %-25s %s\n", $4, $7}' | head -10
    fi

    echo ""
    echo -e "  ${BOLD}Network Interface Statistics:${RESET}"
    if [[ "$OS" == "Darwin" ]]; then
        netstat -ib 2>/dev/null | awk 'NR==1 || /en[0-9]/' | head -6
    elif [[ -f /proc/net/dev ]]; then
        awk 'NR>2 && !/lo:/ {
            gsub(/:/, " ");
            printf "  %-10s  RX: %s bytes  TX: %s bytes\n", $1, $2, $10
        }' /proc/net/dev | head -5
    fi

    # Flag large TIME_WAIT counts as a potential issue
    local tw_count=0
    if has_cmd ss; then
        tw_count=$(ss -tan 2>/dev/null | grep -c TIME-WAIT || echo 0)
    fi
    if [[ $tw_count -gt 500 ]]; then
        flag_bottleneck "$SEV_WARN" "High TIME_WAIT count: $tw_count (consider tcp_tw_reuse)"
        echo -e "  ${YELLOW}[WARN]${RESET} High TIME_WAIT connections: $tw_count"
    fi
}

# ---------------------------------------------------------------------------
# SUMMARY REPORT — consolidate all bottleneck findings
# ---------------------------------------------------------------------------
print_summary() {
    section "Performance Bottleneck Summary"

    if [[ ${#BOTTLENECKS[@]} -eq 0 ]]; then
        echo -e "  ${GREEN}${BOLD}No significant bottlenecks detected.${RESET}"
        echo -e "  System appears healthy across CPU, memory, disk, and network."
    else
        echo -e "  ${BOLD}Detected ${#BOTTLENECKS[@]} issue(s):${RESET}"
        echo ""
        for b in "${BOTTLENECKS[@]}"; do
            if [[ "$b" == *"[CRIT]"* ]]; then
                echo -e "  ${RED}$b${RESET}"
            elif [[ "$b" == *"[WARN]"* ]]; then
                echo -e "  ${YELLOW}$b${RESET}"
            else
                echo -e "  ${GREEN}$b${RESET}"
            fi
        done
        echo ""
        echo -e "  ${BOLD}Next steps:${RESET}"
        echo -e "  - CPU: profile with 'perf top' or 'flamegraph' to find hot functions"
        echo -e "  - Memory: use 'smem' or 'valgrind' to detect leaks"
        echo -e "  - Disk: check application I/O patterns with 'iotop' or 'blktrace'"
        echo -e "  - Network: capture traffic with 'tcpdump' or 'Wireshark'"
    fi
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
main() {
    echo -e "${BOLD}Performance Diagnostics — $(date '+%Y-%m-%d %H:%M:%S') — $OS${RESET}"
    system_overview

    case "${1:---all}" in
        --cpu)     analyze_cpu ;;
        --memory)  analyze_memory ;;
        --disk)    analyze_disk ;;
        --network) analyze_network ;;
        --all)
            analyze_cpu
            analyze_memory
            analyze_disk
            analyze_network
            print_summary
            ;;
        *)
            echo "Usage: $0 [--cpu|--memory|--disk|--network|--all]"
            exit 1
            ;;
    esac
}

main "$@"
