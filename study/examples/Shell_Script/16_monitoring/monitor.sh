#!/usr/bin/env bash
set -euo pipefail

# System Monitoring Dashboard
# Terminal-based real-time system monitoring with color-coded alerts
# Cross-platform support for Linux and macOS

# ============================================================================
# Configuration
# ============================================================================

REFRESH_INTERVAL=${REFRESH_INTERVAL:-2}  # seconds

# Alert thresholds
CPU_WARNING=70
CPU_CRITICAL=90
MEM_WARNING=75
MEM_CRITICAL=90
DISK_WARNING=80
DISK_CRITICAL=95

# Platform detection
OS_TYPE=$(uname -s)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ============================================================================
# Terminal Control
# ============================================================================

# Save cursor position and hide cursor
init_terminal() {
    tput smcup      # Save screen
    tput civis      # Hide cursor
    clear
}

# Restore terminal on exit
cleanup_terminal() {
    tput rmcup      # Restore screen
    tput cnorm      # Show cursor
    clear
}

# Position cursor at row, col
move_cursor() {
    local row=$1
    local col=$2
    tput cup "$row" "$col"
}

# Get terminal dimensions
get_terminal_size() {
    TERM_ROWS=$(tput lines)
    TERM_COLS=$(tput cols)
}

# ============================================================================
# Metric Collection Functions
# ============================================================================

get_cpu_usage() {
    if [[ "$OS_TYPE" == "Linux" ]]; then
        # Linux: use top or /proc/stat
        if command -v top &> /dev/null; then
            top -bn2 -d 0.1 | grep '^%Cpu' | tail -n1 | awk '{print int(100 - $8)}'
        else
            echo "0"
        fi
    elif [[ "$OS_TYPE" == "Darwin" ]]; then
        # macOS: use top
        top -l 2 -n 0 -F | grep 'CPU usage' | tail -n1 | awk '{print int($3 + $5)}'
    else
        echo "0"
    fi
}

get_memory_usage() {
    if [[ "$OS_TYPE" == "Linux" ]]; then
        # Linux: use /proc/meminfo
        if [[ -f /proc/meminfo ]]; then
            local total
            local available
            total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
            available=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
            echo $(( (total - available) * 100 / total ))
        else
            echo "0"
        fi
    elif [[ "$OS_TYPE" == "Darwin" ]]; then
        # macOS: use vm_stat
        local page_size
        local free_pages
        local active_pages
        local inactive_pages
        local wired_pages

        page_size=$(vm_stat | grep 'page size' | awk '{print $8}')
        free_pages=$(vm_stat | grep 'Pages free' | awk '{print $3}' | tr -d '.')
        active_pages=$(vm_stat | grep 'Pages active' | awk '{print $3}' | tr -d '.')
        inactive_pages=$(vm_stat | grep 'Pages inactive' | awk '{print $3}' | tr -d '.')
        wired_pages=$(vm_stat | grep 'Pages wired' | awk '{print $4}' | tr -d '.')

        local used=$((active_pages + inactive_pages + wired_pages))
        local total=$((used + free_pages))

        echo $((used * 100 / total))
    else
        echo "0"
    fi
}

get_disk_usage() {
    # Get disk usage for root filesystem
    df -h / | awk 'NR==2 {print int($5)}'
}

get_load_average() {
    if [[ "$OS_TYPE" == "Linux" ]] || [[ "$OS_TYPE" == "Darwin" ]]; then
        uptime | awk -F'load average:' '{print $2}' | awk '{print $1, $2, $3}' | tr -d ','
    else
        echo "0.00 0.00 0.00"
    fi
}

get_top_processes() {
    local count=${1:-5}

    if [[ "$OS_TYPE" == "Linux" ]]; then
        ps aux --sort=-%cpu | head -n $((count + 1)) | tail -n $count | awk '{printf "%-15s %5s%% %5s%%\n", substr($11,1,15), $3, $4}'
    elif [[ "$OS_TYPE" == "Darwin" ]]; then
        ps aux -r | head -n $((count + 1)) | tail -n $count | awk '{printf "%-15s %5s%% %5s%%\n", substr($11,1,15), $3, $4}'
    fi
}

# ============================================================================
# Display Functions
# ============================================================================

get_color() {
    local value=$1
    local warning=$2
    local critical=$3

    if [[ $value -ge $critical ]]; then
        echo -e "$RED"
    elif [[ $value -ge $warning ]]; then
        echo -e "$YELLOW"
    else
        echo -e "$GREEN"
    fi
}

draw_box() {
    local row=$1
    local col=$2
    local width=$3
    local height=$4
    local title=$5

    # Top border
    move_cursor "$row" "$col"
    echo -ne "${CYAN}┌"
    printf '─%.0s' $(seq 1 $((width - 2)))
    echo -ne "┐${NC}"

    # Title
    move_cursor "$row" $((col + 2))
    echo -ne "${BOLD}${title}${NC}"

    # Side borders
    local i
    for ((i = 1; i < height - 1; i++)); do
        move_cursor $((row + i)) "$col"
        echo -ne "${CYAN}│${NC}"
        move_cursor $((row + i)) $((col + width - 1))
        echo -ne "${CYAN}│${NC}"
    done

    # Bottom border
    move_cursor $((row + height - 1)) "$col"
    echo -ne "${CYAN}└"
    printf '─%.0s' $(seq 1 $((width - 2)))
    echo -ne "┘${NC}"
}

draw_bar() {
    local value=$1
    local max_width=$2
    local warning=$3
    local critical=$4

    local filled=$((value * max_width / 100))
    local color
    color=$(get_color "$value" "$warning" "$critical")

    echo -ne "$color"
    printf '█%.0s' $(seq 1 "$filled")
    echo -ne "$NC"
    printf '░%.0s' $(seq 1 $((max_width - filled)))
}

draw_metric_panel() {
    local row=$1
    local col=$2
    local width=$3
    local title=$4
    local value=$5
    local warning=$6
    local critical=$7

    draw_box "$row" "$col" "$width" 5 "$title"

    # Value
    move_cursor $((row + 2)) $((col + 2))
    local color
    color=$(get_color "$value" "$warning" "$critical")
    printf "%s%3d%%%s" "$color" "$value" "$NC"

    # Progress bar
    move_cursor $((row + 3)) $((col + 2))
    draw_bar "$value" $((width - 4)) "$warning" "$critical"
}

draw_header() {
    move_cursor 0 0
    echo -ne "${BOLD}${CYAN}"
    printf '═%.0s' $(seq 1 "$TERM_COLS")
    echo -ne "$NC"

    move_cursor 0 2
    echo -ne "${BOLD}System Monitor${NC}"

    move_cursor 0 $((TERM_COLS - 30))
    echo -ne "${BOLD}$(date '+%Y-%m-%d %H:%M:%S')${NC}"

    move_cursor 1 0
    echo -ne "${BOLD}${CYAN}"
    printf '═%.0s' $(seq 1 "$TERM_COLS")
    echo -ne "$NC"
}

draw_dashboard() {
    # Collect metrics
    local cpu_usage
    local mem_usage
    local disk_usage
    local load_avg

    cpu_usage=$(get_cpu_usage)
    mem_usage=$(get_memory_usage)
    disk_usage=$(get_disk_usage)
    load_avg=$(get_load_average)

    # Clear and draw
    clear
    draw_header

    # Calculate layout
    local panel_width=$((TERM_COLS / 2 - 3))

    # Top row - 4 metric panels
    local panel_w=$((TERM_COLS / 4 - 2))
    draw_metric_panel 3 2 "$panel_w" "CPU" "$cpu_usage" "$CPU_WARNING" "$CPU_CRITICAL"
    draw_metric_panel 3 $((panel_w + 3)) "$panel_w" "Memory" "$mem_usage" "$MEM_WARNING" "$MEM_CRITICAL"
    draw_metric_panel 3 $((panel_w * 2 + 4)) "$panel_w" "Disk" "$disk_usage" "$DISK_WARNING" "$DISK_CRITICAL"

    # Load average
    draw_box 3 $((panel_w * 3 + 5)) "$panel_w" 5 "Load Avg"
    move_cursor 5 $((panel_w * 3 + 7))
    echo -ne "$load_avg"

    # Process list
    local proc_row=9
    draw_box "$proc_row" 2 $((TERM_COLS - 4)) 12 "Top Processes (CPU)"

    move_cursor $((proc_row + 2)) 4
    printf "${BOLD}%-15s %5s  %5s${NC}" "COMMAND" "CPU%" "MEM%"

    local line=0
    while IFS= read -r process; do
        move_cursor $((proc_row + 3 + line)) 4
        echo -ne "$process"
        ((line++))
    done < <(get_top_processes 8)

    # Status line
    move_cursor $((TERM_ROWS - 2)) 2
    echo -ne "${BOLD}Press Ctrl+C to exit | Refresh: ${REFRESH_INTERVAL}s${NC}"
}

# ============================================================================
# Main Loop
# ============================================================================

main() {
    # Initialize
    init_terminal
    trap cleanup_terminal EXIT INT TERM

    # Main monitoring loop
    while true; do
        get_terminal_size

        # Ensure terminal is large enough
        if [[ $TERM_ROWS -lt 24 ]] || [[ $TERM_COLS -lt 80 ]]; then
            clear
            move_cursor 2 2
            echo -e "${RED}Terminal too small!${NC}"
            move_cursor 3 2
            echo "Minimum size: 80x24"
            move_cursor 4 2
            echo "Current size: ${TERM_COLS}x${TERM_ROWS}"
            sleep 1
            continue
        fi

        # Draw dashboard
        draw_dashboard

        # Wait for refresh interval
        sleep "$REFRESH_INTERVAL"
    done
}

main "$@"
