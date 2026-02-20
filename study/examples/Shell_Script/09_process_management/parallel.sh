#!/usr/bin/env bash
set -euo pipefail

# Parallel Execution Patterns
# Demonstrates running multiple tasks concurrently with controlled parallelism

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
# Parallel Execution Functions
# ============================================================================

# Run commands in parallel with concurrency limit
# Usage: run_parallel <max_jobs> <command1> <command2> ...
run_parallel() {
    local max_jobs="$1"
    shift
    local -a commands=("$@")

    echo -e "${CYAN}Running ${#commands[@]} commands with max $max_jobs parallel jobs${NC}"

    local -a pids=()
    local job_count=0

    for cmd in "${commands[@]}"; do
        # Wait if we've reached the concurrency limit
        while [[ ${#pids[@]} -ge $max_jobs ]]; do
            wait_for_any_pid pids
        done

        # Launch command in background
        (
            eval "$cmd"
        ) &

        local pid=$!
        pids+=("$pid")
        ((job_count++))

        echo -e "  ${GREEN}→${NC} Started job $job_count (PID $pid): $cmd"
    done

    # Wait for all remaining jobs
    echo -e "${CYAN}Waiting for remaining jobs to complete...${NC}"
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    echo -e "${GREEN}✓${NC} All jobs completed"
}

# Wait for any PID in the array to finish, then remove it
wait_for_any_pid() {
    local -n pid_array=$1

    # Wait for any child process
    wait -n 2>/dev/null || true

    # Remove finished PIDs from array
    local -a still_running=()
    for pid in "${pid_array[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            still_running+=("$pid")
        fi
    done

    pid_array=("${still_running[@]}")
}

# Wait for specific PIDs and collect their exit codes
# Returns: associative array of PID -> exit code
wait_for_pids() {
    local -a pids=("$@")

    echo -e "${CYAN}Waiting for ${#pids[@]} process(es)...${NC}"

    local -A exit_codes=()

    for pid in "${pids[@]}"; do
        local exit_code=0
        wait "$pid" || exit_code=$?

        exit_codes[$pid]=$exit_code

        if [[ $exit_code -eq 0 ]]; then
            echo -e "  ${GREEN}✓${NC} PID $pid completed successfully"
        else
            echo -e "  ${RED}✗${NC} PID $pid failed with exit code $exit_code"
        fi
    done

    # Print summary
    local failed_count=0
    for exit_code in "${exit_codes[@]}"; do
        [[ $exit_code -ne 0 ]] && ((failed_count++))
    done

    if [[ $failed_count -eq 0 ]]; then
        echo -e "${GREEN}✓${NC} All processes completed successfully"
    else
        echo -e "${YELLOW}⚠${NC} $failed_count process(es) failed"
    fi

    return $failed_count
}

# Simulate a download task
simulate_download() {
    local file_id="$1"
    local duration="$2"

    echo -e "  ${BLUE}[File $file_id]${NC} Starting download..."
    sleep "$duration"

    # Simulate occasional failures
    if [[ $((RANDOM % 10)) -eq 0 ]]; then
        echo -e "  ${RED}[File $file_id]${NC} Download failed!"
        return 1
    fi

    echo -e "  ${GREEN}[File $file_id]${NC} Download complete (${duration}s)"
    return 0
}

# Parallel download demonstration
parallel_download() {
    local max_parallel="$1"
    local -a files=("${@:2}")

    echo -e "${CYAN}Downloading ${#files[@]} files with max $max_parallel parallel downloads${NC}\n"

    local -a pids=()
    local start_time=$SECONDS

    for file_id in "${files[@]}"; do
        # Limit concurrency
        while [[ ${#pids[@]} -ge $max_parallel ]]; do
            wait_for_any_pid pids
        done

        # Random duration between 1-3 seconds
        local duration=$((1 + RANDOM % 3))

        simulate_download "$file_id" "$duration" &
        pids+=($!)
    done

    # Wait for remaining downloads
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    local elapsed=$((SECONDS - start_time))
    echo -e "\n${GREEN}✓${NC} All downloads completed in ${elapsed}s"
}

# ============================================================================
# Demo Section
# ============================================================================

demo_sequential_vs_parallel() {
    echo -e "\n${BLUE}=== Sequential vs Parallel Execution ===${NC}\n"

    # Define test tasks
    local -a tasks=(
        "sleep 1 && echo 'Task 1 done'"
        "sleep 1 && echo 'Task 2 done'"
        "sleep 1 && echo 'Task 3 done'"
        "sleep 1 && echo 'Task 4 done'"
    )

    # Sequential execution
    echo -e "${YELLOW}Sequential execution:${NC}"
    local start=$SECONDS
    for task in "${tasks[@]}"; do
        eval "$task"
    done
    local seq_time=$((SECONDS - start))
    echo -e "Time: ${seq_time}s\n"

    # Parallel execution
    echo -e "${YELLOW}Parallel execution (max 4 jobs):${NC}"
    start=$SECONDS
    run_parallel 4 "${tasks[@]}"
    local par_time=$((SECONDS - start))
    echo -e "Time: ${par_time}s\n"

    echo -e "${GREEN}Speedup:${NC} ~$((seq_time / par_time))x faster"
}

demo_parallel_with_limit() {
    echo -e "\n${BLUE}=== Parallel Execution with Concurrency Limit ===${NC}\n"

    local -a tasks=()
    for i in {1..8}; do
        tasks+=("sleep 0.5 && echo 'Task $i completed'")
    done

    echo -e "${YELLOW}Running 8 tasks with max 3 parallel jobs:${NC}"
    run_parallel 3 "${tasks[@]}"
}

demo_pid_tracking() {
    echo -e "\n${BLUE}=== PID Tracking and Exit Code Collection ===${NC}\n"

    # Start several background processes
    sleep 0.5 && exit 0 &
    local pid1=$!

    sleep 0.5 && exit 1 &
    local pid2=$!

    sleep 0.5 && exit 0 &
    local pid3=$!

    sleep 0.5 && exit 2 &
    local pid4=$!

    echo "Started 4 background processes: $pid1, $pid2, $pid3, $pid4"
    echo

    wait_for_pids "$pid1" "$pid2" "$pid3" "$pid4"
}

demo_parallel_downloads() {
    echo -e "\n${BLUE}=== Parallel Downloads Simulation ===${NC}\n"

    # Test with different concurrency levels
    local -a files=(1 2 3 4 5 6 7 8)

    echo -e "${YELLOW}Test 1: Sequential (1 at a time)${NC}"
    parallel_download 1 "${files[@]}"

    echo -e "\n${YELLOW}Test 2: Parallel (4 at a time)${NC}"
    parallel_download 4 "${files[@]}"
}

demo_job_control() {
    echo -e "\n${BLUE}=== Job Control Example ===${NC}\n"

    echo "Starting 5 background jobs..."

    local -a pids=()
    for i in {1..5}; do
        (
            sleep $((1 + RANDOM % 3))
            echo "Job $i finished"
        ) &
        pids+=($!)
        echo -e "  Started job $i (PID ${pids[-1]})"
    done

    echo
    echo "Active background jobs:"
    jobs -l

    echo
    echo "Waiting for all jobs to complete..."
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo -e "${GREEN}✓${NC} All jobs completed"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║      Parallel Execution Demo              ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"

    demo_sequential_vs_parallel
    demo_parallel_with_limit
    demo_pid_tracking
    demo_parallel_downloads
    demo_job_control

    echo -e "\n${GREEN}=== All Parallel Execution Demos Complete ===${NC}\n"
}

main "$@"
