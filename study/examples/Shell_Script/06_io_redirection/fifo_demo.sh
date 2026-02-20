#!/usr/bin/env bash
set -euo pipefail

# Named Pipe (FIFO) Demonstrations
# Shows how to use named pipes for inter-process communication

# Cleanup function
cleanup() {
    echo
    echo "Cleaning up..."
    rm -f /tmp/demo_fifo_* /tmp/producer_done
    jobs -p | xargs -r kill 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

echo "=== Named Pipe (FIFO) Demonstrations ==="
echo

# --- Demo 1: Basic FIFO usage ---
demo_basic_fifo() {
    echo "1. Basic FIFO Usage"
    echo "-------------------"

    local fifo="/tmp/demo_fifo_basic"

    # Create named pipe
    mkfifo "$fifo"
    echo "Created FIFO: $fifo"

    # Producer (background)
    {
        echo "Producer: Starting..."
        for i in {1..5}; do
            echo "Message $i"
            sleep 0.5
        done
    } > "$fifo" &

    # Consumer (foreground)
    echo "Consumer: Reading from FIFO..."
    while IFS= read -r line; do
        echo "  Received: $line"
    done < "$fifo"

    echo "Consumer: Done"

    rm -f "$fifo"
    echo
}

# --- Demo 2: Producer-Consumer pattern ---
demo_producer_consumer() {
    echo "2. Producer-Consumer Pattern"
    echo "----------------------------"

    local fifo="/tmp/demo_fifo_pc"
    local done_flag="/tmp/producer_done"

    mkfifo "$fifo"
    rm -f "$done_flag"

    # Producer: Generate data
    producer() {
        echo "Producer: Generating data..."
        for i in {1..10}; do
            # Simulate some work
            local result=$((i * i))
            echo "$i squared is $result"
            sleep 0.2
        done
        echo "DONE"  # Signal completion
        touch "$done_flag"
    }

    # Consumer: Process data
    consumer() {
        echo "Consumer: Processing data..."
        local count=0
        while IFS= read -r line; do
            if [[ "$line" == "DONE" ]]; then
                break
            fi
            ((count++))
            echo "  [$count] Processed: $line"
        done
        echo "Consumer: Processed $count items"
    }

    # Start producer in background
    producer > "$fifo" &
    local producer_pid=$!

    # Start consumer in foreground
    consumer < "$fifo"

    # Wait for producer to finish
    wait "$producer_pid"

    rm -f "$fifo" "$done_flag"
    echo
}

# --- Demo 3: Bidirectional communication ---
demo_bidirectional() {
    echo "3. Bidirectional Communication (Two FIFOs)"
    echo "------------------------------------------"

    local request_fifo="/tmp/demo_fifo_request"
    local response_fifo="/tmp/demo_fifo_response"

    mkfifo "$request_fifo"
    mkfifo "$response_fifo"

    # Server: Responds to requests
    server() {
        echo "Server: Starting..."
        while IFS= read -r request; do
            if [[ "$request" == "QUIT" ]]; then
                echo "GOODBYE" > "$response_fifo"
                break
            fi

            # Process request (simple echo server with uppercase)
            local response="${request^^}"
            echo "Server: Request='$request', Response='$response'"
            echo "$response" > "$response_fifo"
        done < "$request_fifo"
        echo "Server: Shutting down"
    }

    # Client: Sends requests
    client() {
        echo "Client: Connecting..."
        local -a requests=("hello" "world" "test" "QUIT")

        for req in "${requests[@]}"; do
            echo "Client: Sending '$req'"
            echo "$req" > "$request_fifo"

            # Read response
            IFS= read -r response < "$response_fifo"
            echo "Client: Got response '$response'"
            sleep 0.3
        done
        echo "Client: Done"
    }

    # Start server in background
    server &
    local server_pid=$!

    # Give server time to start
    sleep 0.5

    # Run client
    client

    # Wait for server
    wait "$server_pid"

    rm -f "$request_fifo" "$response_fifo"
    echo
}

# --- Demo 4: Load balancing with multiple workers ---
demo_load_balancing() {
    echo "4. Load Balancing with Multiple Workers"
    echo "----------------------------------------"

    local job_fifo="/tmp/demo_fifo_jobs"
    mkfifo "$job_fifo"

    # Worker function
    worker() {
        local worker_id=$1
        echo "Worker $worker_id: Started"

        while IFS= read -r job; do
            if [[ "$job" == "STOP" ]]; then
                break
            fi

            # Simulate work
            echo "  Worker $worker_id: Processing job '$job'"
            sleep 0.5
            echo "  Worker $worker_id: Completed job '$job'"
        done

        echo "Worker $worker_id: Stopped"
    }

    # Start multiple workers
    local num_workers=3
    echo "Starting $num_workers workers..."

    for i in $(seq 1 $num_workers); do
        worker "$i" < "$job_fifo" &
    done

    # Give workers time to start
    sleep 0.5

    # Dispatcher: Send jobs
    echo
    echo "Dispatcher: Sending jobs..."
    {
        for job_num in {1..6}; do
            echo "Job-$job_num"
        done

        # Send stop signal to each worker
        for i in $(seq 1 $num_workers); do
            echo "STOP"
        done
    } > "$job_fifo"

    # Wait for all workers
    wait

    rm -f "$job_fifo"
    echo
}

# --- Demo 5: Pipeline with FIFO ---
demo_pipeline() {
    echo "5. Complex Pipeline with FIFO"
    echo "-----------------------------"

    local fifo1="/tmp/demo_fifo_pipe1"
    local fifo2="/tmp/demo_fifo_pipe2"

    mkfifo "$fifo1" "$fifo2"

    # Stage 1: Generate numbers
    {
        echo "Stage 1: Generating numbers..."
        seq 1 10
    } > "$fifo1" &

    # Stage 2: Square the numbers
    {
        echo "Stage 2: Squaring..."
        while read -r num; do
            echo $((num * num))
        done
    } < "$fifo1" > "$fifo2" &

    # Stage 3: Sum the results
    {
        echo "Stage 3: Summing..."
        local sum=0
        while read -r num; do
            ((sum += num))
        done
        echo "  Total sum: $sum"
    } < "$fifo2"

    # Wait for all stages
    wait

    rm -f "$fifo1" "$fifo2"
    echo
}

# --- Demo 6: Monitoring with timeout ---
demo_timeout() {
    echo "6. FIFO with Timeout"
    echo "--------------------"

    local fifo="/tmp/demo_fifo_timeout"
    mkfifo "$fifo"

    # Slow producer
    {
        sleep 2
        echo "Delayed message"
    } > "$fifo" &
    local producer_pid=$!

    echo "Consumer: Waiting for message (3 second timeout)..."

    # Read with timeout using a background process
    {
        sleep 3
        echo "TIMEOUT"
    } > "$fifo" &
    local timeout_pid=$!

    # Read first message that arrives
    IFS= read -r message < "$fifo"

    if [[ "$message" == "TIMEOUT" ]]; then
        echo "  Result: Timed out!"
        kill $producer_pid 2>/dev/null || true
    else
        echo "  Result: Got message: '$message'"
        kill $timeout_pid 2>/dev/null || true
    fi

    wait 2>/dev/null || true

    rm -f "$fifo"
    echo
}

# Main execution
main() {
    demo_basic_fifo
    demo_producer_consumer
    demo_bidirectional
    demo_load_balancing
    demo_pipeline
    demo_timeout

    echo "=== All Demos Complete ==="
}

main "$@"
