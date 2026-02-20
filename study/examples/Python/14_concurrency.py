"""
Python Concurrency: Threading and Multiprocessing

Demonstrates:
- threading basics
- Thread synchronization (Lock, RLock)
- ThreadPoolExecutor
- multiprocessing.Pool
- Queue for producer-consumer
- GIL limitations
- When to use threads vs processes
"""

import threading
import multiprocessing
import time
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Basic Threading
# =============================================================================

section("Basic Threading")


def worker(name: str, delay: float):
    """Simple worker function."""
    print(f"  {name}: Starting (delay={delay}s)")
    time.sleep(delay)
    print(f"  {name}: Finished")


# Create and start threads
threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(f"Thread-{i}", 0.1 * (i + 1)))
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()

print("All threads completed")


# =============================================================================
# Thread with Return Value
# =============================================================================

section("Thread with Return Value")


def compute_square(n: int) -> int:
    """Compute square of number."""
    time.sleep(0.05)
    return n * n


class ThreadWithReturnValue(threading.Thread):
    """Thread that stores return value."""

    def __init__(self, target, args=()):
        super().__init__(target=target, args=args)
        self.result = None

    def run(self):
        self.result = self._target(*self._args)


threads = []
for i in range(5):
    t = ThreadWithReturnValue(target=compute_square, args=(i,))
    t.start()
    threads.append(t)

results = []
for t in threads:
    t.join()
    results.append(t.result)

print(f"Results: {results}")


# =============================================================================
# Thread Synchronization - Lock
# =============================================================================

section("Thread Synchronization - Lock")

counter = 0
counter_lock = threading.Lock()


def increment_with_lock(name: str, iterations: int):
    """Increment counter with lock."""
    global counter
    for _ in range(iterations):
        with counter_lock:  # Acquire lock
            counter += 1


# Without lock (race condition)
counter = 0
threads = [
    threading.Thread(target=lambda: increment_with_lock("T1", 1000)),
    threading.Thread(target=lambda: increment_with_lock("T2", 1000)),
]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Counter with lock: {counter} (expected: 2000)")


# =============================================================================
# ThreadPoolExecutor
# =============================================================================

section("ThreadPoolExecutor")


def download_file(file_id: int) -> dict:
    """Simulate file download."""
    time.sleep(0.1)
    return {"id": file_id, "size": file_id * 1024}


# Using ThreadPoolExecutor
start = time.perf_counter()

with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit tasks
    futures = [executor.submit(download_file, i) for i in range(10)]

    # Collect results
    results = [future.result() for future in futures]

elapsed = time.perf_counter() - start

print(f"Downloaded {len(results)} files in {elapsed:.2f}s")
print(f"Results: {results[:3]}...")


# Using map()
section("ThreadPoolExecutor with map()")

with ThreadPoolExecutor(max_workers=4) as executor:
    file_ids = range(10)
    results = list(executor.map(download_file, file_ids))

print(f"Downloaded {len(results)} files using map()")


# =============================================================================
# Producer-Consumer with Queue
# =============================================================================

section("Producer-Consumer with Queue")

task_queue = queue.Queue()
result_queue = queue.Queue()


def producer(num_items: int):
    """Produce items."""
    for i in range(num_items):
        task_queue.put(i)
        print(f"  Producer: Added {i}")
        time.sleep(0.02)
    print("  Producer: Done")


def consumer(worker_id: int):
    """Consume items."""
    while True:
        try:
            item = task_queue.get(timeout=0.5)
            print(f"  Consumer-{worker_id}: Processing {item}")
            result = item * 2
            result_queue.put(result)
            time.sleep(0.05)
            task_queue.task_done()
        except queue.Empty:
            break


# Start producer
producer_thread = threading.Thread(target=producer, args=(10,))
producer_thread.start()

# Start consumers
consumer_threads = []
for i in range(3):
    t = threading.Thread(target=consumer, args=(i,))
    t.start()
    consumer_threads.append(t)

# Wait for completion
producer_thread.join()
task_queue.join()

for t in consumer_threads:
    t.join()

print(f"\nResults collected: {result_queue.qsize()} items")


# =============================================================================
# Multiprocessing - CPU-Bound Tasks
# =============================================================================

section("Multiprocessing - CPU-Bound Tasks")


def cpu_intensive(n: int) -> int:
    """CPU-intensive task."""
    total = 0
    for i in range(n):
        total += i * i
    return total


# Sequential
start = time.perf_counter()
results_seq = [cpu_intensive(1000000) for _ in range(4)]
time_seq = time.perf_counter() - start

# Multiprocessing
start = time.perf_counter()
with multiprocessing.Pool(processes=4) as pool:
    results_mp = pool.map(cpu_intensive, [1000000] * 4)
time_mp = time.perf_counter() - start

print(f"Sequential: {time_seq:.2f}s")
print(f"Multiprocessing: {time_mp:.2f}s")
print(f"Speedup: {time_seq / time_mp:.2f}x")


# =============================================================================
# ProcessPoolExecutor
# =============================================================================

section("ProcessPoolExecutor")


def fibonacci(n: int) -> int:
    """Compute Fibonacci (CPU-bound)."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


numbers = [100000, 200000, 300000, 400000]

# Using ProcessPoolExecutor
start = time.perf_counter()
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(fibonacci, numbers))
elapsed = time.perf_counter() - start

print(f"Computed {len(results)} Fibonacci numbers in {elapsed:.2f}s")
print(f"Last 10 digits: {[str(r)[-10:] for r in results]}")


# =============================================================================
# as_completed - Process Results as They Finish
# =============================================================================

section("as_completed - Process Results as They Finish")


def slow_task(task_id: int) -> dict:
    """Task with variable duration."""
    duration = 0.1 * (4 - task_id % 4)  # 0.1s to 0.4s
    time.sleep(duration)
    return {"id": task_id, "duration": duration}


with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(slow_task, i): i for i in range(8)}

    print("Processing results as they complete:")
    for future in as_completed(futures):
        result = future.result()
        print(f"  Task {result['id']} completed (took {result['duration']:.2f}s)")


# =============================================================================
# Thread-Local Storage
# =============================================================================

section("Thread-Local Storage")

thread_local = threading.local()


def use_thread_local(name: str):
    """Each thread has its own local storage."""
    # Set thread-local value
    thread_local.data = f"{name}-data"

    time.sleep(0.1)

    # Access thread-local value
    print(f"  {name}: {thread_local.data}")


threads = [
    threading.Thread(target=use_thread_local, args=(f"Thread-{i}",))
    for i in range(3)
]

for t in threads:
    t.start()
for t in threads:
    t.join()


# =============================================================================
# Daemon Threads
# =============================================================================

section("Daemon Threads")


def daemon_worker():
    """Daemon thread runs in background."""
    print("  Daemon: Starting")
    time.sleep(5)  # Long-running task
    print("  Daemon: Finished")  # Won't print if main exits


daemon = threading.Thread(target=daemon_worker, daemon=True)
daemon.start()

print("Main: Started daemon thread")
time.sleep(0.1)
print("Main: Exiting (daemon will be killed)")


# =============================================================================
# GIL Demonstration
# =============================================================================

section("GIL - Global Interpreter Lock")

print("""
Python's Global Interpreter Lock (GIL):
- Only one thread executes Python bytecode at a time
- Protects Python object memory
- Prevents true parallel execution of Python code

Impact:
- CPU-bound tasks: Threading provides NO speedup (GIL limitation)
  → Use multiprocessing for CPU-bound tasks
- I/O-bound tasks: Threading DOES provide speedup (releases GIL during I/O)
  → Use threading for I/O-bound tasks (network, disk)

Example speedup comparison:
""")


def io_bound_task():
    """I/O-bound - releases GIL during sleep."""
    time.sleep(0.1)
    return "done"


def cpu_bound_task():
    """CPU-bound - holds GIL."""
    return sum(i * i for i in range(100000))


# I/O-bound with threads (good speedup)
start = time.perf_counter()
with ThreadPoolExecutor(max_workers=4) as executor:
    list(executor.map(lambda _: io_bound_task(), range(4)))
time_io_threads = time.perf_counter() - start

# CPU-bound with threads (no speedup due to GIL)
start = time.perf_counter()
with ThreadPoolExecutor(max_workers=4) as executor:
    list(executor.map(lambda _: cpu_bound_task(), range(4)))
time_cpu_threads = time.perf_counter() - start

# CPU-bound with processes (good speedup, no GIL)
start = time.perf_counter()
with ProcessPoolExecutor(max_workers=4) as executor:
    list(executor.map(lambda _: cpu_bound_task(), range(4)))
time_cpu_processes = time.perf_counter() - start

print(f"I/O-bound with threads: {time_io_threads:.3f}s (good)")
print(f"CPU-bound with threads: {time_cpu_threads:.3f}s (no speedup)")
print(f"CPU-bound with processes: {time_cpu_processes:.3f}s (good speedup)")


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Threading:
- Use for I/O-bound tasks (network, disk, database)
- Lightweight (shared memory)
- GIL prevents true parallelism for CPU-bound tasks
- threading.Thread - basic threading
- ThreadPoolExecutor - high-level interface
- Lock/RLock - synchronization
- Queue - thread-safe communication

Multiprocessing:
- Use for CPU-bound tasks (computation, data processing)
- Separate Python interpreter per process (no GIL)
- Higher overhead (separate memory)
- multiprocessing.Pool - pool of worker processes
- ProcessPoolExecutor - high-level interface

Guidelines:
┌─────────────────────┬──────────────┬──────────────────┐
│ Task Type           │ Use          │ Reason           │
├─────────────────────┼──────────────┼──────────────────┤
│ I/O-bound          │ Threading    │ GIL released     │
│ CPU-bound          │ Multiprocess │ No GIL           │
│ Mixed              │ Both         │ Combine          │
│ Simple concurrency │ asyncio      │ Single-threaded  │
└─────────────────────┴──────────────┴──────────────────┘

ThreadPoolExecutor vs ProcessPoolExecutor:
- Same interface (concurrent.futures)
- Easy to switch between them
- Use ProcessPoolExecutor for CPU-intensive work
- Use ThreadPoolExecutor for I/O-intensive work

Synchronization primitives:
- Lock - mutual exclusion
- RLock - reentrant lock
- Semaphore - limit concurrent access
- Event - thread signaling
- Condition - wait for condition
- Queue - thread-safe FIFO

Best practices:
1. Avoid shared state when possible
2. Use locks to protect shared state
3. Prefer ProcessPoolExecutor for CPU-bound
4. Prefer ThreadPoolExecutor for I/O-bound
5. Consider asyncio for simple I/O concurrency
6. Profile before optimizing
""")
