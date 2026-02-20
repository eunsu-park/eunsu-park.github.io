"""
Threading Demonstration

Covers:
1. Thread pools
2. Producer-Consumer pattern with Queue
3. Locks and Semaphores
4. Race conditions (demonstration and fix)
5. Thread-safe data structures
"""

import threading
import queue
import time
import random
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed


# =============================================================================
# 1. BASIC THREADING
# =============================================================================

print("=" * 70)
print("1. BASIC THREADING")
print("=" * 70)


def worker(name: str, work_time: float):
    """Simple worker function"""
    print(f"[{name}] Starting work...")
    time.sleep(work_time)
    print(f"[{name}] Finished work!")
    return f"{name} completed"


def demonstrate_basic_threading():
    """Basic threading example"""
    print("\n[BASIC THREADING]")
    print("-" * 50)

    # Create threads
    threads = []
    for i in range(3):
        thread = threading.Thread(
            target=worker,
            args=(f"Worker-{i}", random.uniform(0.1, 0.5))
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All threads completed!")


# =============================================================================
# 2. THREAD POOL
# =============================================================================

print("\n" + "=" * 70)
print("2. THREAD POOL")
print("=" * 70)


def process_task(task_id: int) -> dict:
    """Simulate processing a task"""
    print(f"Processing task {task_id} on thread {threading.current_thread().name}")
    time.sleep(random.uniform(0.1, 0.5))
    return {"task_id": task_id, "result": task_id * 2}


def demonstrate_thread_pool():
    """Thread pool executor example"""
    print("\n[THREAD POOL]")
    print("-" * 50)

    tasks = range(10)

    # Use ThreadPoolExecutor for better resource management
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        futures = [executor.submit(process_task, task_id) for task_id in tasks]

        # Process results as they complete
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Task {result['task_id']} completed: {result['result']}")

    print(f"\nAll {len(results)} tasks completed!")


# =============================================================================
# 3. PRODUCER-CONSUMER PATTERN
# =============================================================================

print("\n" + "=" * 70)
print("3. PRODUCER-CONSUMER PATTERN")
print("=" * 70)


class ProducerConsumer:
    """Producer-Consumer pattern using Queue"""

    def __init__(self, max_queue_size: int = 5):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()

    def producer(self, name: str, num_items: int):
        """Producer thread function"""
        for i in range(num_items):
            if self.stop_event.is_set():
                break

            item = f"{name}-Item-{i}"
            print(f"[{name}] Producing {item}...")
            self.queue.put(item)  # Blocks if queue is full
            print(f"[{name}] Produced {item} (Queue size: {self.queue.qsize()})")
            time.sleep(random.uniform(0.1, 0.3))

        print(f"[{name}] Finished producing")

    def consumer(self, name: str):
        """Consumer thread function"""
        while not self.stop_event.is_set():
            try:
                # Wait for item with timeout
                item = self.queue.get(timeout=0.5)
                print(f"[{name}] Consuming {item}...")
                time.sleep(random.uniform(0.2, 0.4))  # Simulate processing
                self.queue.task_done()
                print(f"[{name}] Consumed {item}")
            except queue.Empty:
                # Queue is empty, check if we should stop
                continue

        print(f"[{name}] Finished consuming")

    def run(self, num_producers: int = 2, num_consumers: int = 3, items_per_producer: int = 5):
        """Run the producer-consumer demo"""
        threads = []

        # Start producers
        for i in range(num_producers):
            thread = threading.Thread(
                target=self.producer,
                args=(f"Producer-{i}", items_per_producer)
            )
            threads.append(thread)
            thread.start()

        # Start consumers
        for i in range(num_consumers):
            thread = threading.Thread(
                target=self.consumer,
                args=(f"Consumer-{i}",)
            )
            threads.append(thread)
            thread.start()

        # Wait for all producers to finish
        for thread in threads[:num_producers]:
            thread.join()

        # Wait for queue to be empty
        self.queue.join()

        # Signal consumers to stop
        self.stop_event.set()

        # Wait for all consumers to finish
        for thread in threads[num_producers:]:
            thread.join()


# =============================================================================
# 4. LOCKS AND SYNCHRONIZATION
# =============================================================================

print("\n" + "=" * 70)
print("4. LOCKS AND SYNCHRONIZATION")
print("=" * 70)


class BankAccount:
    """Thread-safe bank account using Lock"""

    def __init__(self, initial_balance: float = 0):
        self.balance = initial_balance
        self._lock = threading.Lock()

    def deposit(self, amount: float, name: str = ""):
        """Deposit money (thread-safe)"""
        with self._lock:  # Acquire lock
            current = self.balance
            time.sleep(0.001)  # Simulate processing delay
            self.balance = current + amount
            print(f"[{name}] Deposited ${amount:.2f}, New balance: ${self.balance:.2f}")

    def withdraw(self, amount: float, name: str = "") -> bool:
        """Withdraw money (thread-safe)"""
        with self._lock:
            if self.balance >= amount:
                current = self.balance
                time.sleep(0.001)  # Simulate processing delay
                self.balance = current - amount
                print(f"[{name}] Withdrew ${amount:.2f}, New balance: ${self.balance:.2f}")
                return True
            else:
                print(f"[{name}] Insufficient funds for ${amount:.2f}")
                return False

    def get_balance(self) -> float:
        """Get current balance (thread-safe)"""
        with self._lock:
            return self.balance


def demonstrate_locks():
    """Demonstrate lock usage"""
    print("\n[LOCKS]")
    print("-" * 50)

    account = BankAccount(1000.0)

    def make_transactions(name: str):
        """Perform multiple transactions"""
        for _ in range(3):
            if random.random() > 0.5:
                account.deposit(random.uniform(10, 50), name)
            else:
                account.withdraw(random.uniform(10, 50), name)
            time.sleep(0.01)

    # Create multiple threads performing transactions
    threads = []
    for i in range(3):
        thread = threading.Thread(target=make_transactions, args=(f"Thread-{i}",))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print(f"\nFinal balance: ${account.get_balance():.2f}")


# =============================================================================
# 5. RACE CONDITIONS (DEMONSTRATION AND FIX)
# =============================================================================

print("\n" + "=" * 70)
print("5. RACE CONDITIONS")
print("=" * 70)


class CounterWithRaceCondition:
    """Counter with deliberate race condition"""

    def __init__(self):
        self.count = 0

    def increment(self):
        """Increment counter (NOT thread-safe!)"""
        current = self.count
        time.sleep(0.0001)  # Simulate work - amplifies race condition
        self.count = current + 1


class ThreadSafeCounter:
    """Thread-safe counter using Lock"""

    def __init__(self):
        self.count = 0
        self._lock = threading.Lock()

    def increment(self):
        """Increment counter (thread-safe)"""
        with self._lock:
            current = self.count
            time.sleep(0.0001)
            self.count = current + 1


def demonstrate_race_condition():
    """Demonstrate race condition and its fix"""
    print("\n[RACE CONDITION DEMO]")
    print("-" * 50)

    # ❌ BAD: Counter with race condition
    print("WITHOUT LOCK (Race Condition):")
    unsafe_counter = CounterWithRaceCondition()

    def increment_unsafe(counter, times):
        for _ in range(times):
            counter.increment()

    threads = []
    num_threads = 10
    increments_per_thread = 100

    for _ in range(num_threads):
        thread = threading.Thread(target=increment_unsafe, args=(unsafe_counter, increments_per_thread))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    expected = num_threads * increments_per_thread
    print(f"Expected: {expected}")
    print(f"Actual: {unsafe_counter.count}")
    print(f"Lost updates: {expected - unsafe_counter.count}")

    # ✅ GOOD: Thread-safe counter
    print("\nWITH LOCK (Thread-Safe):")
    safe_counter = ThreadSafeCounter()

    def increment_safe(counter, times):
        for _ in range(times):
            counter.increment()

    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=increment_safe, args=(safe_counter, increments_per_thread))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print(f"Expected: {expected}")
    print(f"Actual: {safe_counter.count}")
    print(f"Lost updates: {expected - safe_counter.count}")


# =============================================================================
# 6. SEMAPHORE (LIMITED RESOURCES)
# =============================================================================

print("\n" + "=" * 70)
print("6. SEMAPHORE")
print("=" * 70)


class ResourcePool:
    """Limited resource pool using Semaphore"""

    def __init__(self, max_resources: int):
        self.semaphore = threading.Semaphore(max_resources)
        self.max_resources = max_resources

    def use_resource(self, name: str, duration: float):
        """Use a resource (blocks if all resources busy)"""
        print(f"[{name}] Waiting for resource...")
        with self.semaphore:  # Acquire resource
            print(f"[{name}] Acquired resource, using for {duration:.2f}s")
            time.sleep(duration)
            print(f"[{name}] Released resource")


def demonstrate_semaphore():
    """Demonstrate semaphore for resource limiting"""
    print("\n[SEMAPHORE DEMO]")
    print("-" * 50)

    # Only 2 concurrent resource users allowed
    pool = ResourcePool(max_resources=2)

    def worker(name: str):
        pool.use_resource(name, random.uniform(0.2, 0.5))

    # Create more threads than available resources
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker, args=(f"Worker-{i}",))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


# =============================================================================
# DEMONSTRATION
# =============================================================================

def print_summary():
    print("\n" + "=" * 70)
    print("THREADING SUMMARY")
    print("=" * 70)

    print("""
1. BASIC THREADING
   ✓ threading.Thread for simple parallelism
   ✓ .start() to begin execution
   ✓ .join() to wait for completion

2. THREAD POOL
   ✓ ThreadPoolExecutor for better resource management
   ✓ Automatic thread reuse
   ✓ Context manager ensures cleanup
   ✓ as_completed() for results as they finish

3. PRODUCER-CONSUMER
   ✓ queue.Queue for thread-safe communication
   ✓ Automatic blocking when full/empty
   ✓ task_done() and join() for synchronization

4. LOCKS
   ✓ threading.Lock for mutual exclusion
   ✓ 'with' statement for automatic acquire/release
   ✓ Prevents race conditions

5. RACE CONDITIONS
   ✓ Occur when threads access shared data without synchronization
   ✓ Results in lost updates
   ✓ Fixed with locks

6. SEMAPHORE
   ✓ Limits concurrent access to resources
   ✓ Like a lock but allows N concurrent users
   ✓ Good for resource pools (DB connections, etc.)

WHEN TO USE THREADING:
  ✓ I/O-bound operations (network, file I/O)
  ✓ Concurrent API calls
  ✓ GUI applications (keep UI responsive)

WHEN NOT TO USE THREADING:
  ✗ CPU-bound operations (use multiprocessing instead)
  ✗ GIL limits Python threading for CPU work

BEST PRACTICES:
  • Use ThreadPoolExecutor instead of raw threads
  • Always synchronize shared data access
  • Prefer queue.Queue for thread communication
  • Use 'with' statement for locks
  • Avoid global state when possible
  • Be careful with thread-unsafe libraries
""")


if __name__ == "__main__":
    print("THREADING DEMONSTRATIONS")
    print("=" * 70)

    demonstrate_basic_threading()
    time.sleep(0.5)

    demonstrate_thread_pool()
    time.sleep(0.5)

    print("\n" + "=" * 70)
    print("PRODUCER-CONSUMER PATTERN")
    print("=" * 70)
    pc = ProducerConsumer(max_queue_size=3)
    pc.run(num_producers=2, num_consumers=3, items_per_producer=5)

    time.sleep(0.5)
    demonstrate_locks()

    time.sleep(0.5)
    demonstrate_race_condition()

    time.sleep(0.5)
    demonstrate_semaphore()

    print_summary()
