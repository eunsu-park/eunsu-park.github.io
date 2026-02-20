"""
Asyncio Demonstration

Covers:
1. Basic async/await
2. Concurrent task execution
3. Gathering results
4. Semaphores for rate limiting
5. Error handling in async code
6. Async context managers
7. Practical examples (HTTP-like requests, data processing)
"""

import asyncio
import time
import random
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# 1. BASIC ASYNC/AWAIT
# =============================================================================

print("=" * 70)
print("1. BASIC ASYNC/AWAIT")
print("=" * 70)


async def simple_coroutine(name: str, delay: float) -> str:
    """Simple async function"""
    print(f"[{name}] Starting (delay: {delay:.2f}s)...")
    await asyncio.sleep(delay)  # Non-blocking sleep
    print(f"[{name}] Finished!")
    return f"{name} completed"


async def demonstrate_basic_async():
    """Basic async/await demonstration"""
    print("\n[BASIC ASYNC]")
    print("-" * 50)

    # Sequential execution (still async, but one at a time)
    print("Sequential:")
    start = time.time()
    result1 = await simple_coroutine("Task-1", 0.3)
    result2 = await simple_coroutine("Task-2", 0.2)
    print(f"Results: {result1}, {result2}")
    print(f"Time: {time.time() - start:.2f}s\n")

    # Concurrent execution
    print("Concurrent:")
    start = time.time()
    results = await asyncio.gather(
        simple_coroutine("Task-A", 0.3),
        simple_coroutine("Task-B", 0.2),
        simple_coroutine("Task-C", 0.25)
    )
    print(f"Results: {results}")
    print(f"Time: {time.time() - start:.2f}s (much faster!)")


# =============================================================================
# 2. SIMULATED HTTP REQUESTS
# =============================================================================

print("\n" + "=" * 70)
print("2. SIMULATED HTTP REQUESTS")
print("=" * 70)


class ResponseStatus(Enum):
    """HTTP response status"""
    SUCCESS = 200
    NOT_FOUND = 404
    ERROR = 500


@dataclass
class Response:
    """Simulated HTTP response"""
    url: str
    status: ResponseStatus
    data: Any
    duration: float


async def fetch_url(url: str, delay: float = None) -> Response:
    """
    Simulate fetching a URL.
    In real code, use aiohttp library.
    """
    if delay is None:
        delay = random.uniform(0.1, 0.5)

    print(f"Fetching {url}...")
    start = time.time()

    await asyncio.sleep(delay)  # Simulate network delay

    # Simulate occasional errors
    if random.random() < 0.1:
        status = ResponseStatus.ERROR
        data = {"error": "Server error"}
    else:
        status = ResponseStatus.SUCCESS
        data = {"url": url, "content": f"Content from {url}"}

    duration = time.time() - start
    print(f"Fetched {url} in {duration:.2f}s")

    return Response(url, status, data, duration)


async def demonstrate_concurrent_requests():
    """Demonstrate concurrent HTTP-like requests"""
    print("\n[CONCURRENT REQUESTS]")
    print("-" * 50)

    urls = [
        "https://api.example.com/users",
        "https://api.example.com/posts",
        "https://api.example.com/comments",
        "https://api.example.com/photos",
        "https://api.example.com/todos",
    ]

    # Fetch all URLs concurrently
    start = time.time()
    responses = await asyncio.gather(*[fetch_url(url) for url in urls])
    total_time = time.time() - start

    # Process results
    successful = [r for r in responses if r.status == ResponseStatus.SUCCESS]
    failed = [r for r in responses if r.status == ResponseStatus.ERROR]

    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    # If we did this synchronously:
    total_delay = sum(r.duration for r in responses)
    print(f"Sequential would take: {total_delay:.2f}s")
    print(f"Speedup: {total_delay / total_time:.2f}x")


# =============================================================================
# 3. RATE LIMITING WITH SEMAPHORE
# =============================================================================

print("\n" + "=" * 70)
print("3. RATE LIMITING WITH SEMAPHORE")
print("=" * 70)


async def rate_limited_fetch(url: str, semaphore: asyncio.Semaphore) -> Response:
    """Fetch URL with rate limiting"""
    async with semaphore:  # Only N concurrent requests
        return await fetch_url(url)


async def demonstrate_rate_limiting():
    """Demonstrate rate limiting with semaphore"""
    print("\n[RATE LIMITING]")
    print("-" * 50)

    urls = [f"https://api.example.com/item/{i}" for i in range(10)]

    # Limit to 3 concurrent requests
    semaphore = asyncio.Semaphore(3)

    print(f"Fetching {len(urls)} URLs with max 3 concurrent requests...")
    start = time.time()

    tasks = [rate_limited_fetch(url, semaphore) for url in urls]
    responses = await asyncio.gather(*tasks)

    total_time = time.time() - start
    print(f"\nCompleted in {total_time:.2f}s")
    print(f"Average per request: {total_time / len(urls):.2f}s")


# =============================================================================
# 4. ERROR HANDLING
# =============================================================================

print("\n" + "=" * 70)
print("4. ERROR HANDLING")
print("=" * 70)


async def risky_operation(task_id: int) -> dict:
    """Operation that might fail"""
    await asyncio.sleep(random.uniform(0.1, 0.3))

    # 30% chance of failure
    if random.random() < 0.3:
        raise ValueError(f"Task {task_id} failed!")

    return {"task_id": task_id, "result": task_id * 2}


async def demonstrate_error_handling():
    """Demonstrate error handling in async code"""
    print("\n[ERROR HANDLING]")
    print("-" * 50)

    # Approach 1: gather with return_exceptions=True
    print("Approach 1: gather with return_exceptions")
    tasks = [risky_operation(i) for i in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i}: Error - {result}")
        else:
            print(f"Task {i}: Success - {result}")

    # Approach 2: Individual try/except
    print("\nApproach 2: Individual try/except wrappers")

    async def safe_operation(task_id: int) -> dict:
        try:
            return await risky_operation(task_id)
        except Exception as e:
            return {"task_id": task_id, "error": str(e)}

    tasks = [safe_operation(i) for i in range(5)]
    results = await asyncio.gather(*tasks)

    for result in results:
        if "error" in result:
            print(f"Task {result['task_id']}: Error - {result['error']}")
        else:
            print(f"Task {result['task_id']}: Success - {result['result']}")


# =============================================================================
# 5. ASYNC CONTEXT MANAGERS
# =============================================================================

print("\n" + "=" * 70)
print("5. ASYNC CONTEXT MANAGERS")
print("=" * 70)


class AsyncDatabaseConnection:
    """Simulated async database connection"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connected = False

    async def __aenter__(self):
        """Async context manager entry"""
        print(f"Connecting to {self.connection_string}...")
        await asyncio.sleep(0.1)  # Simulate connection delay
        self.connected = True
        print("Connected!")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        print("Disconnecting...")
        await asyncio.sleep(0.05)  # Simulate disconnection delay
        self.connected = False
        print("Disconnected!")
        return False

    async def query(self, sql: str) -> List[dict]:
        """Simulate async query"""
        if not self.connected:
            raise RuntimeError("Not connected")

        print(f"Executing: {sql}")
        await asyncio.sleep(0.1)  # Simulate query execution
        return [{"id": 1, "name": "Result"}]


async def demonstrate_async_context_manager():
    """Demonstrate async context manager"""
    print("\n[ASYNC CONTEXT MANAGER]")
    print("-" * 50)

    async with AsyncDatabaseConnection("postgresql://localhost/mydb") as db:
        results = await db.query("SELECT * FROM users")
        print(f"Query results: {results}")
    # Connection automatically closed


# =============================================================================
# 6. PRACTICAL EXAMPLE: DATA PROCESSING PIPELINE
# =============================================================================

print("\n" + "=" * 70)
print("6. PRACTICAL EXAMPLE: DATA PROCESSING PIPELINE")
print("=" * 70)


async def fetch_data(source_id: int) -> List[dict]:
    """Fetch data from source"""
    await asyncio.sleep(random.uniform(0.1, 0.3))
    return [
        {"source": source_id, "value": random.randint(1, 100)}
        for _ in range(random.randint(5, 10))
    ]


async def process_item(item: dict) -> dict:
    """Process a single item"""
    await asyncio.sleep(0.01)  # Simulate processing
    return {
        **item,
        "processed": True,
        "doubled": item["value"] * 2
    }


async def save_results(results: List[dict]) -> None:
    """Save processed results"""
    await asyncio.sleep(0.1)  # Simulate save delay
    print(f"Saved {len(results)} items to database")


async def data_processing_pipeline():
    """Complete data processing pipeline"""
    print("\n[DATA PROCESSING PIPELINE]")
    print("-" * 50)

    # Step 1: Fetch data from multiple sources concurrently
    print("Step 1: Fetching data from sources...")
    source_ids = range(1, 6)
    fetch_tasks = [fetch_data(source_id) for source_id in source_ids]
    data_batches = await asyncio.gather(*fetch_tasks)

    # Flatten data
    all_data = [item for batch in data_batches for item in batch]
    print(f"Fetched {len(all_data)} items")

    # Step 2: Process items concurrently (with rate limiting)
    print("\nStep 2: Processing items...")
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent processing

    async def process_with_limit(item):
        async with semaphore:
            return await process_item(item)

    process_tasks = [process_with_limit(item) for item in all_data]
    processed_items = await asyncio.gather(*process_tasks)

    print(f"Processed {len(processed_items)} items")

    # Step 3: Save results
    print("\nStep 3: Saving results...")
    await save_results(processed_items)

    print("\nPipeline completed!")
    return processed_items


# =============================================================================
# 7. TASK MANAGEMENT
# =============================================================================

print("\n" + "=" * 70)
print("7. TASK MANAGEMENT")
print("=" * 70)


async def demonstrate_task_management():
    """Demonstrate task creation and management"""
    print("\n[TASK MANAGEMENT]")
    print("-" * 50)

    async def background_task(name: str, duration: float):
        """Background task"""
        print(f"[{name}] Started")
        await asyncio.sleep(duration)
        print(f"[{name}] Completed")
        return f"{name} result"

    # Create tasks
    task1 = asyncio.create_task(background_task("Task-1", 0.3))
    task2 = asyncio.create_task(background_task("Task-2", 0.2))
    task3 = asyncio.create_task(background_task("Task-3", 0.4))

    # Wait for specific task
    print("Waiting for Task-2...")
    result = await task2
    print(f"Task-2 result: {result}")

    # Wait for all tasks
    print("\nWaiting for all tasks...")
    results = await asyncio.gather(task1, task3)
    print(f"All results: {results}")

    # Cancel task example
    print("\nTask cancellation example:")
    long_task = asyncio.create_task(background_task("Long-Task", 5.0))
    await asyncio.sleep(0.1)
    long_task.cancel()

    try:
        await long_task
    except asyncio.CancelledError:
        print("Long-Task was cancelled")


# =============================================================================
# DEMONSTRATION
# =============================================================================

async def main():
    """Main demonstration function"""
    print("ASYNCIO DEMONSTRATIONS")
    print("=" * 70)

    await demonstrate_basic_async()
    await demonstrate_concurrent_requests()
    await demonstrate_rate_limiting()
    await demonstrate_error_handling()
    await demonstrate_async_context_manager()
    await data_processing_pipeline()
    await demonstrate_task_management()

    print_summary()


def print_summary():
    print("\n" + "=" * 70)
    print("ASYNCIO SUMMARY")
    print("=" * 70)

    print("""
1. BASIC ASYNC/AWAIT
   ✓ async def creates coroutine function
   ✓ await suspends execution
   ✓ Enables concurrency without threads

2. CONCURRENT EXECUTION
   ✓ asyncio.gather() runs multiple coroutines
   ✓ Much faster than sequential
   ✓ Great for I/O-bound operations

3. RATE LIMITING
   ✓ asyncio.Semaphore limits concurrency
   ✓ Prevents overwhelming resources
   ✓ async with for automatic release

4. ERROR HANDLING
   ✓ return_exceptions=True in gather
   ✓ Individual try/except in coroutines
   ✓ Handle errors without stopping all tasks

5. ASYNC CONTEXT MANAGERS
   ✓ async with for async resource management
   ✓ __aenter__ and __aexit__ methods
   ✓ Guaranteed cleanup

6. TASK MANAGEMENT
   ✓ asyncio.create_task() for background tasks
   ✓ Can wait for specific tasks
   ✓ Can cancel tasks

WHEN TO USE ASYNCIO:
  ✓ I/O-bound operations (HTTP, database, files)
  ✓ Many concurrent operations
  ✓ Network services
  ✓ Web scraping
  ✓ API integrations

ASYNC vs THREADING:
  Asyncio:
    ✓ Single-threaded (no race conditions!)
    ✓ Explicit concurrency points (await)
    ✓ Better for I/O-bound with many tasks
    ✓ Lower overhead

  Threading:
    ✓ True parallelism (on multiple cores)
    ✓ Better for blocking I/O (no async support)
    ✓ Implicit concurrency (harder to reason about)
    ✓ Need locks for shared data

BEST PRACTICES:
  • Use aiohttp for real HTTP requests
  • Always await async functions
  • Use semaphores for rate limiting
  • Handle errors appropriately
  • Use async context managers
  • Don't mix blocking and async code
  • Use asyncio.run() as entry point
  • Be careful with shared mutable state

COMMON PITFALLS:
  ✗ Forgetting await (coroutine never runs!)
  ✗ Using blocking I/O in async code
  ✗ Not handling exceptions
  ✗ Creating too many concurrent tasks
""")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
