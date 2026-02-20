"""
Asynchronous Programming with asyncio

Demonstrates:
- async/await syntax
- asyncio.create_task
- asyncio.gather
- Concurrent execution
- Async generators
- asyncio.Queue
- Error handling in async code
"""

import asyncio
import time
from typing import List, AsyncIterator


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Basic async/await
# =============================================================================

section("Basic async/await")


async def simple_coroutine(name: str, delay: float) -> str:
    """Simple async function."""
    print(f"  {name}: Starting (delay={delay}s)")
    await asyncio.sleep(delay)
    print(f"  {name}: Finished")
    return f"Result from {name}"


async def basic_example():
    """Run basic async function."""
    result = await simple_coroutine("Task-A", 0.1)
    print(f"  Got result: {result}")


asyncio.run(basic_example())


# =============================================================================
# Concurrent Execution with create_task
# =============================================================================

section("Concurrent Execution with create_task")


async def fetch_data(task_id: int, delay: float) -> dict:
    """Simulate fetching data from API."""
    print(f"  Task-{task_id}: Starting fetch (delay={delay:.2f}s)")
    await asyncio.sleep(delay)
    print(f"  Task-{task_id}: Fetch complete")
    return {"id": task_id, "data": f"Data-{task_id}"}


async def concurrent_tasks():
    """Run multiple tasks concurrently."""
    start = time.perf_counter()

    # Create tasks
    task1 = asyncio.create_task(fetch_data(1, 0.2))
    task2 = asyncio.create_task(fetch_data(2, 0.15))
    task3 = asyncio.create_task(fetch_data(3, 0.1))

    # Wait for all tasks
    result1 = await task1
    result2 = await task2
    result3 = await task3

    elapsed = time.perf_counter() - start

    print(f"\n  Results: {[result1, result2, result3]}")
    print(f"  Total time: {elapsed:.2f}s (concurrent)")
    print(f"  Sequential would take: {0.2 + 0.15 + 0.1:.2f}s")


asyncio.run(concurrent_tasks())


# =============================================================================
# asyncio.gather
# =============================================================================

section("asyncio.gather - Run Multiple Coroutines")


async def download_file(file_id: int) -> str:
    """Simulate file download."""
    delay = 0.1 + (file_id % 3) * 0.05
    print(f"  Downloading file-{file_id}...")
    await asyncio.sleep(delay)
    return f"file-{file_id}.dat"


async def gather_example():
    """Use gather to run coroutines concurrently."""
    start = time.perf_counter()

    # gather runs all coroutines concurrently
    results = await asyncio.gather(
        download_file(1),
        download_file(2),
        download_file(3),
        download_file(4),
        download_file(5)
    )

    elapsed = time.perf_counter() - start

    print(f"\n  Downloaded: {results}")
    print(f"  Total time: {elapsed:.2f}s")


asyncio.run(gather_example())


# =============================================================================
# Error Handling
# =============================================================================

section("Error Handling in Async Code")


async def risky_operation(task_id: int) -> str:
    """Operation that might fail."""
    await asyncio.sleep(0.05)
    if task_id == 2:
        raise ValueError(f"Task-{task_id} failed!")
    return f"Success-{task_id}"


async def error_handling_example():
    """Handle errors in async code."""
    print("Without error handling:")
    try:
        results = await asyncio.gather(
            risky_operation(1),
            risky_operation(2),  # This will fail
            risky_operation(3)
        )
    except ValueError as e:
        print(f"  Caught exception: {e}")

    print("\nWith return_exceptions=True:")
    results = await asyncio.gather(
        risky_operation(1),
        risky_operation(2),  # This will fail
        risky_operation(3),
        return_exceptions=True
    )

    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"  Task-{i}: ERROR - {result}")
        else:
            print(f"  Task-{i}: {result}")


asyncio.run(error_handling_example())


# =============================================================================
# Async Generators
# =============================================================================

section("Async Generators")


async def async_range(n: int) -> AsyncIterator[int]:
    """Async generator yielding numbers."""
    for i in range(n):
        print(f"  Yielding {i}")
        await asyncio.sleep(0.05)
        yield i


async def fetch_pages(num_pages: int) -> AsyncIterator[dict]:
    """Simulate fetching pages from API."""
    for page in range(1, num_pages + 1):
        await asyncio.sleep(0.1)
        yield {
            "page": page,
            "data": [f"item-{page}-{i}" for i in range(3)]
        }


async def async_generator_example():
    """Use async generators."""
    print("Async range:")
    async for i in async_range(5):
        print(f"  Received: {i}")

    print("\nAsync page fetcher:")
    async for page_data in fetch_pages(3):
        print(f"  Page {page_data['page']}: {page_data['data']}")


asyncio.run(async_generator_example())


# =============================================================================
# asyncio.Queue - Producer/Consumer
# =============================================================================

section("asyncio.Queue - Producer/Consumer Pattern")


async def producer(queue: asyncio.Queue, producer_id: int, num_items: int):
    """Produce items and put them in queue."""
    for i in range(num_items):
        item = f"P{producer_id}-Item{i}"
        await asyncio.sleep(0.05)
        await queue.put(item)
        print(f"  Producer-{producer_id}: produced {item}")

    await queue.put(None)  # Sentinel value


async def consumer(queue: asyncio.Queue, consumer_id: int):
    """Consume items from queue."""
    while True:
        item = await queue.get()

        if item is None:
            queue.task_done()
            print(f"  Consumer-{consumer_id}: received sentinel, stopping")
            break

        print(f"  Consumer-{consumer_id}: processing {item}")
        await asyncio.sleep(0.08)
        queue.task_done()


async def producer_consumer_example():
    """Producer-consumer with asyncio.Queue."""
    queue = asyncio.Queue(maxsize=5)

    # Create producers and consumers
    producers = [
        asyncio.create_task(producer(queue, i, 3))
        for i in range(2)
    ]

    consumers = [
        asyncio.create_task(consumer(queue, i))
        for i in range(2)
    ]

    # Wait for producers to finish
    await asyncio.gather(*producers)

    # Wait for queue to be processed
    await queue.join()

    # Cancel consumers (they're waiting on empty queue)
    for c in consumers:
        c.cancel()


asyncio.run(producer_consumer_example())


# =============================================================================
# Timeouts
# =============================================================================

section("Timeouts with asyncio.wait_for")


async def slow_operation():
    """Slow operation that might timeout."""
    print("  Starting slow operation...")
    await asyncio.sleep(2.0)
    return "Operation complete"


async def timeout_example():
    """Demonstrate timeout handling."""
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=0.5)
        print(f"  Result: {result}")
    except asyncio.TimeoutError:
        print("  Operation timed out after 0.5s")


asyncio.run(timeout_example())


# =============================================================================
# Concurrent HTTP Requests (Simulated)
# =============================================================================

section("Simulated Concurrent HTTP Requests")


async def http_get(url: str) -> dict:
    """Simulate HTTP GET request."""
    # Simulate network latency
    delay = 0.1 + hash(url) % 10 / 100
    await asyncio.sleep(delay)

    return {
        "url": url,
        "status": 200,
        "data": f"Content from {url}",
        "time": delay
    }


async def fetch_all_urls(urls: List[str]) -> List[dict]:
    """Fetch all URLs concurrently."""
    tasks = [http_get(url) for url in urls]
    return await asyncio.gather(*tasks)


async def http_example():
    """Fetch multiple URLs concurrently."""
    urls = [
        "https://example.com/api/users",
        "https://example.com/api/posts",
        "https://example.com/api/comments",
        "https://example.com/api/photos",
    ]

    start = time.perf_counter()
    results = await fetch_all_urls(urls)
    elapsed = time.perf_counter() - start

    print("  Fetch results:")
    for result in results:
        print(f"    {result['url']}: {result['status']} ({result['time']:.3f}s)")

    print(f"\n  Total time: {elapsed:.2f}s (concurrent)")
    print(f"  Sequential would take: {sum(r['time'] for r in results):.2f}s")


asyncio.run(http_example())


# =============================================================================
# Running Synchronous Code
# =============================================================================

section("Running Synchronous Code with run_in_executor")


def blocking_io_operation(n: int) -> str:
    """Blocking I/O operation (sync function)."""
    print(f"  Blocking operation {n} starting...")
    time.sleep(0.1)
    return f"Result-{n}"


async def run_blocking_code():
    """Run blocking code in executor."""
    loop = asyncio.get_event_loop()

    # Run blocking calls in thread pool
    results = await asyncio.gather(
        loop.run_in_executor(None, blocking_io_operation, 1),
        loop.run_in_executor(None, blocking_io_operation, 2),
        loop.run_in_executor(None, blocking_io_operation, 3)
    )

    print(f"  Results: {results}")


asyncio.run(run_blocking_code())


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Asyncio patterns covered:
1. async/await - define and await coroutines
2. asyncio.create_task - schedule concurrent execution
3. asyncio.gather - run multiple coroutines, collect results
4. Error handling - try/except and return_exceptions
5. Async generators - async for and yield
6. asyncio.Queue - producer/consumer pattern
7. Timeouts - asyncio.wait_for
8. run_in_executor - run blocking code without blocking event loop

Benefits of asyncio:
- Efficient I/O-bound concurrency
- Single-threaded (no GIL issues)
- Clean async/await syntax
- Better resource utilization than threads for I/O

Use asyncio for:
- Network requests (HTTP, websockets)
- Database queries
- File I/O
- Any I/O-bound operations

Don't use for CPU-bound tasks (use multiprocessing instead).
""")
