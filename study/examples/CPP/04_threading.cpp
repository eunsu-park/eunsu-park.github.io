/*
 * Multithreading and Concurrency Demo
 *
 * Demonstrates:
 * - std::thread basics
 * - std::mutex, std::lock_guard, std::unique_lock
 * - std::async and std::future
 * - std::condition_variable
 * - Thread-safe queue example
 * - std::jthread (C++20)
 *
 * Compile: g++ -std=c++20 -Wall -Wextra -pthread 04_threading.cpp -o threading
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <vector>
#include <queue>
#include <chrono>
#include <string>
#include <numeric>

// ============ Basic Thread ============
void print_numbers(int id, int count) {
    for (int i = 0; i < count; i++) {
        std::cout << "Thread " << id << ": " << i << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void demo_basic_thread() {
    std::cout << "\n=== Basic std::thread ===\n";

    std::thread t1(print_numbers, 1, 3);
    std::thread t2(print_numbers, 2, 3);

    // Must join or detach
    t1.join();
    t2.join();

    std::cout << "All threads finished\n";
}

// ============ Mutex and Lock Guard ============
std::mutex cout_mutex;
int shared_counter = 0;

void increment_counter(int id, int iterations) {
    for (int i = 0; i < iterations; i++) {
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            shared_counter++;
            std::cout << "Thread " << id << " incremented counter to "
                      << shared_counter << "\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void demo_mutex() {
    std::cout << "\n=== Mutex and Lock Guard ===\n";

    shared_counter = 0;

    std::thread t1(increment_counter, 1, 5);
    std::thread t2(increment_counter, 2, 5);

    t1.join();
    t2.join();

    std::cout << "Final counter value: " << shared_counter << "\n";
}

// ============ std::async and std::future ============
int compute_sum(int n) {
    std::cout << "Computing sum(1.." << n << ")...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return n * (n + 1) / 2;
}

void demo_async() {
    std::cout << "\n=== std::async and std::future ===\n";

    // Launch async task
    std::future<int> result1 = std::async(std::launch::async, compute_sum, 100);
    std::future<int> result2 = std::async(std::launch::async, compute_sum, 200);

    std::cout << "Doing other work while tasks run...\n";

    // Get results (blocks if not ready)
    std::cout << "Result 1: " << result1.get() << "\n";
    std::cout << "Result 2: " << result2.get() << "\n";
}

// ============ Condition Variable ============
std::queue<int> data_queue;
std::mutex queue_mutex;
std::condition_variable data_cond;
bool finished = false;

void producer() {
    for (int i = 0; i < 5; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            data_queue.push(i);
            std::cout << "Produced: " << i << "\n";
        }

        data_cond.notify_one();
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        finished = true;
    }
    data_cond.notify_all();
}

void consumer(int id) {
    while (true) {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // Wait for data or finish signal
        data_cond.wait(lock, [] { return !data_queue.empty() || finished; });

        if (!data_queue.empty()) {
            int value = data_queue.front();
            data_queue.pop();
            lock.unlock();

            std::cout << "Consumer " << id << " consumed: " << value << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        } else if (finished) {
            break;
        }
    }
}

void demo_condition_variable() {
    std::cout << "\n=== Condition Variable (Producer-Consumer) ===\n";

    // Clear state
    while (!data_queue.empty()) data_queue.pop();
    finished = false;

    std::thread prod(producer);
    std::thread cons1(consumer, 1);
    std::thread cons2(consumer, 2);

    prod.join();
    cons1.join();
    cons2.join();

    std::cout << "Producer-consumer finished\n";
}

// ============ Thread-Safe Queue ============
template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;

public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        cond_.notify_one();
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty(); });
        value = std::move(queue_.front());
        queue_.pop();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

void demo_thread_safe_queue() {
    std::cout << "\n=== Thread-Safe Queue ===\n";

    ThreadSafeQueue<int> queue;

    auto producer_task = [&queue]() {
        for (int i = 0; i < 5; i++) {
            queue.push(i);
            std::cout << "Pushed: " << i << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    };

    auto consumer_task = [&queue](int id) {
        for (int i = 0; i < 3; i++) {
            int value;
            queue.wait_and_pop(value);
            std::cout << "Consumer " << id << " popped: " << value << "\n";
        }
    };

    std::thread prod(producer_task);
    std::thread cons1(consumer_task, 1);
    std::thread cons2(consumer_task, 2);

    prod.join();
    cons1.join();
    cons2.join();

    std::cout << "Thread-safe queue demo finished\n";
}

// ============ std::jthread (C++20) ============
void demo_jthread() {
    std::cout << "\n=== std::jthread (C++20 - Auto-joining) ===\n";

    {
        std::jthread t([](std::stop_token stoken) {
            int count = 0;
            while (!stoken.stop_requested() && count < 5) {
                std::cout << "jthread working... " << count << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                count++;
            }
            std::cout << "jthread stopping\n";
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(600));
        std::cout << "Requesting stop...\n";
        t.request_stop();

        // No need to call join() - automatic on destruction
    }

    std::cout << "jthread automatically joined\n";
}

// ============ Parallel Accumulation ============
template<typename Iterator, typename T>
T parallel_accumulate(Iterator first, Iterator last, T init) {
    unsigned long const length = std::distance(first, last);

    if (length == 0) return init;

    unsigned long const min_per_thread = 25;
    unsigned long const max_threads = (length + min_per_thread - 1) / min_per_thread;
    unsigned long const hardware_threads = std::thread::hardware_concurrency();
    unsigned long const num_threads = std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);
    unsigned long const block_size = length / num_threads;

    std::vector<std::future<T>> futures(num_threads - 1);

    Iterator block_start = first;
    for (unsigned long i = 0; i < (num_threads - 1); ++i) {
        Iterator block_end = block_start;
        std::advance(block_end, block_size);

        futures[i] = std::async(std::launch::async,
                                [block_start, block_end]() {
                                    return std::accumulate(block_start, block_end, T{});
                                });

        block_start = block_end;
    }

    T result = std::accumulate(block_start, last, init);

    for (auto& f : futures) {
        result += f.get();
    }

    return result;
}

void demo_parallel_accumulate() {
    std::cout << "\n=== Parallel Accumulation ===\n";

    std::vector<int> data(1000);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = i + 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    int result = parallel_accumulate(data.begin(), data.end(), 0);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Sum: " << result << "\n";
    std::cout << "Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " μs\n";
}

// ============ Main ============
int main() {
    std::cout << "Multithreading and Concurrency Demo\n";
    std::cout << "====================================\n";
    std::cout << "Hardware concurrency: " << std::thread::hardware_concurrency() << "\n";

    demo_basic_thread();
    demo_mutex();
    demo_async();
    demo_condition_variable();
    demo_thread_safe_queue();
    demo_jthread();
    demo_parallel_accumulate();

    std::cout << "\nAll demos completed!\n";
    return 0;
}
