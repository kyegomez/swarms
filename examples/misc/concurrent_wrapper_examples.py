"""
Examples demonstrating the concurrent wrapper decorator functionality.

This file shows how to use the concurrent and concurrent_class_executor
decorators to enable concurrent execution of functions and class methods.
"""

import time
import asyncio
from typing import Dict, Any
import requests

from swarms.utils.concurrent_wrapper import (
    concurrent,
    concurrent_class_executor,
    thread_executor,
    process_executor,
    async_executor,
    batch_executor,
    ExecutorType,
)


# Example 1: Basic concurrent function execution
@concurrent(
    name="data_processor",
    description="Process data concurrently",
    max_workers=4,
    timeout=30,
    retry_on_failure=True,
    max_retries=2,
)
def process_data(data: str) -> str:
    """Simulate data processing with a delay."""
    time.sleep(1)  # Simulate work
    return f"processed_{data}"


# Example 2: Thread-based executor for I/O bound tasks
@thread_executor(max_workers=8, timeout=60)
def fetch_url(url: str) -> Dict[str, Any]:
    """Fetch data from a URL."""
    try:
        response = requests.get(url, timeout=10)
        return {
            "url": url,
            "status_code": response.status_code,
            "content_length": len(response.content),
            "success": response.status_code == 200,
        }
    except Exception as e:
        return {"url": url, "error": str(e), "success": False}


# Example 3: Process-based executor for CPU-intensive tasks
@process_executor(max_workers=2, timeout=120)
def cpu_intensive_task(n: int) -> float:
    """Perform CPU-intensive computation."""
    result = 0.0
    for i in range(n):
        result += (i**0.5) * (i**0.3)
    return result


# Example 4: Async executor for async functions
@async_executor(max_workers=5)
async def async_task(task_id: int) -> str:
    """Simulate an async task."""
    await asyncio.sleep(0.5)  # Simulate async work
    return f"async_result_{task_id}"


# Example 5: Batch processing
@batch_executor(batch_size=10, max_workers=3)
def process_item(item: str) -> str:
    """Process a single item."""
    time.sleep(0.1)  # Simulate work
    return item.upper()


# Example 6: Class with concurrent methods
@concurrent_class_executor(
    name="DataProcessor",
    max_workers=4,
    methods=["process_batch", "validate_data"],
)
class DataProcessor:
    """A class with concurrent processing capabilities."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def process_batch(self, data: str) -> str:
        """Process a batch of data."""
        time.sleep(0.5)  # Simulate processing
        return f"processed_{data}"

    def validate_data(self, data: str) -> bool:
        """Validate data."""
        time.sleep(0.2)  # Simulate validation
        return len(data) > 0

    def normal_method(self, x: int) -> int:
        """A normal method (not concurrent)."""
        return x * 2


# Example 7: Function with custom configuration
@concurrent(
    name="custom_processor",
    description="Custom concurrent processor",
    max_workers=6,
    timeout=45,
    executor_type=ExecutorType.THREAD,
    return_exceptions=True,
    ordered=False,
    retry_on_failure=True,
    max_retries=3,
    retry_delay=0.5,
)
def custom_processor(item: str, multiplier: int = 1) -> str:
    """Custom processor with parameters."""
    time.sleep(0.3)
    return f"{item}_{multiplier}" * multiplier


def example_1_basic_concurrent_execution():
    """Example 1: Basic concurrent execution."""
    print("=== Example 1: Basic Concurrent Execution ===")

    # Prepare data
    data_items = [f"item_{i}" for i in range(10)]

    # Execute concurrently
    results = process_data.concurrent_execute(*data_items)

    # Process results
    successful_results = [r.value for r in results if r.success]
    failed_results = [r for r in results if not r.success]

    print(f"Successfully processed: {len(successful_results)} items")
    print(f"Failed: {len(failed_results)} items")
    print(f"Sample results: {successful_results[:3]}")
    print()


def example_2_thread_based_execution():
    """Example 2: Thread-based execution for I/O bound tasks."""
    print("=== Example 2: Thread-based Execution ===")

    # URLs to fetch
    urls = [
        "https://httpbin.org/get",
        "https://httpbin.org/status/200",
        "https://httpbin.org/status/404",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
    ]

    # Execute concurrently
    results = fetch_url.concurrent_execute(*urls)

    # Process results
    successful_fetches = [
        r.value
        for r in results
        if r.success and r.value.get("success")
    ]
    failed_fetches = [
        r.value
        for r in results
        if r.success and not r.value.get("success")
    ]

    print(f"Successful fetches: {len(successful_fetches)}")
    print(f"Failed fetches: {len(failed_fetches)}")
    print(
        f"Sample successful result: {successful_fetches[0] if successful_fetches else 'None'}"
    )
    print()


def example_3_process_based_execution():
    """Example 3: Process-based execution for CPU-intensive tasks."""
    print("=== Example 3: Process-based Execution ===")

    # CPU-intensive tasks
    tasks = [100000, 200000, 300000, 400000]

    # Execute concurrently
    results = cpu_intensive_task.concurrent_execute(*tasks)

    # Process results
    successful_results = [r.value for r in results if r.success]
    execution_times = [r.execution_time for r in results if r.success]

    print(f"Completed {len(successful_results)} CPU-intensive tasks")
    print(
        f"Average execution time: {sum(execution_times) / len(execution_times):.3f}s"
    )
    print(
        f"Sample result: {successful_results[0] if successful_results else 'None'}"
    )
    print()


def example_4_batch_processing():
    """Example 4: Batch processing."""
    print("=== Example 4: Batch Processing ===")

    # Items to process
    items = [f"item_{i}" for i in range(25)]

    # Process in batches
    results = process_item.concurrent_batch(items, batch_size=5)

    # Process results
    successful_results = [r.value for r in results if r.success]

    print(f"Processed {len(successful_results)} items in batches")
    print(f"Sample results: {successful_results[:5]}")
    print()


def example_5_class_concurrent_execution():
    """Example 5: Class with concurrent methods."""
    print("=== Example 5: Class Concurrent Execution ===")

    # Create processor instance
    processor = DataProcessor({"batch_size": 10})

    # Prepare data
    data_items = [f"data_{i}" for i in range(8)]

    # Execute concurrent methods
    process_results = processor.process_batch.concurrent_execute(
        *data_items
    )
    validate_results = processor.validate_data.concurrent_execute(
        *data_items
    )

    # Process results
    processed_items = [r.value for r in process_results if r.success]
    valid_items = [r.value for r in validate_results if r.success]

    print(f"Processed {len(processed_items)} items")
    print(f"Validated {len(valid_items)} items")
    print(f"Sample processed: {processed_items[:3]}")
    print(f"Sample validation: {valid_items[:3]}")
    print()


def example_6_custom_configuration():
    """Example 6: Custom configuration with exceptions and retries."""
    print("=== Example 6: Custom Configuration ===")

    # Items with different multipliers
    items = [f"item_{i}" for i in range(6)]
    multipliers = [1, 2, 3, 1, 2, 3]

    # Execute with custom configuration
    results = custom_processor.concurrent_execute(
        *items, **{"multiplier": multipliers}
    )

    # Process results
    successful_results = [r.value for r in results if r.success]
    failed_results = [r for r in results if not r.success]

    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    print(f"Sample results: {successful_results[:3]}")
    print()


def example_7_concurrent_mapping():
    """Example 7: Concurrent mapping over a list."""
    print("=== Example 7: Concurrent Mapping ===")

    # Items to map over
    items = [f"map_item_{i}" for i in range(15)]

    # Map function over items
    results = process_data.concurrent_map(items)

    # Process results
    mapped_results = [r.value for r in results if r.success]

    print(f"Mapped over {len(mapped_results)} items")
    print(f"Sample mapped results: {mapped_results[:5]}")
    print()


def example_8_error_handling():
    """Example 8: Error handling and retries."""
    print("=== Example 8: Error Handling ===")

    @concurrent(
        max_workers=3,
        return_exceptions=True,
        retry_on_failure=True,
        max_retries=2,
    )
    def unreliable_function(x: int) -> int:
        """A function that sometimes fails."""
        if x % 3 == 0:
            raise ValueError(f"Failed for {x}")
        time.sleep(0.1)
        return x * 2

    # Execute with potential failures
    results = unreliable_function.concurrent_execute(*range(10))

    # Process results
    successful_results = [r.value for r in results if r.success]
    failed_results = [r.exception for r in results if not r.success]

    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    print(f"Sample successful: {successful_results[:3]}")
    print(
        f"Sample failures: {[type(e).__name__ for e in failed_results[:3]]}"
    )
    print()


def main():
    """Run all examples."""
    print("Concurrent Wrapper Examples")
    print("=" * 50)
    print()

    try:
        example_1_basic_concurrent_execution()
        example_2_thread_based_execution()
        example_3_process_based_execution()
        example_4_batch_processing()
        example_5_class_concurrent_execution()
        example_6_custom_configuration()
        example_7_concurrent_mapping()
        example_8_error_handling()

        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
