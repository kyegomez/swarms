import asyncio
import concurrent.futures
import json
import os
import psutil
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from swarms.structs.agent import Agent
from loguru import logger


class AgentBenchmark:
    def __init__(
        self,
        num_iterations: int = 5,
        output_dir: str = "benchmark_results",
    ):
        self.num_iterations = num_iterations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Use process pool for CPU-bound tasks
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(os.cpu_count(), 4)
        )

        # Use thread pool for I/O-bound tasks
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(os.cpu_count() * 2, 8)
        )

        self.default_queries = [
            "Conduct an analysis of the best real undervalued ETFs",
            "What are the top performing tech stocks this quarter?",
            "Analyze current market trends in renewable energy sector",
            "Compare Bitcoin and Ethereum investment potential",
            "Evaluate the risk factors in emerging markets",
        ]

        self.agent = self._initialize_agent()
        self.process = psutil.Process()

        # Cache for storing repeated query results
        self._query_cache = {}

    def _initialize_agent(self) -> Agent:
        return Agent(
            agent_name="Financial-Analysis-Agent",
            agent_description="Personal finance advisor agent",
            # system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
            max_loops=1,
            model_name="gpt-4o-mini",
            dynamic_temperature_enabled=True,
            interactive=False,
        )

    def _get_system_metrics(self) -> Dict[str, float]:
        # Optimized system metrics collection
        return {
            "cpu_percent": self.process.cpu_percent(),
            "memory_mb": self.process.memory_info().rss / 1024 / 1024,
        }

    def _calculate_statistics(
        self, values: List[float]
    ) -> Dict[str, float]:
        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(sorted_values)
        mean_val = sum(values) / n

        stats = {
            "mean": mean_val,
            "median": sorted_values[n // 2],
            "min": sorted_values[0],
            "max": sorted_values[-1],
        }

        # Only calculate stdev if we have enough values
        if n > 1:
            stats["std_dev"] = (
                sum((x - mean_val) ** 2 for x in values) / n
            ) ** 0.5

        return {k: round(v, 3) for k, v in stats.items()}

    async def process_iteration(
        self, query: str, iteration: int
    ) -> Dict[str, Any]:
        """Process a single iteration of a query"""
        try:
            # Check cache for repeated queries
            cache_key = f"{query}_{iteration}"
            if cache_key in self._query_cache:
                return self._query_cache[cache_key]

            iteration_start = datetime.datetime.now()
            pre_metrics = self._get_system_metrics()

            # Run the agent
            try:
                self.agent.run(query)
                success = True
            except Exception as e:
                str(e)
                success = False

            execution_time = (
                datetime.datetime.now() - iteration_start
            ).total_seconds()
            post_metrics = self._get_system_metrics()

            result = {
                "execution_time": execution_time,
                "success": success,
                "pre_metrics": pre_metrics,
                "post_metrics": post_metrics,
                "iteration_data": {
                    "iteration": iteration + 1,
                    "execution_time": round(execution_time, 3),
                    "success": success,
                    "system_metrics": {
                        "pre": pre_metrics,
                        "post": post_metrics,
                    },
                },
            }

            # Cache the result
            self._query_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {e}")
            raise

    async def run_benchmark(
        self, queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run the benchmark asynchronously"""
        queries = queries or self.default_queries
        benchmark_data = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "num_iterations": self.num_iterations,
                "agent_config": {
                    "model_name": self.agent.model_name,
                    "max_loops": self.agent.max_loops,
                },
            },
            "results": {},
        }

        async def process_query(query: str):
            query_results = {
                "execution_times": [],
                "system_metrics": [],
                "iterations": [],
            }

            # Process iterations concurrently
            tasks = [
                self.process_iteration(query, i)
                for i in range(self.num_iterations)
            ]
            iteration_results = await asyncio.gather(*tasks)

            for result in iteration_results:
                query_results["execution_times"].append(
                    result["execution_time"]
                )
                query_results["system_metrics"].append(
                    result["post_metrics"]
                )
                query_results["iterations"].append(
                    result["iteration_data"]
                )

            # Calculate statistics
            query_results["statistics"] = {
                "execution_time": self._calculate_statistics(
                    query_results["execution_times"]
                ),
                "memory_usage": self._calculate_statistics(
                    [
                        m["memory_mb"]
                        for m in query_results["system_metrics"]
                    ]
                ),
                "cpu_usage": self._calculate_statistics(
                    [
                        m["cpu_percent"]
                        for m in query_results["system_metrics"]
                    ]
                ),
            }

            return query, query_results

        # Execute all queries concurrently
        query_tasks = [process_query(query) for query in queries]
        query_results = await asyncio.gather(*query_tasks)

        for query, results in query_results:
            benchmark_data["results"][query] = results

        return benchmark_data

    def save_results(self, benchmark_data: Dict[str, Any]) -> str:
        """Save benchmark results efficiently"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            self.output_dir / f"benchmark_results_{timestamp}.json"
        )

        # Write results in a single operation
        with open(filename, "w") as f:
            json.dump(benchmark_data, f, indent=2)

        logger.info(f"Benchmark results saved to: {filename}")
        return str(filename)

    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of the benchmark results"""
        print("\n=== Benchmark Summary ===")
        for query, data in results["results"].items():
            print(f"\nQuery: {query[:50]}...")
            stats = data["statistics"]["execution_time"]
            print(f"Average time: {stats['mean']:.2f}s")
            print(
                f"Memory usage (avg): {data['statistics']['memory_usage']['mean']:.1f}MB"
            )
            print(
                f"CPU usage (avg): {data['statistics']['cpu_usage']['mean']:.1f}%"
            )

    async def run_with_timeout(
        self, timeout: int = 300
    ) -> Dict[str, Any]:
        """Run benchmark with timeout"""
        try:
            return await asyncio.wait_for(
                self.run_benchmark(), timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Benchmark timed out after {timeout} seconds"
            )
            raise

    def cleanup(self):
        """Cleanup resources"""
        self.process_pool.shutdown()
        self.thread_pool.shutdown()
        self._query_cache.clear()


async def main():
    try:
        # Create and run benchmark
        benchmark = AgentBenchmark(num_iterations=1)

        # Run benchmark with timeout
        results = await benchmark.run_with_timeout(timeout=300)

        # Save results
        benchmark.save_results(results)

        # Print summary
        benchmark.print_summary(results)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
    finally:
        # Cleanup resources
        benchmark.cleanup()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
