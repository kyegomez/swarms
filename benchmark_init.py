from time import perf_counter_ns
import psutil
import os
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from statistics import mean, median, stdev, variance
from swarms.structs.agent import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)


def get_memory_stats(memory_readings):
    """Calculate memory statistics"""
    return {
        "peak": max(memory_readings),
        "min": min(memory_readings),
        "mean": mean(memory_readings),
        "median": median(memory_readings),
        "stdev": (
            stdev(memory_readings) if len(memory_readings) > 1 else 0
        ),
        "variance": (
            variance(memory_readings)
            if len(memory_readings) > 1
            else 0
        ),
    }


def get_time_stats(times):
    """Calculate time statistics"""
    return {
        "total": sum(times),
        "mean": mean(times),
        "median": median(times),
        "min": min(times),
        "max": max(times),
        "stdev": stdev(times) if len(times) > 1 else 0,
        "variance": variance(times) if len(times) > 1 else 0,
    }


def benchmark_multiple_agents(num_agents=100):
    console = Console()
    init_times = []
    memory_readings = []
    process = psutil.Process(os.getpid())

    # Create benchmark tables
    time_table = Table(title="Time Statistics")
    time_table.add_column("Metric", style="cyan")
    time_table.add_column("Value", style="green")

    memory_table = Table(title="Memory Statistics")
    memory_table.add_column("Metric", style="cyan")
    memory_table.add_column("Value", style="green")

    initial_memory = process.memory_info().rss / 1024
    start_total_time = perf_counter_ns()

    # Initialize agents and measure performance
    for i in range(num_agents):
        start_time = perf_counter_ns()

        Agent(
            agent_name=f"Financial-Analysis-Agent-{i}",
            agent_description="Personal finance advisor agent",
            system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
            max_loops=2,
            model_name="gpt-4o-mini",
            dynamic_temperature_enabled=True,
            interactive=False,
        )

        init_time = (perf_counter_ns() - start_time) / 1_000_000
        init_times.append(init_time)

        current_memory = process.memory_info().rss / 1024
        memory_readings.append(current_memory - initial_memory)

        if (i + 1) % 10 == 0:
            console.print(
                f"Created {i + 1} agents...", style="bold blue"
            )

    total_elapsed_time = (
        perf_counter_ns() - start_total_time
    ) / 1_000_000

    # Calculate statistics
    time_stats = get_time_stats(init_times)
    memory_stats = get_memory_stats(memory_readings)

    # Add time measurements
    time_table.add_row(
        "Total Wall Time", f"{total_elapsed_time:.2f} ms"
    )
    time_table.add_row(
        "Total Init Time", f"{time_stats['total']:.2f} ms"
    )
    time_table.add_row(
        "Average Init Time", f"{time_stats['mean']:.2f} ms"
    )
    time_table.add_row(
        "Median Init Time", f"{time_stats['median']:.2f} ms"
    )
    time_table.add_row("Fastest Init", f"{time_stats['min']:.2f} ms")
    time_table.add_row("Slowest Init", f"{time_stats['max']:.2f} ms")
    time_table.add_row(
        "Std Deviation", f"{time_stats['stdev']:.2f} ms"
    )
    time_table.add_row(
        "Variance", f"{time_stats['variance']:.4f} ms²"
    )
    time_table.add_row(
        "Throughput",
        f"{(num_agents/total_elapsed_time) * 1000:.2f} agents/second",
    )

    # Add memory measurements
    memory_table.add_row(
        "Peak Memory Usage", f"{memory_stats['peak']:.2f} KB"
    )
    memory_table.add_row(
        "Minimum Memory Usage", f"{memory_stats['min']:.2f} KB"
    )
    memory_table.add_row(
        "Average Memory Usage", f"{memory_stats['mean']:.2f} KB"
    )
    memory_table.add_row(
        "Median Memory Usage", f"{memory_stats['median']:.2f} KB"
    )
    memory_table.add_row(
        "Memory Std Deviation", f"{memory_stats['stdev']:.2f} KB"
    )
    memory_table.add_row(
        "Memory Variance", f"{memory_stats['variance']:.2f} KB²"
    )
    memory_table.add_row(
        "Avg Memory Per Agent",
        f"{memory_stats['mean']/num_agents:.2f} KB",
    )

    # Create and display panels
    time_panel = Panel(
        time_table,
        title="Time Benchmark Results",
        border_style="blue",
        padding=(1, 2),
    )

    memory_panel = Panel(
        memory_table,
        title="Memory Benchmark Results",
        border_style="green",
        padding=(1, 2),
    )

    console.print(time_panel)
    console.print("\n")
    console.print(memory_panel)


if __name__ == "__main__":
    benchmark_multiple_agents(1000)
