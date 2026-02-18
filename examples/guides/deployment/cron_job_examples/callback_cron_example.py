"""
Callback CronJob Example

This example demonstrates how to use the new callback functionality in CronJob
to customize output processing while the cron job is still running.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict
from loguru import logger

from swarms import Agent, CronJob


def create_sample_agent():
    """Create a sample agent for demonstration."""
    return Agent(
        agent_name="Sample-Analysis-Agent",
        system_prompt="""You are a data analysis agent. Analyze the given data and provide insights.
        Keep your responses concise and focused on key findings.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        print_on=False,
    )


# Example 1: Simple output transformation callback
def transform_output_callback(
    output: Any, task: str, metadata: Dict
) -> Dict:
    """Transform the agent output into a structured format.

    Args:
        output: The original output from the agent
        task: The task that was executed
        metadata: Job metadata including execution count, timestamp, etc.

    Returns:
        Dict: Transformed output with additional metadata
    """
    return {
        "original_output": output,
        "transformed_at": datetime.fromtimestamp(
            metadata["timestamp"]
        ).isoformat(),
        "execution_number": metadata["execution_count"],
        "task_executed": task,
        "job_status": (
            "running" if metadata["is_running"] else "stopped"
        ),
        "uptime_seconds": (
            metadata["uptime"] if metadata["start_time"] else 0
        ),
    }


# Example 2: Output filtering and enhancement callback
def filter_and_enhance_callback(
    output: Any, task: str, metadata: Dict
) -> Dict:
    """Filter and enhance the output based on execution count and content.

    Args:
        output: The original output from the agent
        task: The task that was executed
        metadata: Job metadata

    Returns:
        Dict: Filtered and enhanced output
    """
    # Only include outputs that contain certain keywords
    if isinstance(output, str):
        if any(
            keyword in output.lower()
            for keyword in [
                "important",
                "key",
                "significant",
                "trend",
            ]
        ):
            enhanced_output = {
                "content": output,
                "priority": "high",
                "execution_id": metadata["execution_count"],
                "timestamp": metadata["timestamp"],
                "analysis_type": "priority_content",
            }
        else:
            enhanced_output = {
                "content": output,
                "priority": "normal",
                "execution_id": metadata["execution_count"],
                "timestamp": metadata["timestamp"],
                "analysis_type": "standard_content",
            }
    else:
        enhanced_output = {
            "content": str(output),
            "priority": "unknown",
            "execution_id": metadata["execution_count"],
            "timestamp": metadata["timestamp"],
            "analysis_type": "non_string_content",
        }

    return enhanced_output


# Example 3: Real-time monitoring callback
class MonitoringCallback:
    """Callback class that provides real-time monitoring capabilities."""

    def __init__(self):
        self.output_history = []
        self.error_count = 0
        self.success_count = 0
        self.last_execution_time = None

    def __call__(
        self, output: Any, task: str, metadata: Dict
    ) -> Dict:
        """Monitor and track execution metrics.

        Args:
            output: The original output from the agent
            task: The task that was executed
            metadata: Job metadata

        Returns:
            Dict: Output with monitoring information
        """
        current_time = time.time()

        # Calculate execution time
        if self.last_execution_time:
            execution_time = current_time - self.last_execution_time
        else:
            execution_time = 0

        self.last_execution_time = current_time

        # Track success/error
        if output and output != "Error":
            self.success_count += 1
            status = "success"
        else:
            self.error_count += 1
            status = "error"

        # Store in history (keep last 100)
        monitoring_data = {
            "output": output,
            "status": status,
            "execution_time": execution_time,
            "execution_count": metadata["execution_count"],
            "timestamp": metadata["timestamp"],
            "task": task,
            "metrics": {
                "success_rate": self.success_count
                / (self.success_count + self.error_count),
                "total_executions": self.success_count
                + self.error_count,
                "error_count": self.error_count,
                "success_count": self.success_count,
            },
        }

        self.output_history.append(monitoring_data)
        if len(self.output_history) > 100:
            self.output_history.pop(0)

        return monitoring_data

    def get_summary(self) -> Dict:
        """Get monitoring summary."""
        return {
            "total_executions": self.success_count + self.error_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": (
                self.success_count
                / (self.success_count + self.error_count)
                if (self.success_count + self.error_count) > 0
                else 0
            ),
            "history_length": len(self.output_history),
            "last_execution_time": self.last_execution_time,
        }


# Example 4: API integration callback
def api_webhook_callback(
    output: Any, task: str, metadata: Dict
) -> Dict:
    """Callback that could send output to an external API.

    Args:
        output: The original output from the agent
        task: The task that was executed
        metadata: Job metadata

    Returns:
        Dict: Output with API integration metadata
    """
    # In a real implementation, you would send this to your API
    api_payload = {
        "data": output,
        "source": "cron_job",
        "job_id": metadata["job_id"],
        "execution_id": metadata["execution_count"],
        "timestamp": metadata["timestamp"],
        "task": task,
    }

    # Simulate API call (replace with actual HTTP request)
    logger.info(
        f"Would send to API: {json.dumps(api_payload, indent=2)}"
    )

    return {
        "output": output,
        "api_status": "sent",
        "api_payload": api_payload,
        "execution_id": metadata["execution_count"],
    }


def main():
    """Demonstrate different callback usage patterns."""
    logger.info("🚀 Starting Callback CronJob Examples")

    # Create the agent
    agent = create_sample_agent()

    # Example 1: Simple transformation callback
    logger.info("📝 Example 1: Simple Output Transformation")
    transform_cron = CronJob(
        agent=agent,
        interval="15seconds",
        job_id="transform-example",
        callback=transform_output_callback,
    )

    # Example 2: Filtering and enhancement callback
    logger.info("🔍 Example 2: Output Filtering and Enhancement")
    filter_cron = CronJob(
        agent=agent,
        interval="20seconds",
        job_id="filter-example",
        callback=filter_and_enhance_callback,
    )

    # Example 3: Monitoring callback
    logger.info("📊 Example 3: Real-time Monitoring")
    monitoring_callback = MonitoringCallback()
    monitoring_cron = CronJob(
        agent=agent,
        interval="25seconds",
        job_id="monitoring-example",
        callback=monitoring_callback,
    )

    # Example 4: API integration callback
    logger.info("🌐 Example 4: API Integration")
    api_cron = CronJob(
        agent=agent,
        interval="30seconds",
        job_id="api-example",
        callback=api_webhook_callback,
    )

    # Start all cron jobs
    logger.info("▶️  Starting all cron jobs...")

    # Start them in separate threads to run concurrently
    import threading

    def run_cron(cron_job, task):
        try:
            cron_job.run(task=task)
        except KeyboardInterrupt:
            cron_job.stop()

    # Start each cron job in its own thread
    threads = []
    tasks = [
        "Analyze the current market trends and provide key insights",
        "What are the most important factors affecting today's economy?",
        "Provide a summary of recent technological developments",
        "Analyze the impact of current events on business operations",
    ]

    for i, (cron_job, task) in enumerate(
        [
            (transform_cron, tasks[0]),
            (filter_cron, tasks[1]),
            (monitoring_cron, tasks[2]),
            (api_cron, tasks[3]),
        ]
    ):
        thread = threading.Thread(
            target=run_cron,
            args=(cron_job, task),
            daemon=True,
            name=f"cron-thread-{i}",
        )
        thread.start()
        threads.append(thread)

    logger.info("✅ All cron jobs started successfully!")
    logger.info("📊 Press Ctrl+C to stop and see monitoring summary")

    try:
        # Let them run for a while
        time.sleep(120)  # Run for 2 minutes

        # Show monitoring summary
        logger.info("📈 Monitoring Summary:")
        logger.info(
            json.dumps(monitoring_callback.get_summary(), indent=2)
        )

        # Show execution stats for each cron job
        for cron_job, name in [
            (transform_cron, "Transform"),
            (filter_cron, "Filter"),
            (monitoring_cron, "Monitoring"),
            (api_cron, "API"),
        ]:
            stats = cron_job.get_execution_stats()
            logger.info(
                f"{name} Cron Stats: {json.dumps(stats, indent=2)}"
            )

    except KeyboardInterrupt:
        logger.info("⏹️  Stopping all cron jobs...")

        # Stop all cron jobs
        for cron_job in [
            transform_cron,
            filter_cron,
            monitoring_cron,
            api_cron,
        ]:
            cron_job.stop()

        # Show final monitoring summary
        logger.info("📊 Final Monitoring Summary:")
        logger.info(
            json.dumps(monitoring_callback.get_summary(), indent=2)
        )


if __name__ == "__main__":
    main()
