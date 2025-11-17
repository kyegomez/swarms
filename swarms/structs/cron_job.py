import threading
import time
import traceback
from typing import Any, Callable, List, Optional, Union

import schedule
from loguru import logger


class CronJobError(Exception):
    """Base exception class for CronJob errors."""

    pass


class CronJobConfigError(CronJobError):
    """Exception raised for configuration errors in CronJob."""

    pass


class CronJobScheduleError(CronJobError):
    """Exception raised for scheduling related errors in CronJob."""

    pass


class CronJobExecutionError(CronJobError):
    """Exception raised for execution related errors in CronJob."""

    pass


class CronJob:
    """A wrapper class that turns any callable (including Swarms agents) into a scheduled cron job.

    This class provides functionality to schedule and run tasks at specified intervals using
    the schedule library with cron-style scheduling.

    Attributes:
        agent: The Swarms Agent instance or callable to be scheduled
        interval: The interval string (e.g., "5seconds", "10minutes", "1hour")
        job_id: Unique identifier for the job
        is_running: Flag indicating if the job is currently running
        thread: Thread object for running the job
        callback: Optional callback function to customize output processing
    """

    def __init__(
        self,
        agent: Optional[Union[Any, Callable]] = None,
        interval: Optional[str] = None,
        job_id: Optional[str] = None,
        callback: Optional[Callable[[Any, str, dict], Any]] = None,
    ):
        """Initialize the CronJob wrapper.

        Args:
            agent: The Swarms Agent instance or callable to be scheduled
            interval: The interval string (e.g., "5seconds", "10minutes", "1hour")
            job_id: Optional unique identifier for the job. If not provided, one will be generated.
            callback: Optional callback function to customize output processing.
                     Signature: callback(output: Any, task: str, metadata: dict) -> Any
                     - output: The original output from the agent
                     - task: The task that was executed
                     - metadata: Dictionary containing job_id, timestamp, execution_count, etc.
                     Returns: The customized output

        Raises:
            CronJobConfigError: If the interval format is invalid
        """
        self.agent = agent
        self.interval = interval
        self.job_id = job_id or f"job_{id(self)}"
        self.is_running = False
        self.thread = None
        self.schedule = schedule.Scheduler()
        self.callback = callback
        self.execution_count = 0
        self.start_time = None

        logger.info(f"Initializing CronJob with ID: {self.job_id}")

        self.reliability_check()

    def reliability_check(self):
        if self.agent is None:
            raise CronJobConfigError(
                "Agent must be provided during initialization"
            )

        # Parse interval if provided
        if self.interval:
            try:
                self._parse_interval(self.interval)
                logger.info(
                    f"CronJob {self.job_id}: Successfully configured interval: {self.interval}"
                )
            except ValueError as e:
                logger.error(
                    f"CronJob {self.job_id}: Failed to parse interval: {self.interval}"
                )
                raise CronJobConfigError(
                    f"Invalid interval format: {str(e)}"
                )

    def _parse_interval(self, interval: str):
        """Parse the interval string and set up the schedule.

        Args:
            interval: String in format "Xunit" where X is a number and unit is
                     seconds, minutes, or hours (e.g., "5seconds", "10minutes")

        Raises:
            CronJobConfigError: If the interval format is invalid or unit is unsupported
        """
        try:
            # Extract number and unit from interval string
            import re

            match = re.match(r"(\d+)(\w+)", interval.lower())
            if not match:
                raise CronJobConfigError(
                    f"Invalid interval format: {interval}. Expected format: '<number><unit>' (e.g., '5seconds', '10minutes')"
                )

            number = int(match.group(1))
            unit = match.group(2)

            # Map units to scheduling methods
            unit_map = {
                "second": self.every_seconds,
                "seconds": self.every_seconds,
                "minute": self.every_minutes,
                "minutes": self.every_minutes,
                "hour": lambda x: self.schedule.every(x).hours.do(
                    self._run_job
                ),
                "hours": lambda x: self.schedule.every(x).hours.do(
                    self._run_job
                ),
            }

            if unit not in unit_map:
                supported_units = ", ".join(unit_map.keys())
                raise CronJobConfigError(
                    f"Unsupported time unit: {unit}. Supported units are: {supported_units}"
                )

            self._interval_method = lambda task: unit_map[unit](
                number, task
            )
            logger.debug(f"Configured {number} {unit} interval")

        except ValueError as e:
            raise CronJobConfigError(
                f"Invalid interval number: {str(e)}"
            )
        except Exception as e:
            raise CronJobConfigError(
                f"Error parsing interval: {str(e)}"
            )

    def _run(self, task: str, **kwargs):
        """Run the scheduled job with the given task and additional parameters.

        Args:
            task: The task string to be executed by the agent
            **kwargs: Additional parameters to pass to the agent's run method
                     (e.g., img=image_path, streaming_callback=callback_func)

        Raises:
            CronJobConfigError: If agent or interval is not configured
            CronJobExecutionError: If task execution fails
        """
        try:
            if not self.agent:
                raise CronJobConfigError(
                    "Agent must be provided during initialization"
                )

            if not self.interval:
                raise CronJobConfigError(
                    "Interval must be provided during initialization"
                )

            logger.info(f"Scheduling task for job {self.job_id}")
            # Schedule the task with additional parameters
            self._interval_method(task, **kwargs)

            # Start the job
            self.start()
            logger.info(f"Successfully started job {self.job_id}")

        except Exception as e:
            logger.error(
                f"CronJob: Failed to run job {self.job_id}: {str(e)}"
            )
            raise CronJobExecutionError(
                f"Failed to run job: {str(e)} Traceback: {traceback.format_exc()}"
            )

    def run(self, task: str, **kwargs):
        try:
            job = self._run(task, **kwargs)

            while True:
                time.sleep(1)

            return job
        except KeyboardInterrupt:
            logger.info(
                f"CronJob: {self.job_id} received keyboard interrupt, stopping cron jobs..."
            )
            self.stop()
        except Exception as e:
            logger.error(
                f"CronJob: {self.job_id} error in main: {str(e)} Traceback: {traceback.format_exc()}"
            )
            raise

    def batched_run(self, tasks: List[str], **kwargs):
        """Run the scheduled job with the given tasks and additional parameters.

        Args:
            tasks: The list of task strings to be executed by the agent
            **kwargs: Additional parameters to pass to the agent's run method
        """
        outputs = []
        for task in tasks:
            output = self.run(task, **kwargs)
            outputs.append(output)
        return outputs

    def __call__(self, task: str, **kwargs):
        """Call the CronJob instance as a function.

        Args:
            task: The task string to be executed
            **kwargs: Additional parameters to pass to the agent's run method
        """
        return self.run(task, **kwargs)

    def _run_job(self, task: str, **kwargs) -> Any:
        """Internal method to run the job with provided task and parameters.

        Args:
            task: The task string to be executed
            **kwargs: Additional parameters to pass to the agent's run method
                     (e.g., img=image_path, streaming_callback=callback_func)

        Returns:
            Any: The result of the task execution (original or customized by callback)

        Raises:
            CronJobExecutionError: If task execution fails
        """
        try:
            logger.debug(f"Executing task for job {self.job_id}")

            # Execute the agent
            if isinstance(self.agent, Callable):
                original_output = self.agent.run(task=task, **kwargs)
            else:
                original_output = self.agent(task, **kwargs)

            # Increment execution count
            self.execution_count += 1

            # Prepare metadata for callback
            metadata = {
                "job_id": self.job_id,
                "timestamp": time.time(),
                "execution_count": self.execution_count,
                "task": task,
                "kwargs": kwargs,
                "start_time": self.start_time,
                "is_running": self.is_running,
            }

            # Apply callback if provided
            if self.callback:
                try:
                    customized_output = self.callback(
                        original_output, task, metadata
                    )
                    logger.debug(
                        f"Callback applied to job {self.job_id}, execution {self.execution_count}"
                    )
                    return customized_output
                except Exception as callback_error:
                    logger.warning(
                        f"Callback failed for job {self.job_id}: {callback_error}"
                    )
                    # Return original output if callback fails
                    return original_output

            return original_output

        except Exception as e:
            logger.error(
                f"Task execution failed for job {self.job_id}: {str(e)}"
            )
            raise CronJobExecutionError(
                f"Task execution failed: {str(e)}"
            )

    def every_seconds(self, seconds: int, task: str, **kwargs):
        """Schedule the job to run every specified number of seconds.

        Args:
            seconds: Number of seconds between executions
            task: The task to execute
            **kwargs: Additional parameters to pass to the agent's run method
        """
        logger.debug(
            f"Scheduling job {self.job_id} every {seconds} seconds"
        )
        self.schedule.every(seconds).seconds.do(
            self._run_job, task, **kwargs
        )

    def every_minutes(self, minutes: int, task: str, **kwargs):
        """Schedule the job to run every specified number of minutes.

        Args:
            minutes: Number of minutes between executions
            task: The task to execute
            **kwargs: Additional parameters to pass to the agent's run method
        """
        logger.debug(
            f"Scheduling job {self.job_id} every {minutes} minutes"
        )
        self.schedule.every(minutes).minutes.do(
            self._run_job, task, **kwargs
        )

    def start(self):
        """Start the scheduled job in a separate thread.

        Raises:
            CronJobExecutionError: If the job fails to start
        """
        try:
            if not self.is_running:
                self.is_running = True
                self.start_time = time.time()
                self.thread = threading.Thread(
                    target=self._run_schedule,
                    daemon=True,
                    name=f"cronjob_{self.job_id}",
                )
                self.thread.start()
                logger.info(f"Started job {self.job_id}")
            else:
                logger.warning(
                    f"Job {self.job_id} is already running"
                )
        except Exception as e:
            logger.error(
                f"Failed to start job {self.job_id}: {str(e)}"
            )
            raise CronJobExecutionError(
                f"Failed to start job: {str(e)}"
            )

    def stop(self):
        """Stop the scheduled job.

        Raises:
            CronJobExecutionError: If the job fails to stop properly
        """
        try:
            logger.info(f"Stopping job {self.job_id}")
            self.is_running = False
            if self.thread:
                self.thread.join(
                    timeout=5
                )  # Wait up to 5 seconds for thread to finish
                if self.thread.is_alive():
                    logger.warning(
                        f"Job {self.job_id} thread did not terminate gracefully"
                    )
                self.schedule.clear()
                logger.info(f"Successfully stopped job {self.job_id}")
        except Exception as e:
            logger.error(
                f"Error stopping job {self.job_id}: {str(e)}"
            )
            raise CronJobExecutionError(
                f"Failed to stop job: {str(e)}"
            )

    def _run_schedule(self):
        """Internal method to run the schedule loop."""
        logger.debug(f"Starting schedule loop for job {self.job_id}")
        while self.is_running:
            try:
                self.schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(
                    f"Error in schedule loop for job {self.job_id}: {str(e)}"
                )
                self.is_running = False
                raise CronJobExecutionError(
                    f"Schedule loop failed: {str(e)}"
                )

    def set_callback(self, callback: Callable[[Any, str, dict], Any]):
        """Set or update the callback function for output customization.

        Args:
            callback: Function to customize output processing.
                     Signature: callback(output: Any, task: str, metadata: dict) -> Any
        """
        self.callback = callback
        logger.info(f"Callback updated for job {self.job_id}")

    def get_execution_stats(self) -> dict:
        """Get execution statistics for the cron job.

        Returns:
            dict: Statistics including execution count, start time, running status, etc.
        """
        return {
            "job_id": self.job_id,
            "is_running": self.is_running,
            "execution_count": self.execution_count,
            "start_time": self.start_time,
            "uptime": (
                time.time() - self.start_time
                if self.start_time
                else 0
            ),
            "interval": self.interval,
        }


# # Example usage
# if __name__ == "__main__":
#     # Initialize the agent
#     agent = Agent(
#         agent_name="Quantitative-Trading-Agent",
#         agent_description="Advanced quantitative trading and algorithmic analysis agent",
#         system_prompt="""You are an expert quantitative trading agent with deep expertise in:
#         - Algorithmic trading strategies and implementation
#         - Statistical arbitrage and market making
#         - Risk management and portfolio optimization
#         - High-frequency trading systems
#         - Market microstructure analysis
#         - Quantitative research methodologies
#         - Financial mathematics and stochastic processes
#         - Machine learning applications in trading

#         Your core responsibilities include:
#         1. Developing and backtesting trading strategies
#         2. Analyzing market data and identifying alpha opportunities
#         3. Implementing risk management frameworks
#         4. Optimizing portfolio allocations
#         5. Conducting quantitative research
#         6. Monitoring market microstructure
#         7. Evaluating trading system performance

#         You maintain strict adherence to:
#         - Mathematical rigor in all analyses
#         - Statistical significance in strategy development
#         - Risk-adjusted return optimization
#         - Market impact minimization
#         - Regulatory compliance
#         - Transaction cost analysis
#         - Performance attribution

#         You communicate in precise, technical terms while maintaining clarity for stakeholders.""",
#         max_loops=1,
#         model_name="gpt-4.1",
#         dynamic_temperature_enabled=True,
#         output_type="str-all-except-first",
#         streaming_on=True,
#         print_on=True,
#         telemetry_enable=False,
#     )

#     # Example 1: Basic usage with just a task
#     logger.info("Starting example cron job")
#     cron_job = CronJob(agent=agent, interval="10seconds")
#     cron_job.run(
#         task="What are the best top 3 etfs for gold coverage?"
#     )
