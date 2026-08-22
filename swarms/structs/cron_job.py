import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

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
    """Turn any callable, including a Swarms ``Agent``, into a scheduled job.

    One ``CronJob`` binds **one** agent to **one** interval. It schedules the
    task, runs it on its own background thread, and keeps running it until you
    stop it. For a fleet of agents on different cadences, see :meth:`run_many`.

    **Failure behaviour.** A task that raises is logged and retried on the next
    tick, the way cron behaves. It does not take the schedule down. Set
    ``max_consecutive_errors`` to stop a job that is failing every single time;
    when that budget is exhausted the job stops *and* :meth:`run` raises, so a
    dead schedule is never mistaken for a healthy one.

    Args:
        agent: The Swarms ``Agent`` instance or plain callable to schedule.
            Anything exposing ``run(task=...)`` is called through it; anything
            else is called directly as ``agent(task)``.
        interval: How often to run, as ``"<number><unit>"`` where unit is one of
            ``second(s)``, ``minute(s)``, ``hour(s)``. For example ``"30seconds"``,
            ``"10minutes"``, ``"1hour"``. Must be greater than zero.
        job_id: Unique identifier. Generated if omitted.
        callback: Optional post-processor, called as
            ``callback(output, task, metadata)`` and returning the value the job
            should yield. A callback that raises is logged and the original
            output is used.
        max_consecutive_errors: Stop after this many back-to-back failures.
            ``None`` (default) retries forever.

    Attributes:
        is_running (bool): Whether the scheduler thread is live.
        execution_count (int): Successful executions so far.
        error_count (int): Total failed executions.
        consecutive_errors (int): Failures since the last success. Reset to 0
            on any successful tick.
        last_error (Optional[Exception]): The most recent failure, if any.
        start_time (Optional[float]): Epoch seconds when the job started.

    Raises:
        CronJobConfigError: If ``agent`` is missing, or ``interval`` is empty,
            zero, or malformed.

    Example:
        >>> from swarms import Agent
        >>> from swarms.structs.cron_job import CronJob
        >>> agent = Agent(agent_name="Reporter", model_name="gpt-5.4", max_loops=1)
        >>> job = CronJob(agent=agent, interval="10minutes")
        >>> job.run("Summarise anything new since the last check")  # blocks
    """

    def __init__(
        self,
        agent: Optional[Union[Any, Callable]] = None,
        interval: Optional[str] = None,
        job_id: Optional[str] = None,
        callback: Optional[Callable[[Any, str, dict], Any]] = None,
        max_consecutive_errors: Optional[int] = None,
    ):
        """Initialize the CronJob wrapper.

        Args:
            agent: The Swarms Agent instance or callable to be scheduled
            interval: The interval string (e.g., "5seconds", "10minutes", "1hour")
            max_consecutive_errors: Give up after this many back-to-back failed
                executions. ``None`` (the default) never gives up, which is what
                cron does: a task that fails is logged and retried on the next
                tick. Set an integer to stop a job that is failing every time.
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

        # Failures stay visible without taking the schedule down.
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error = None
        self.max_consecutive_errors = max_consecutive_errors
        self._stopped_due_to_error = False

        logger.info(f"Initializing CronJob with ID: {self.job_id}")

        self.reliability_check()

    def reliability_check(self):
        if self.agent is None:
            raise CronJobConfigError(
                "Agent must be provided during initialization"
            )

        # An empty string is a bad interval, not a missing one: fail here, not in _run().
        if (
            self.interval is not None
            and not str(self.interval).strip()
        ):
            raise CronJobConfigError(
                f"Interval cannot be empty, got {self.interval!r}. "
                'Use a value like "5seconds", "10minutes" or "1hour".'
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

            # "0second" parsed fine and then scheduled a job that never fired:
            # no error, no log line, just a cron job that silently does nothing.
            if number == 0:
                raise CronJobConfigError(
                    f"Interval must be greater than zero, got {interval!r}. "
                    "A zero interval schedules a job that never runs."
                )

            # Map units to scheduling methods
            unit_map = {
                "second": self.every_seconds,
                "seconds": self.every_seconds,
                "minute": self.every_minutes,
                "minutes": self.every_minutes,
                "hour": lambda x, task, **kwargs: self.schedule.every(
                    x
                ).hours.do(self._run_job, task, **kwargs),
                "hours": lambda x, task, **kwargs: self.schedule.every(
                    x
                ).hours.do(
                    self._run_job, task, **kwargs
                ),
            }

            if unit not in unit_map:
                supported_units = ", ".join(unit_map.keys())
                raise CronJobConfigError(
                    f"Unsupported time unit: {unit}. Supported units are: {supported_units}"
                )

            self._interval_method = lambda task, **kwargs: unit_map[
                unit
            ](number, task, **kwargs)
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

        Returns:
            The scheduled job instance

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
            job = self._interval_method(task, **kwargs)

            # Start the job
            self.start()
            logger.info(f"Successfully started job {self.job_id}")
            return job

        except Exception as e:
            logger.error(
                f"CronJob: Failed to run job {self.job_id}: {str(e)}"
            )
            raise CronJobExecutionError(
                f"Failed to run job: {str(e)} Traceback: {traceback.format_exc()}"
            )

    def _raise_if_stopped_by_errors(self):
        """Surface a schedule that gave up instead of returning normally.

        ``_block_forever`` loops on ``is_running``, which the scheduler clears
        both on a clean ``stop()`` and on giving up after repeated failures.
        Without this check the two are indistinguishable to the caller: ``run``
        hands back a ``Job`` object and the schedule is quietly dead.
        """
        if self._stopped_due_to_error:
            raise CronJobExecutionError(
                f"Job {self.job_id} stopped after "
                f"{self.consecutive_errors} consecutive failed executions "
                f"({self.error_count} total). Last error: {self.last_error}"
            )

    def _block_forever(self):
        """Block until interrupted or stopped."""
        try:
            while self.is_running:
                time.sleep(1)
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

    def run(self, task: str, **kwargs):
        """Schedule ``task`` and block, running it every interval until stopped.

        Blocks the calling thread. The task runs on a background thread, so use
        ``KeyboardInterrupt`` (Ctrl-C) or :meth:`stop` from another thread to end
        it. A failing task is logged and retried on the next tick.

        Args:
            task: The task string handed to the agent on every tick.
            **kwargs: Forwarded to the agent's ``run`` on every tick, for example
                ``img="chart.png"`` or ``streaming_callback=fn``.

        Returns:
            schedule.Job: The scheduled job, returned once the schedule stops.

        Raises:
            CronJobConfigError: If no agent or interval was configured.
            CronJobExecutionError: If scheduling failed, or if the job gave up
                after exhausting ``max_consecutive_errors``. The message names
                the failure count and the last error.

        Example:
            >>> job = CronJob(agent=agent, interval="30seconds")
            >>> job.run("Check the BTC price and flag moves over 2%")
        """
        job = self._run(task, **kwargs)
        self._block_forever()
        self._raise_if_stopped_by_errors()
        return job

    def batched_run(self, tasks: List[str], **kwargs):
        """Schedule several tasks on the **same** interval, then block.

        Every task in ``tasks`` is registered before blocking, so all of them
        run on each tick. This is one agent doing several things on one cadence.
        For several agents on *different* cadences use :meth:`run_many`.

        Args:
            tasks: Task strings, each scheduled at this job's interval.
            **kwargs: Forwarded to the agent's ``run`` for every task.

        Returns:
            List[schedule.Job]: One scheduled job per task, in input order.

        Raises:
            CronJobConfigError: If no agent or interval was configured.
            CronJobExecutionError: If scheduling failed, or if the job gave up
                after exhausting ``max_consecutive_errors``.

        Example:
            >>> job = CronJob(agent=agent, interval="1hour")
            >>> job.batched_run(["Check inventory", "Check refunds"])
        """
        outputs = []
        for task in tasks:
            output = self._run(task, **kwargs)
            outputs.append(output)
        self._block_forever()
        self._raise_if_stopped_by_errors()
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

            # Dispatch on having run(), not on Callable: a plain function is Callable too.
            runner = getattr(self.agent, "run", None)
            if callable(runner):
                original_output = runner(task=task, **kwargs)
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
        return self.schedule.every(seconds).seconds.do(
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
        return self.schedule.every(minutes).minutes.do(
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
        """Stop the job and join its scheduler thread.

        Safe to call from another thread while :meth:`run` is blocking, and safe
        to call on a job that is already stopped. Waits up to 5 seconds for the
        scheduler thread to exit.

        Raises:
            CronJobExecutionError: If the job fails to stop properly.

        Example:
            >>> threading.Timer(60, job.stop).start()  # stop after a minute
            >>> job.run("Poll the queue")
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
                self.consecutive_errors = 0
            except Exception as e:
                # Log and keep going: one failed execution must not kill the scheduler.
                self.error_count += 1
                self.consecutive_errors += 1
                self.last_error = e
                logger.error(
                    f"Execution failed for job {self.job_id} "
                    f"(failure {self.consecutive_errors} in a row, "
                    f"{self.error_count} total): {str(e)}\n"
                    f"{traceback.format_exc()}"
                )

                if (
                    self.max_consecutive_errors is not None
                    and self.consecutive_errors
                    >= self.max_consecutive_errors
                ):
                    logger.error(
                        f"Job {self.job_id} stopping: "
                        f"{self.consecutive_errors} consecutive failures "
                        f"reached max_consecutive_errors="
                        f"{self.max_consecutive_errors}"
                    )
                    self._stopped_due_to_error = True
                    self.is_running = False
                    return

            time.sleep(1)

    def set_callback(self, callback: Callable[[Any, str, dict], Any]):
        """Set or update the callback function for output customization.

        Args:
            callback: Function to customize output processing.
                     Signature: callback(output: Any, task: str, metadata: dict) -> Any
        """
        self.callback = callback
        logger.info(f"Callback updated for job {self.job_id}")

    def get_execution_stats(self) -> dict:
        """Snapshot of how the job is doing, safe to poll while it runs.

        Returns:
            dict: ``job_id``, ``is_running``, ``execution_count`` (successes),
            ``start_time``, ``uptime`` in seconds, ``interval``, plus the failure
            picture: ``error_count`` (total failures), ``consecutive_errors``
            (since the last success), ``last_error`` as a string or ``None``, and
            ``stopped_due_to_error`` which is ``True`` only when the job gave up
            after exhausting ``max_consecutive_errors``.

        Example:
            >>> stats = job.get_execution_stats()
            >>> if stats["consecutive_errors"] > 3:
            ...     alert(f"{stats['job_id']} is struggling: {stats['last_error']}")
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
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "last_error": (
                str(self.last_error) if self.last_error else None
            ),
            "stopped_due_to_error": self._stopped_due_to_error,
        }

    @classmethod
    def run_many(
        cls,
        schedules: List[Dict[str, Any]],
        block: bool = True,
    ) -> List["CronJob"]:
        """Run several agents together, each on its own interval.

        A ``CronJob`` binds one agent to one interval, so a fleet on mixed
        cadences needs one job per agent. This builds them, starts them all,
        and optionally blocks until interrupted.

        Each job keeps its own scheduler thread, so the agents are isolated:
        one failing does not delay or stop the others, and each carries its
        own error counters and ``max_consecutive_errors`` budget.

        Args:
            schedules: One mapping per agent. ``agent``, ``interval`` and
                ``task`` are required. Optional keys: ``job_id``, ``callback``,
                ``max_consecutive_errors``, and ``kwargs`` (a dict forwarded to
                the agent's ``run``).
            block: Hold the calling thread until KeyboardInterrupt, then stop
                every job. Pass ``False`` to start them and return immediately,
                leaving the caller responsible for ``stop_many``.

        Returns:
            List[CronJob]: The started jobs, in the order given, so they can be
            inspected via ``get_execution_stats()`` or stopped individually.

        Raises:
            CronJobConfigError: If ``schedules`` is empty or an entry is
                missing a required key.
            CronJobExecutionError: If, once blocking ends, any job had stopped
                because it exhausted ``max_consecutive_errors``.

        Example:
            >>> CronJob.run_many([
            ...     {"agent": price_agent, "interval": "30seconds",
            ...      "task": "Check BTC price"},
            ...     {"agent": digest_agent, "interval": "1hour",
            ...      "task": "Summarise the last hour"},
            ... ])
        """
        if not schedules:
            raise CronJobConfigError(
                "run_many requires at least one schedule"
            )

        jobs: List["CronJob"] = []
        for index, spec in enumerate(schedules):
            missing = [
                key
                for key in ("agent", "interval", "task")
                if not spec.get(key)
            ]
            if missing:
                raise CronJobConfigError(
                    f"schedules[{index}] is missing required "
                    f"key(s): {', '.join(missing)}"
                )

            job = cls(
                agent=spec["agent"],
                interval=spec["interval"],
                job_id=spec.get("job_id"),
                callback=spec.get("callback"),
                max_consecutive_errors=spec.get(
                    "max_consecutive_errors"
                ),
            )
            # _run schedules the task and starts this job's own thread; it
            # does not block, which is what lets the fleet run together.
            job._run(spec["task"], **(spec.get("kwargs") or {}))
            jobs.append(job)

        logger.info(
            f"run_many started {len(jobs)} job(s): "
            + ", ".join(f"{j.job_id}@{j.interval}" for j in jobs)
        )

        if not block:
            return jobs

        try:
            # Wait while the per-job threads work; exit once every job has stopped.
            while any(job.is_running for job in jobs):
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info(
                "run_many received keyboard interrupt, stopping all jobs"
            )
        finally:
            cls.stop_many(jobs)

        failed = [job for job in jobs if job._stopped_due_to_error]
        if failed:
            raise CronJobExecutionError(
                f"{len(failed)} of {len(jobs)} job(s) stopped after "
                "repeated failures: "
                + "; ".join(
                    f"{job.job_id} ({job.consecutive_errors} consecutive, "
                    f"last error: {job.last_error})"
                    for job in failed
                )
            )

        return jobs

    @staticmethod
    def stop_many(jobs: List["CronJob"]) -> None:
        """Stop every job in ``jobs``, continuing past any that fail to stop.

        Args:
            jobs: The jobs to stop, typically the return value of
                :meth:`run_many` called with ``block=False``.
        """
        for job in jobs:
            try:
                job.stop()
            except Exception as e:
                # One job refusing to stop must not strand the rest.
                logger.error(f"Failed to stop job {job.job_id}: {e}")


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
#         model_name="gpt-5.4",
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
