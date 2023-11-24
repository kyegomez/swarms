import logging
import queue
import threading
from time import sleep
from typing import Callable, Dict, List

from termcolor import colored

from swarms.structs.flow import Flow
from swarms.utils.decorators import (
    error_decorator,
    log_decorator,
    timing_decorator,
)


class AutoScaler:
    """
    The AutoScaler is like a kubernetes pod, that autoscales an agent or worker or boss!

    Wraps around a structure like SequentialWorkflow
    and or Flow and parallelizes them on multiple threads so they're split across devices
    and you can use them like that
    Args:

        initial_agents (int, optional): Number of initial agents. Defaults to 10.
        scale_up_factor (int, optional): Scale up factor. Defaults to 1.
        idle_threshold (float, optional): Idle threshold. Defaults to 0.2.
        busy_threshold (float, optional): Busy threshold. Defaults to 0.7.
        agent ([type], optional): Agent. Defaults to None.


    Methods:
        add_task: Add task to queue
        scale_up: Scale up
        scale_down: Scale down
        monitor_and_scale: Monitor and scale
        start: Start scaling
        del_agent: Delete an agent

    Usage
    ```
    from swarms.swarms import AutoScaler
    from swarms.structs.flow import Flow

    @AutoScaler
    flow = Flow()

    flow.run("what is your name")
    ```
    """

    @log_decorator
    @error_decorator
    @timing_decorator
    def __init__(
        self,
        initial_agents=10,
        scale_up_factor=1,
        idle_threshold=0.2,
        busy_threshold=0.7,
        agent=None,
    ):
        self.agent = agent or Flow
        self.agents_pool = [self.agent() for _ in range(initial_agents)]
        self.task_queue = queue.Queue()
        self.scale_up_factor = scale_up_factor
        self.idle_threshold = idle_threshold
        self.lock = threading.Lock()

    def add_task(self, task):
        """Add tasks to queue"""
        try:
            self.tasks_queue.put(task)
        except Exception as error:
            print(
                f"Error adding task to queue: {error} try again with a new task"
            )

    @log_decorator
    @error_decorator
    @timing_decorator
    def scale_up(self):
        """Add more agents"""
        try:
            with self.lock:
                new_agents_counts = len(self.agents_pool) * self.scale_up_factor
                for _ in range(new_agents_counts):
                    self.agents_pool.append(Flow())
        except Exception as error:
            print(f"Error scaling up: {error} try again with a new task")

    def scale_down(self):
        """scale down"""
        try:
            with self.lock:
                if len(self.agents_pool) > 10:  # ensure minmum of 10 agents
                    del self.agents_pool[-1]  # remove last agent
        except Exception as error:
            print(f"Error scaling down: {error} try again with a new task")

    @log_decorator
    @error_decorator
    @timing_decorator
    def monitor_and_scale(self):
        """Monitor and scale"""
        try:
            while True:
                sleep(60)  # check minute
                pending_tasks = self.task_queue.qsize()
                active_agents = sum(
                    [1 for agent in self.agents_pool if agent.is_busy()]
                )

                if pending_tasks / len(self.agents_pool) > self.busy_threshold:
                    self.scale_up()
                elif (
                    active_agents / len(self.agents_pool) < self.idle_threshold
                ):
                    self.scale_down()
        except Exception as error:
            print(
                f"Error monitoring and scaling: {error} try again with a new"
                " task"
            )

    @log_decorator
    @error_decorator
    @timing_decorator
    def start(self):
        """Start scaling"""
        try:
            monitor_thread = threading.Thread(target=self.monitor_and_scale)
            monitor_thread.start()

            while True:
                task = self.task_queue.get()
                if task:
                    available_agent = next(
                        (agent for agent in self.agents_pool)
                    )
                    if available_agent:
                        available_agent.run(task)
        except Exception as error:
            print(f"Error starting: {error} try again with a new task")

    def check_agent_health(self):
        """Checks the health of each agent and replaces unhealthy agents."""
        for i, agent in enumerate(self.agents_pool):
            if not agent.is_healthy():
                logging.warning(f"Replacing unhealthy agent at index {i}")
                self.agents_pool[i] = self.agent()

    def balance_load(self):
        """Distributes tasks among agents based on their current load."""
        while not self.task_queue.empty():
            for agent in self.agents_pool:
                if agent.can_accept_task():
                    task = self.task_queue.get()
                    agent.run(task)

    def set_scaling_strategy(self, strategy: Callable[[int, int], int]):
        """Set a custom scaling strategy."""
        self.custom_scale_strategy = strategy

    def execute_scaling_strategy(self):
        """Execute the custom scaling strategy if defined."""
        if hasattr(self, "custom_scale_strategy"):
            scale_amount = self.custom_scale_strategy(
                self.task_queue.qsize(), len(self.agents_pool)
            )
            if scale_amount > 0:
                for _ in range(scale_amount):
                    self.agents_pool.append(self.agent())
            elif scale_amount < 0:
                for _ in range(abs(scale_amount)):
                    if len(self.agents_pool) > 10:
                        del self.agents_pool[-1]

    def report_agent_metrics(self) -> Dict[str, List[float]]:
        """Collects and reports metrics from each agent."""
        metrics = {"completion_time": [], "success_rate": [], "error_rate": []}
        for agent in self.agents_pool:
            agent_metrics = agent.get_metrics()
            for key in metrics.keys():
                metrics[key].append(agent_metrics.get(key, 0.0))
        return metrics

    def report(self):
        """Reports the current state of the autoscaler."""
        self.check_agent_health()
        self.balance_load()
        self.execute_scaling_strategy()
        metrics = self.report_agent_metrics()
        print(metrics)

    def print_dashboard(self):
        """Prints a dashboard of the current state of the autoscaler."""
        print(
            colored(
                f"""

            Autoscaler Dashboard
            --------------------
            Agents: {len(self.agents_pool)}
            Initial Agents: {self.initial_agents}
            self.scale_up_factor: {self.scale_up_factor}
            self.idle_threshold: {self.idle_threshold}
            self.busy_threshold: {self.busy_threshold}
            self.task_queue.qsize(): {self.task_queue.qsize()}
            self.task_queue.empty(): {self.task_queue.empty()}
            self.task_queue.full(): {self.task_queue.full()}
            self.task_queue.maxsize: {self.task_queue.maxsize}

            """,
                "cyan",
            )
        )
