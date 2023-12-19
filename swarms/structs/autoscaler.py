import logging
import queue
import threading
from time import sleep
from typing import Callable, Dict, List, Optional

from termcolor import colored

from swarms.structs.agent import Agent
from swarms.utils.decorators import (
    error_decorator,
    log_decorator,
    timing_decorator,
)


class AutoScaler:
    """
    AutoScaler class

    The AutoScaler class is responsible for managing the agents pool
    and the task queue. It also monitors the health of the agents and
    scales the pool up or down based on the number of pending tasks
    and the current load of the agents.

    Args:
        initial_agents (Optional[int], optional): Initial number of
            agents to start with. Defaults to 10.
        scale_up_factor (int, optional): Factor by which to scale up
            the agents pool. Defaults to 1.
        idle_threshold (float, optional): Threshold for scaling down
            the agents pool. Defaults to 0.2.
        busy_threshold (float, optional): Threshold for scaling up
            the agents pool. Defaults to 0.7.
        agents (List[Agent], optional): List of agents to use in the
            pool. Defaults to None.
        autoscale (Optional[bool], optional): Whether to autoscale
            the agents pool. Defaults to True.
        min_agents (Optional[int], optional): Minimum number of
            agents to keep in the pool. Defaults to 10.
        max_agents (Optional[int], optional): Maximum number of
            agents to keep in the pool. Defaults to 100.
        custom_scale_strategy (Optional[Callable], optional): Custom
            scaling strategy to use. Defaults to None.

    Methods:
        add_task: Add tasks to queue
        scale_up: Add more agents
        scale_down: scale down
        run: Run agent the task on the agent id
        monitor_and_scale: Monitor and scale
        start: Start scaling
        check_agent_health: Checks the health of each agent and
            replaces unhealthy agents.
        balance_load: Distributes tasks among agents based on their
            current load.
        set_scaling_strategy: Set a custom scaling strategy.
        execute_scaling_strategy: Execute the custom scaling strategy
            if defined.
        report_agent_metrics: Collects and reports metrics from each
            agent.
        report: Reports the current state of the autoscaler.
        print_dashboard: Prints a dashboard of the current state of
            the autoscaler.

    Examples:
    >>> import os
    >>> from dotenv import load_dotenv
    >>> # Import the OpenAIChat model and the Agent struct
    >>> from swarms.models import OpenAIChat
    >>> from swarms.structs import Agent
    >>> from swarms.structs.autoscaler import AutoScaler
    >>> # Load the environment variables
    >>> load_dotenv()
    >>> # Get the API key from the environment
    >>> api_key = os.environ.get("OPENAI_API_KEY")
    >>> # Initialize the language model
    >>> llm = OpenAIChat(
    ...     temperature=0.5,
    ...     openai_api_key=api_key,
    ... )
    >>> ## Initialize the workflow
    >>> agent = Agent(llm=llm, max_loops=1, dashboard=True)
    >>> # Load the autoscaler
    >>> autoscaler = AutoScaler(
    ...     initial_agents=2,
    ...     scale_up_factor=1,
    ...     idle_threshold=0.2,
    ...     busy_threshold=0.7,
    ...     agents=[agent],
    ...     autoscale=True,
    ...     min_agents=1,
    ...     max_agents=5,
    ...     custom_scale_strategy=None,
    ... )
    >>> print(autoscaler)
    >>> # Run the workflow on a task
    >>> out = autoscaler.run(agent.id, "Generate a 10,000 word blog on health and wellness.")
    >>> print(out)

    """

    @log_decorator
    @error_decorator
    @timing_decorator
    def __init__(
        self,
        initial_agents: Optional[int] = 10,
        scale_up_factor: int = 1,
        idle_threshold: float = 0.2,
        busy_threshold: float = 0.7,
        agents: List[Agent] = None,
        autoscale: Optional[bool] = True,
        min_agents: Optional[int] = 10,
        max_agents: Optional[int] = 100,
        custom_scale_strategy: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        self.agents_pool = agents or [
            agents[0]() for _ in range(initial_agents)
        ]
        self.task_queue = queue.Queue()
        self.scale_up_factor = scale_up_factor
        self.idle_threshold = idle_threshold
        self.busy_threshold = busy_threshold
        self.lock = threading.Lock()
        self.agents = agents
        self.autoscale = autoscale
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.custom_scale_strategy = custom_scale_strategy

    def add_task(self, task):
        """Add tasks to queue"""
        try:
            self.task_queue.put(task)
        except Exception as error:
            print(
                f"Error adding task to queue: {error} try again with"
                " a new task"
            )

    @log_decorator
    @error_decorator
    @timing_decorator
    def scale_up(self):
        """Add more agents"""
        try:
            with self.lock:
                new_agents_counts = (
                    len(self.agents_pool) * self.scale_up_factor
                )
                for _ in range(new_agents_counts):
                    self.agents_pool.append(self.agents[0]())
        except Exception as error:
            print(
                f"Error scaling up: {error} try again with a new task"
            )

    def scale_down(self):
        """scale down"""
        try:
            with self.lock:
                if (
                    len(self.agents_pool) > 10
                ):  # ensure minmum of 10 agents
                    del self.agents_pool[-1]  # remove last agent
        except Exception as error:
            print(
                f"Error scaling down: {error} try again with a new"
                " task"
            )

    def run(
        self, agent_id, task: Optional[str] = None, *args, **kwargs
    ):
        """Run agent the task on the agent id

        Args:
            agent_id (_type_): _description_
            task (str, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        for agent in self.agents_pool:
            if agent.id == agent_id:
                return agent.run(task, *args, **kwargs)
        raise ValueError(f"No agent found with ID {agent_id}")

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
                    [
                        1
                        for agent in self.agents_pool
                        if agent.is_busy()
                    ]
                )

                if (
                    pending_tasks / len(self.agents_pool)
                    > self.busy_threshold
                ):
                    self.scale_up()
                elif (
                    active_agents / len(self.agents_pool)
                    < self.idle_threshold
                ):
                    self.scale_down()
        except Exception as error:
            print(
                f"Error monitoring and scaling: {error} try again"
                " with a new task"
            )

    @log_decorator
    @error_decorator
    @timing_decorator
    def start(self):
        """Start scaling"""
        try:
            monitor_thread = threading.Thread(
                target=self.monitor_and_scale
            )
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
            print(
                f"Error starting: {error} try again with a new task"
            )

    def check_agent_health(self):
        """Checks the health of each agent and replaces unhealthy agents."""
        for i, agent in enumerate(self.agents_pool):
            if not agent.is_healthy():
                logging.warning(
                    f"Replacing unhealthy agent at index {i}"
                )
                self.agents_pool[i] = self.agent()

    def balance_load(self):
        """Distributes tasks among agents based on their current load."""
        while not self.task_queue.empty():
            for agent in self.agents_pool:
                if agent.can_accept_task():
                    task = self.task_queue.get()
                    agent.run(task)

    def set_scaling_strategy(
        self, strategy: Callable[[int, int], int]
    ):
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
        metrics = {
            "completion_time": [],
            "success_rate": [],
            "error_rate": [],
        }
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
