import queue
import threading
from time import sleep
from swarms.utils.decorators import error_decorator, log_decorator, timing_decorator
from swarms.structs.flow import Flow


class AutoScaler:
    """
    The AutoScaler is like a kubernetes pod, that autoscales an agent or worker or boss!
    # TODO Handle task assignment and task delegation
    # TODO: User task => decomposed into very small sub tasks => sub tasks assigned to workers => workers complete and update the swarm, can ask for help from other agents.
    # TODO: Missing, Task Assignment, Task delegation, Task completion, Swarm level communication with vector db


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
    # usage of usage
    auto_scaler = AutoScaler(agent=YourCustomAgent)
    auto_scaler.start()

    for i in range(100):
    auto_scaler.add_task9f"task {I}})
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
        self.tasks_queue.put(task)

    @log_decorator
    @error_decorator
    @timing_decorator
    def scale_up(self):
        """Add more agents"""
        with self.lock:
            new_agents_counts = len(self.agents_pool) * self.scale_up_factor
            for _ in range(new_agents_counts):
                self.agents_pool.append(Flow())

    def scale_down(self):
        """scale down"""
        with self.lock:
            if len(self.agents_pool) > 10:  # ensure minmum of 10 agents
                del self.agents_pool[-1]  # remove last agent

    @log_decorator
    @error_decorator
    @timing_decorator
    def monitor_and_scale(self):
        """Monitor and scale"""
        while True:
            sleep(60)  # check minute
            pending_tasks = self.task_queue.qsize()
            active_agents = sum(
                [1 for agent in self.agents_pool if agent.is_busy()])

            if pending_tasks / len(self.agents_pool) > self.busy_threshold:
                self.scale_up()
            elif active_agents / len(self.agents_pool) < self.idle_threshold:
                self.scale_down()

    @log_decorator
    @error_decorator
    @timing_decorator
    def start(self):
        """Start scaling"""
        monitor_thread = threading.Thread(target=self.monitor_and_scale)
        monitor_thread.start()

        while True:
            task = self.task_queue.get()
            if task:
                available_agent = next((agent for agent in self.agents_pool))
                if available_agent:
                    available_agent.run(task)

    # def del_agent(self):
    #     """Delete an agent"""
    #     with self.lock:
    #         if self.agents_pool:
    #             self.agents_poo.pop()
    #             del agent_to_remove
