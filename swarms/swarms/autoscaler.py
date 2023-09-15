import queue
import threading
from time import sleep

from swarms.utils.decorators import error_decorator, log_decorator, timing_decorator
from swarms.workers.worker import Worker

class AutoScaler:
    """
    The AutoScaler is like a kubernetes pod, that autoscales an agent or worker or boss!
    # TODO Handle task assignment and task delegation
    # TODO: User task => decomposed into very small sub tasks => sub tasks assigned to workers => workers complete and update the swarm, can ask for help from other agents. 
    # TODO: Missing, Task Assignment, Task delegation, Task completion, Swarm level communication with vector db

    
    Example
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
        self.agent = agent or Worker
        self.agents_pool = [self.agent() for _ in range(initial_agents)]
        self.task_queue = queue.Queue()
        self.scale_up_factor = scale_up_factor
        self.idle_threshold = idle_threshold
        self.lock = threading.Lock()

    def add_task(self, task):
        self.tasks_queue.put(task)
    
    @log_decorator
    @error_decorator
    @timing_decorator
    def scale_up(self):
        with self.lock:
            new_agents_counts = len(self.agents_pool) * self.scale_up_factor
            for _ in range(new_agents_counts):
                self.agents_pool.append(Worker())
    
    def scale_down(self):
        with self.lock:
            if len(self.agents_pool) > 10: #ensure minmum of 10 agents
                del self.agents_pool[-1] #remove last agent
    
    @log_decorator
    @error_decorator
    @timing_decorator
    def monitor_and_scale(self):
        while True:
            sleep(60)#check minute
            pending_tasks = self.task_queue.qsize()
            active_agents = sum([1 for agent in self.agents_pool if agent.is_busy()])

            if pending_tasks / len(self.agents_pool) > self.busy_threshold:
                self.scale_up()
            elif active_agents / len(self.agents_pool) < self.idle_threshold:
                self.scale_down()

    @log_decorator
    @error_decorator
    @timing_decorator
    def start(self):
        monitor_thread = threading.Thread(target=self.monitor_and_scale)
        monitor_thread.start()

        while True:
            task = self.task_queue.get()
            if task:
                available_agent = next((agent for agent in self.agents_pool))
                if available_agent:
                    available_agent.run(task)

    def del_agent(self):
        with self.lock:
            if self.agents_pool:
                agent_to_remove = self.agents_poo.pop()
                del agent_to_remove

