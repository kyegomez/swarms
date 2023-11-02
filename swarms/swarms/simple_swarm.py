from queue import Queue, PriorityQueue


class SimpleSwarm:
    def __init__(
        self,
        llm,
        num_agents: int = None,
        openai_api_key: str = None,
        ai_name: str = None,
        rounds: int = 1,
        *args,
        **kwargs,
    ):
        """

        Usage:
        # Initialize the swarm with 5 agents, an API key, and a name for the AI model
        swarm = SimpleSwarm(num_agents=5, openai_api_key="YOUR_OPENAI_API_KEY", ai_name="Optimus Prime")

        # Normal task without priority
        normal_task = "Describe the process of photosynthesis in simple terms."
        swarm.distribute_task(normal_task)

        # Priority task; lower numbers indicate higher priority (e.g., 1 is higher priority than 2)
        priority_task = "Translate the phrase 'Hello World' to French."
        swarm.distribute_task(priority_task, priority=1)

        # Run the tasks and gather the responses
        responses = swarm.run()

        # Print responses
        for response in responses:
            print(response)

        # Providing feedback to the system (this is a stubbed method and won't produce a visible effect, but serves as an example)
        swarm.provide_feedback("Improve translation accuracy.")

        # Perform a health check on the agents (this is also a stubbed method, illustrating potential usage)
        swarm.health_check()

        """
        self.llm = llm
        self.agents = [self.llm for _ in range(num_agents)]
        self.task_queue = Queue()
        self.priority_queue = PriorityQueue()

    def distribute(self, task: str = None, priority=None):
        """Distribute a task to the agents"""
        if priority:
            self.priority_queue.put((priority, task))
        else:
            self.task_queue.put(task)

    def _process_task(self, task):
        # TODO, Implement load balancing, fallback mechanism
        for worker in self.agents:
            response = worker.run(task)
            if response:
                return response
        return "All Agents failed"

    def run(self):
        """Run the simple swarm"""

        responses = []

        # process high priority tasks first
        while not self.priority_queue.empty():
            _, task = self.priority_queue.get()
            responses.append(self._process_task(task))

        # process normal tasks
        while not self.task_queue.empty():
            task = self.task_queue.get()
            responses.append(self._process_task(task))

        return responses

    def run_old(self, task):
        responses = []

        for worker in self.agents:
            response = worker.run(task)
            responses.append(response)

        return responses

    def __call__(self, task):
        return self.run(task)
