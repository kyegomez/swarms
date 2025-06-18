"""Example of applying stopping conditions to a swarm."""

from swarms import Agent, check_done
from swarms.structs.round_robin import RoundRobinSwarm


class StoppableRoundRobinSwarm(RoundRobinSwarm):
    """Round-robin swarm that stops when a condition is met."""

    def __init__(self, *args, stopping_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stopping_function = stopping_function

    def run(self, task: str, *args, **kwargs):
        result = task
        n = len(self.agents)
        for _ in range(self.max_loops):
            for _ in range(n):
                agent = self.agents[self.index]
                result = self._execute_agent(agent, result, *args, **kwargs)
                self.index = (self.index + 1) % n
                if self.stopping_function and self.stopping_function(result):
                    print("Stopping condition met.")
                    return result
        return result


agent1 = Agent(
    agent_name="Worker-1",
    system_prompt="Return <DONE> when finished.",
    stopping_func=check_done,
    max_loops=5,
)

agent2 = Agent(
    agent_name="Worker-2",
    system_prompt="Return <DONE> when finished.",
    stopping_func=check_done,
    max_loops=5,
)

swarm = StoppableRoundRobinSwarm(
    agents=[agent1, agent2],
    max_loops=3,
    stopping_function=check_done,
)

result = swarm.run("Collect and summarize data.")
print(result)
