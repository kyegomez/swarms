from typing import List
from swarms import Worker

class MultiAgentDebate:
    def __init__(self, agents: List[Worker]):
        self.agents = agents

    def run(self, task: str):
        results = []
        for agent in self.agents:
            response = agent.run(task)
            results.append({
                'agent': agent.ai_name,
                'response': response
            })
        return results

# Initialize agents
agents = [
    Worker(openai_api_key="", ai_name="Optimus Prime"),
    Worker(openai_api_key="", ai_name="Bumblebee"),
    Worker(openai_api_key="", ai_name="Megatron")
]

# Initialize multi-agent debate
debate = MultiAgentDebate(agents)

# Run task
task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
results = debate.run(task)

# Print results
for result in results:
    print(f"Agent {result['agent']} responded: {result['response']}")