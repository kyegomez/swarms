from typing import List, Callable
from swarms import Worker

class MultiAgentDebate:
    def __init__(self, agents: List[Worker], selection_func: Callable[[int, List[Worker]], int]):
        self.agents = agents
        self.selection_func = selection_func

    def run(self, task: str):
        results = []

        for i in range(len(self.agents)):
            # Select the speaker based on the selection function
            speaker_idx = self.selection_func(i, self.agents)
            speaker = self.agents[speaker_idx]
            response = speaker.run(task)
            results.append({
                'agent': speaker.ai_name,
                'response': response
            })
        return results

# Define a selection function
def select_speaker(step: int, agents: List[Worker]) -> int:
    # This function selects the speaker in a round-robin fashion
    return step % len(agents)

# Initialize agents
worker1 = Worker(openai_api_key="", ai_name="Optimus Prime")
worker2 = Worker(openai_api_key="", ai_name="Bumblebee")
worker3 = Worker(openai_api_key="", ai_name="Megatron")

agents = [
    worker1,
    worker2,
    worker3
]

# Initialize multi-agent debate with the selection function
debate = MultiAgentDebate(agents, select_speaker)

# Run task
task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
results = debate.run(task)

# Print results
for result in results:
    print(f"Agent {result['agent']} responded: {result['response']}")