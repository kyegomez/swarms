from typing import List, Callable
from swarms.workers.worker import Worker


# Define a selection function
def select_speaker(step: int, agents: List[Worker]) -> int:
    # This function selects the speaker in a round-robin fashion
    return step % len(agents)


class MultiAgentDebate:
    """
    MultiAgentDebate

    Args:


    """

    def __init__(
        self, agents: List[Worker], selection_func: Callable[[int, List[Worker]], int]
    ):
        self.agents = agents
        self.selection_func = selection_func

    def reset_agents(self):
        for agent in self.agents:
            agent.reset()

    def inject_agent(self, agent: Worker):
        self.agents.append(agent)

    def run(self, task: str, max_iters: int = None):
        self.reset_agents()
        results = []
        for i in range(max_iters or len(self.agents)):
            speaker_idx = self.selection_func(i, self.agents)
            speaker = self.agents[speaker_idx]
            response = speaker.run(task)
            results.append({"agent": speaker.ai_name, "response": response})
        return results

    def update_task(self, task: str):
        self.task = task

    def format_results(self, results):
        formatted_results = "\n".join(
            [
                f"Agent {result['agent']} responded: {result['response']}"
                for result in results
            ]
        )

        return formatted_results
