from swarms.structs.flow import Flow


# Define a selection function
def select_speaker(step: int, agents) -> int:
    # This function selects the speaker in a round-robin fashion
    return step % len(agents)


class MultiAgentDebate:
    """
    MultiAgentDebate


    Args:
        agents: Flow
        selection_func: callable
        max_iters: int

    Usage:
    >>> from swarms import MultiAgentDebate
    >>> from swarms.structs.flow import Flow
    >>> agents = Flow()
    >>> agents.append(lambda x: x)
    >>> agents.append(lambda x: x)
    >>> agents.append(lambda x: x)

    """

    def __init__(
        self,
        agents: Flow,
        selection_func: callable = select_speaker,
        max_iters: int = None,
    ):
        self.agents = agents
        self.selection_func = selection_func
        self.max_iters = max_iters

    def inject_agent(self, agent):
        """Injects an agent into the debate"""
        self.agents.append(agent)

    def run(
        self,
        task: str,
    ):
        """
        MultiAgentDebate

        Args:
            task: str

        Returns:
            results: list

        """
        results = []
        for i in range(self.max_iters or len(self.agents)):
            speaker_idx = self.selection_func(i, self.agents)
            speaker = self.agents[speaker_idx]
            response = speaker(task)
            results.append({"response": response})
        return results

    def update_task(self, task: str):
        """Update the task"""
        self.task = task

    def format_results(self, results):
        """Format the results"""
        formatted_results = "\n".join(
            [f"Agent responded: {result['response']}" for result in results]
        )

        return formatted_results
