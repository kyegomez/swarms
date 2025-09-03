import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.multi_agent_exec import run_agents_concurrently
from swarms.structs.swarm_id import swarm_id
from swarms.utils.formatter import formatter
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="majority_voting")


CONSENSUS_AGENT_PROMPT = """
Review the responses from all agents above. For each agent (referenced by their name), 
provide a thorough, objective evaluation of their contribution to the task. 
Compare and contrast the responses, highlighting strengths, weaknesses, and unique perspectives. 
Determine which response(s) best address the task overall, and explain your reasoning clearly. 
If possible, provide a ranked list or clear recommendation for the best response(s) based on the quality, 
relevance, and completeness of the answers. 
Be fair, detailed, and unbiased in your analysis, regardless of the topic.
"""


class MajorityVoting:
    """
    A multi-loop majority voting system for agents that enables iterative consensus building.

    This system allows agents to run multiple loops where each subsequent loop considers
    the previous consensus, enabling agents to refine their responses and build towards
    a more robust final consensus. The system maintains conversation history across
    all loops and provides methods to analyze the evolution of consensus over time.

    Key Features:
    - Multi-loop consensus building with configurable loop count
    - Agent memory retention across loops
    - Comprehensive consensus history tracking
    - Flexible output formats (string, dict, list)
    - Loop-by-loop analysis capabilities
    """

    def __init__(
        self,
        id: str = swarm_id(),
        name: str = "MajorityVoting",
        description: str = "A multi-loop majority voting system for agents",
        agents: List[Agent] = None,
        consensus_agent: Optional[Agent] = None,
        autosave: bool = False,
        verbose: bool = False,
        max_loops: int = 1,
        output_type: OutputType = "dict",
        consensus_agent_prompt: str = CONSENSUS_AGENT_PROMPT,
        *args,
        **kwargs,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.agents = agents
        self.consensus_agent = consensus_agent
        self.autosave = autosave
        self.verbose = verbose
        self.max_loops = max_loops
        self.output_type = output_type
        self.consensus_agent_prompt = consensus_agent_prompt

        self.conversation = Conversation(
            time_enabled=False, *args, **kwargs
        )

        self.initialize_majority_voting()

    def initialize_majority_voting(self):

        if self.agents is None:
            raise ValueError("Agents list is empty")

        # Log the agents
        formatter.print_panel(
            f"Initializing majority voting system\nNumber of agents: {len(self.agents)}\nAgents: {', '.join(agent.agent_name for agent in self.agents)}",
            title="Majority Voting",
        )

        if self.consensus_agent is None:
            # if no consensus agent is provided, use the last agent
            self.consensus_agent = self.agents[-1]

    def run(self, task: str, *args, **kwargs) -> List[Any]:
        """
        Runs the majority voting system with multi-loop functionality and returns the majority vote.

        Args:
            task (str): The task to be performed by the agents.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Any]: The majority vote.

        """

        self.conversation.add(
            role="user",
            content=task,
        )

        for i in range(self.max_loops):
            output = run_agents_concurrently(
                agents=self.agents,
                task=self.conversation.get_str(),
                max_workers=os.cpu_count(),
            )

            for agent, output in zip(self.agents, output):
                self.conversation.add(
                    role=agent.agent_name,
                    content=output,
                )

            # Now run the consensus agent
            consensus_output = self.consensus_agent.run(
                task=(
                    f"History: {self.conversation.get_str()} \n\n {self.consensus_agent_prompt}"
                ),
            )

            self.conversation.add(
                role=self.consensus_agent.agent_name,
                content=consensus_output,
            )

        return history_output_formatter(
            conversation=self.conversation,
            type=self.output_type,
        )

    def batch_run(
        self, tasks: List[str], *args, **kwargs
    ) -> List[Any]:
        """
        Runs the majority voting system in batch mode.

        Args:
            tasks (List[str]): List of tasks to be performed by the agents.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Any]: List of majority votes for each task.
        """
        return [self.run(task, *args, **kwargs) for task in tasks]

    def run_concurrently(
        self, tasks: List[str], *args, **kwargs
    ) -> List[Any]:
        """
        Runs the majority voting system concurrently.

        Args:
            tasks (List[str]): List of tasks to be performed by the agents.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Any]: List of majority votes for each task.
        """
        with ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            futures = [
                executor.submit(self.run, task, *args, **kwargs)
                for task in tasks
            ]
            return [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]
