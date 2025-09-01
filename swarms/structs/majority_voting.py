import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.multi_agent_exec import run_agents_concurrently
from swarms.structs.swarm_id import swarm_id
from swarms.utils.formatter import formatter
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="majority_voting")


class MajorityVoting:

    def __init__(
        self,
        id: str = swarm_id(),
        name: str = "MajorityVoting",
        description: str = "A majority voting system for agents",
        agents: List[Agent] = None,
        consensus_agent: Optional[Agent] = None,
        autosave: bool = False,
        verbose: bool = False,
        max_loops: int = 1,
        output_type: OutputType = "dict",
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
        Runs the majority voting system and returns the majority vote.

        Args:
            task (str): The task to be performed by the agents.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Any]: The majority vote.

        """
        results = run_agents_concurrently(
            self.agents, task, max_workers=os.cpu_count()
        )

        # Add responses to conversation and log them
        for agent, response in zip(self.agents, results):

            response = (
                response if isinstance(response, list) else [response]
            )
            self.conversation.add(agent.agent_name, response)

        responses = self.conversation.return_history_as_string()
        # print(responses)

        prompt = f"""Conduct a detailed majority voting analysis on the following conversation:
        {responses}

        Between the following agents: {[agent.agent_name for agent in self.agents]}

        Please:
        1. Identify the most common answer/recommendation across all agents
        2. Analyze any major disparities or contrasting viewpoints between agents
        3. Highlight key areas of consensus and disagreement
        4. Evaluate the strength of the majority opinion
        5. Note any unique insights from minority viewpoints
        6. Provide a final synthesized recommendation based on the majority consensus

        Focus on finding clear patterns while being mindful of important nuances in the responses.
        """

        # If an output parser is provided, parse the responses
        if self.consensus_agent is not None:
            majority_vote = self.consensus_agent.run(prompt)

            self.conversation.add(
                self.consensus_agent.agent_name, majority_vote
            )
        else:
            # fetch the last agent
            majority_vote = self.agents[-1].run(prompt)

            self.conversation.add(
                self.agents[-1].agent_name, majority_vote
            )

        # Return the majority vote
        # return self.conversation.return_history_as_string()
        if self.output_type == "str":
            return self.conversation.get_str()
        elif self.output_type == "dict":
            return self.conversation.return_messages_as_dictionary()
        elif self.output_type == "list":
            return self.conversation.return_messages_as_list()
        else:
            return self.conversation.return_history_as_string()

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
