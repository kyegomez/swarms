import asyncio
import concurrent.futures
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.multi_agent_exec import run_agents_concurrently
from swarms.structs.output_types import OutputType
from swarms.utils.formatter import formatter
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="majority_voting")


def extract_last_python_code_block(text):
    """
    Extracts the last Python code block from the given text.

    Args:
        text (str): The text to search for Python code blocks.

    Returns:
        str or None: The last Python code block found in the text, or None if no code block is found.
    """
    # The regular expression pattern for Python code blocks
    pattern = r"```[pP]ython(.*?)```"

    # Find all matches in the text
    matches = re.findall(pattern, text, re.DOTALL)

    # If there are matches, return the last one
    if matches:
        return matches[-1].strip()
    else:
        return None


def parse_code_completion(agent_response, question):
    """
    Parses the code completion response from the agent and extracts the last Python code block.

    Args:
        agent_response (str): The response from the agent.
        question (str): The original question.

    Returns:
        tuple: A tuple containing the parsed Python code and a boolean indicating success.
    """
    python_code = extract_last_python_code_block(agent_response)
    if python_code is None:
        if agent_response.count("impl]") == 0:
            python_code = agent_response
        else:
            python_code_lines = agent_response.split("\n")
            python_code = ""
            in_func = False
            for line in python_code_lines:
                if in_func:
                    python_code += line + "\n"
                if "impl]" in line:
                    in_func = True
    if python_code.count("def") == 0:
        python_code = question + python_code
    return python_code, True


def most_frequent(
    clist: list,
    cmp_func: callable = None,
):
    """
    Finds the most frequent element in a list based on a comparison function.

    Args:
        clist (list): The list of elements to search.
        cmp_func (function, optional): The comparison function used to determine the frequency of elements.
            If not provided, the default comparison function is used.

    Returns:
        tuple: A tuple containing the most frequent element and its frequency.
    """
    counter = 0
    num = clist[0]

    for i in clist:
        current_frequency = sum(cmp_func(i, item) for item in clist)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num, counter


def majority_voting(answers: List[str]):
    """
    Performs majority voting on a list of answers and returns the most common answer.

    Args:
        answers (list): A list of answers.

    Returns:
        The most common answer in the list.
    """
    counter = Counter(answers)
    if counter:
        answer = counter.most_common(1)[0][0]
    else:
        answer = "I don't know"

    return answer


class MajorityVoting:
    """
    Class representing a majority voting system for agents.

    Args:
        agents (list): A list of agents to be used in the majority voting system.
        output_parser (function, optional): A function used to parse the output of the agents.
            If not provided, the default majority voting function is used.
        autosave (bool, optional): A boolean indicating whether to autosave the conversation to a file.
        verbose (bool, optional): A boolean indicating whether to enable verbose logging.
    Examples:
        >>> from swarms.structs.agent import Agent
        >>> from swarms.structs.majority_voting import MajorityVoting
        >>> agents = [
        ...     Agent("GPT-3"),
        ...     Agent("Codex"),
        ...     Agent("Tabnine"),
        ... ]
        >>> majority_voting = MajorityVoting(agents)
        >>> majority_voting.run("What is the capital of France?")
        'Paris'

    """

    def __init__(
        self,
        name: str = "MajorityVoting",
        description: str = "A majority voting system for agents",
        agents: List[Agent] = [],
        output_parser: Optional[Callable] = majority_voting,
        consensus_agent: Optional[Agent] = None,
        autosave: bool = False,
        verbose: bool = False,
        max_loops: int = 1,
        output_type: OutputType = "dict",
        *args,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.agents = agents
        self.output_parser = output_parser
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

    async def run_async(
        self, tasks: List[str], *args, **kwargs
    ) -> List[Any]:
        """
        Runs the majority voting system concurrently using asyncio.

        Args:
            tasks (List[str]): List of tasks to be performed by the agents.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Any]: List of majority votes for each task.
        """
        return await asyncio.gather(
            *[self.run(task, *args, **kwargs) for task in tasks]
        )
