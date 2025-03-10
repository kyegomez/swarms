from typing import List

from loguru import logger

from swarms.prompts.reasoning_prompt import REASONING_PROMPT
from swarms.structs.agent import Agent
from swarms.structs.output_types import OutputType
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


class ReasoningDuo:
    """
    ReasoningDuo is a class that encapsulates the functionality of two agents: a reasoning agent and a main agent.

    Attributes:
        model_name (str): The name of the model used for the reasoning agent.
        description (str): A description of the reasoning agent.
        model_names (list[str]): A list of model names for the agents.
        system_prompt (str): The system prompt for the main agent.
        reasoning_agent (Agent): An instance of the Agent class for reasoning tasks.
        main_agent (Agent): An instance of the Agent class for main tasks.
    """

    def __init__(
        self,
        agent_name: str = "reasoning-agent-01",
        agent_description: str = "A highly intelligent and thoughtful AI designed to provide accurate and well-reasoned answers to the user's questions.",
        model_name: str = "gpt-4o-mini",
        description: str = "A highly intelligent and thoughtful AI designed to provide accurate and well-reasoned answers to the user's questions.",
        model_names: list[str] = ["gpt-4o-mini", "gpt-4o"],
        system_prompt: str = "You are a helpful assistant that can answer questions and help with tasks.",
        output_type: OutputType = "dict",
    ):
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.model_name = model_name
        self.description = description
        self.output_type = output_type
        self.reasoning_agent = Agent(
            agent_name="Your",
            description="A highly intelligent and thoughtful AI designed to provide accurate and well-reasoned answers to the user's questions.",
            system_prompt=REASONING_PROMPT,
            max_loops=1,
            model_name=model_names[0],
            dynamic_temperature_enabled=True,
        )

        self.main_agent = Agent(
            agent_name=self.agent_name,
            description=self.agent_description,
            system_prompt=system_prompt,
            max_loops=1,
            model_name=model_names[1],
            dynamic_temperature_enabled=True,
        )

        self.conversation = Conversation()

    def run(self, task: str):
        """
        Executes the reasoning and main agents on the provided task.

        Args:
            task (str): The task to be processed by the agents.

        Returns:
            str: The output from the main agent after processing the task.
        """
        logger.info(f"Running task: {task}")

        self.conversation.add(role="user", content=task)

        output_reasoner = self.reasoning_agent.run(task)

        self.conversation.add(
            role=self.reasoning_agent.agent_name,
            content=output_reasoner,
        )

        prompt = f"Task: {task} \n\n Your thoughts: {output_reasoner}"

        output_main = self.main_agent.run(prompt)

        self.conversation.add(
            role=self.main_agent.agent_name, content=output_main
        )

        return history_output_formatter(
            self.conversation, self.output_type
        )

    def batched_run(self, tasks: List[str]):
        """
        Executes the run method for a list of tasks.

        Args:
            tasks (list[str]): A list of tasks to be processed.

        Returns:
            list: A list of outputs from the main agent for each task.
        """
        outputs = []
        for task in tasks:
            logger.info(f"Processing task: {task}")
            outputs.append(self.run(task))
        return outputs
