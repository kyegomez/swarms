from typing import List, Optional

from loguru import logger

from swarms.prompts.reasoning_prompt import REASONING_PROMPT
from swarms.structs.agent import Agent
from swarms.utils.output_types import OutputType
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
import uuid


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
        id: str = str(uuid.uuid4()),
        agent_name: str = "reasoning-agent-01",
        agent_description: str = "A highly intelligent and thoughtful AI designed to provide accurate and well-reasoned answers to the user's questions.",
        model_name: str = "gpt-4o-mini",
        description: str = "A highly intelligent and thoughtful AI designed to provide accurate and well-reasoned answers to the user's questions.",
        model_names: list[str] = ["gpt-4o-mini", "gpt-4o"],
        system_prompt: str = "You are a helpful assistant that can answer questions and help with tasks.",
        output_type: OutputType = "dict-all-except-first",
        reasoning_model_name: Optional[
            str
        ] = "claude-3-5-sonnet-20240620",
        max_loops: int = 1,
        *args,
        **kwargs,
    ):
        self.id = id
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.model_name = model_name
        self.description = description
        self.output_type = output_type
        self.reasoning_model_name = reasoning_model_name
        self.max_loops = max_loops

        if self.reasoning_model_name is None:
            self.reasoning_model_name = model_names[0]

        self.conversation = Conversation()

        self.reasoning_agent = Agent(
            agent_name=self.agent_name,
            description=self.agent_description,
            system_prompt=REASONING_PROMPT,
            max_loops=1,
            model_name=self.reasoning_model_name,
            dynamic_temperature_enabled=True,
            *args,
            **kwargs,
        )

        self.main_agent = Agent(
            agent_name=self.agent_name,
            description=self.agent_description,
            system_prompt=system_prompt,
            max_loops=1,
            model_name=model_names[1],
            dynamic_temperature_enabled=True,
            *args,
            **kwargs,
        )

    def step(self, task: str, img: Optional[str] = None):
        """
        Executes one step of reasoning and main agent processing.

        Args:
            task (str): The task to be processed.
            img (Optional[str]): Optional image input.
        """
        # For reasoning agent, use the current task (which may include conversation context)
        output_reasoner = self.reasoning_agent.run(task, img=img)

        self.conversation.add(
            role=self.reasoning_agent.agent_name,
            content=output_reasoner,
        )

        # For main agent, always use the full conversation context
        output_main = self.main_agent.run(
            task=self.conversation.get_str(), img=img
        )

        self.conversation.add(
            role=self.main_agent.agent_name, content=output_main
        )

    def run(self, task: str, img: Optional[str] = None):
        """
        Executes the reasoning and main agents on the provided task.

        Args:
            task (str): The task to be processed by the agents.
            img (Optional[str]): Optional image input.

        Returns:
            str: The output from the main agent after processing the task.
        """
        logger.info(
            f"Running task: {task} with max_loops: {self.max_loops}"
        )
        self.conversation.add(role="user", content=task)

        for loop_iteration in range(self.max_loops):
            logger.info(
                f"Loop iteration {loop_iteration + 1}/{self.max_loops}"
            )

            if loop_iteration == 0:
                # First iteration: use original task
                current_task = task
            else:
                # Subsequent iterations: use task with context of previous reasoning
                current_task = f"Continue reasoning and refining your analysis. Original task: {task}\n\nPrevious conversation context:\n{self.conversation.get_str()}"

            self.step(task=current_task, img=img)

        return history_output_formatter(
            self.conversation, self.output_type
        )

    def batched_run(
        self, tasks: List[str], imgs: Optional[List[str]] = None
    ):
        """
        Executes the run method for a list of tasks.

        Args:
            tasks (list[str]): A list of tasks to be processed.
            imgs (Optional[List[str]]): Optional list of images corresponding to tasks.

        Returns:
            list: A list of outputs from the main agent for each task.
        """
        outputs = []

        # Handle case where imgs is None
        if imgs is None:
            imgs = [None] * len(tasks)

        for task, img in zip(tasks, imgs):
            logger.info(f"Processing task: {task}")
            outputs.append(self.run(task, img=img))
        return outputs
