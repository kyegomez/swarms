from typing import List, Optional

from loguru import logger

from swarms.structs.execution_utils import batched_run
from swarms.prompts.reasoning_prompt import REASONING_PROMPT
from swarms.structs.agent import Agent
from swarms.utils.output_types import OutputType
from swarms.structs.context_utils import (
    agent_answer,
    messages_for,
    split_last_turn,
)
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.generate_id import generate_id


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
        id: Optional[str] = None,
        agent_name: str = "reasoning-agent-01",
        agent_description: str = "A highly intelligent and thoughtful AI designed to provide accurate and well-reasoned answers to the user's questions.",
        model_name: str = "gpt-5.4",
        description: str = "A highly intelligent and thoughtful AI designed to provide accurate and well-reasoned answers to the user's questions.",
        model_names: list[str] = ["gpt-5.4", "gpt-5.4"],
        system_prompt: str = "You are a helpful assistant that can answer questions and help with tasks.",
        output_type: OutputType = "dict-all-except-first",
        reasoning_model_name: Optional[str] = "gpt-4o",
        max_loops: int = 1,
        *args,
        **kwargs,
    ):
        self.id = id or generate_id("reasoning-duo")
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

        # Distinct names: a shared one made both agents the same speaker in
        # the conversation, so neither could tell its own turns from the
        # other's.
        self.reasoning_agent = Agent(
            agent_name=f"{self.agent_name}-reasoning",
            description=self.agent_description,
            system_prompt=REASONING_PROMPT,
            max_loops=1,
            model_name=self.reasoning_model_name,
            dynamic_temperature_enabled=True,
            *args,
            **kwargs,
        )

        self.main_agent = Agent(
            agent_name=f"{self.agent_name}-main",
            description=self.agent_description,
            system_prompt=system_prompt,
            max_loops=1,
            model_name=model_names[1],
            dynamic_temperature_enabled=True,
            *args,
            **kwargs,
        )

    def _run_agent(
        self,
        agent,
        task: Optional[str] = None,
        img: Optional[str] = None,
    ) -> str:
        """Run one agent on the shared conversation and record its answer.

        The conversation is delivered as typed chat turns, so each agent reads
        its own prior output as ``assistant`` and its partner's as a labelled
        ``user`` turn instead of one flattened block.

        Args:
            agent: The agent to run.
            task (Optional[str]): A new instruction to append as this turn.
                When None, the newest turn already in the conversation is used.
            img (Optional[str]): Optional image input.

        Returns:
            str: The agent's answer.
        """
        if task is not None:
            self.conversation.add(role="user", content=task)

        prior, step_task = split_last_turn(
            messages_for(agent.agent_name, self.conversation)
        )
        response = agent.run(task=step_task, messages=prior, img=img)

        answer = agent_answer(agent, fallback=response)
        self.conversation.add(role=agent.agent_name, content=answer)
        return answer

    def step(self, task: str, img: Optional[str] = None):
        """
        Executes one step of reasoning and main agent processing.

        Args:
            task (str): The task to be processed.
            img (Optional[str]): Optional image input.
        """
        self._run_agent(self.reasoning_agent, task, img=img)
        self._run_agent(self.main_agent, img=img)

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
        # The task is appended by _run_agent on the first iteration; adding it
        # here too would give the reasoning agent the same turn twice.
        for loop_iteration in range(self.max_loops):
            logger.info(
                f"Loop iteration {loop_iteration + 1}/{self.max_loops}"
            )

            # Prior turns reach the agents as messages, so a later loop only
            # needs the new instruction rather than a re-rendered transcript.
            current_task = (
                task
                if loop_iteration == 0
                else "Continue reasoning and refining your analysis."
            )

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
            imgs (Optional[List[str]]): One image per task, paired by
                position. Must be the same length as ``tasks``.

        Returns:
            list: A list of outputs from the main agent for each task.

        Raises:
            ValueError: If ``imgs`` is given and is not one per task.
        """
        return batched_run(self.run, tasks, imgs=imgs)
