import concurrent.futures
import os
import uuid
from typing import List, Optional

from swarms.prompts.ag_prompt import AGGREGATOR_SYSTEM_PROMPT_MAIN
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.ma_utils import list_all_agents
from swarms.structs.multi_agent_exec import run_agents_concurrently
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="mixture_of_agents")


class MixtureOfAgents:
    """
    A class to manage and run a mixture of agents, aggregating their responses.
    """

    def __init__(
        self,
        id: str = str(uuid.uuid4()),
        name: str = "MixtureOfAgents",
        description: str = "A class to run a mixture of agents and aggregate their responses.",
        agents: List[Agent] = None,
        aggregator_agent: Agent = None,
        aggregator_system_prompt: str = AGGREGATOR_SYSTEM_PROMPT_MAIN,
        layers: int = 3,
        max_loops: int = 1,
        output_type: OutputType = "final",
        aggregator_model_name: str = "claude-sonnet-4-20250514",
    ) -> None:
        """
        Initialize the Mixture of Agents class with agents and configuration.

        Args:
            name (str, optional): The name of the mixture of agents. Defaults to "MixtureOfAgents".
            description (str, optional): A description of the mixture of agents. Defaults to "A class to run a mixture of agents and aggregate their responses.".
            agents (List[Agent], optional): A list of reference agents to be used in the mixture. Defaults to [].
            aggregator_agent (Agent, optional): The aggregator agent to be used in the mixture. Defaults to None.
            aggregator_system_prompt (str, optional): The system prompt for the aggregator agent. Defaults to "".
            layers (int, optional): The number of layers to process in the mixture. Defaults to 3.
        """
        self.name = name
        self.description = description
        self.agents = agents
        self.aggregator_agent = aggregator_agent
        self.aggregator_system_prompt = aggregator_system_prompt
        self.layers = layers
        self.max_loops = max_loops
        self.output_type = output_type
        self.aggregator_model_name = aggregator_model_name

        self.reliability_check()

        self.conversation = Conversation()

        list_all_agents(
            agents=self.agents,
            conversation=self.conversation,
            description=self.description,
            name=self.name,
            add_to_conversation=True,
        )

        if self.aggregator_agent is None:
            self.aggregator_agent = self.aggregator_agent_setup()

    def reliability_check(self) -> None:
        """
        Performs a reliability check on the Mixture of Agents class.
        """
        logger.info(
            "Checking the reliability of the Mixture of Agents class."
        )

        if len(self.agents) == 0:
            raise ValueError("No agents provided.")

        if not self.aggregator_system_prompt:
            raise ValueError("No aggregator system prompt provided.")

        if not self.layers:
            raise ValueError("No layers provided.")

        logger.info("Reliability check passed.")
        logger.info("Mixture of Agents class is ready for use.")

    def aggregator_agent_setup(self):
        return Agent(
            agent_name="Aggregator Agent",
            agent_description="An agent that aggregates the responses of the other agents.",
            system_prompt=self.aggregator_system_prompt,
            model_name=self.aggregator_model_name,
            temperature=0.5,
            max_loops=1,
            output_type="str-all-except-first",
            dynamic_context_window=True,
        )

    def step(
        self,
        task: str,
        img: Optional[str] = None,
    ):
        # # Run agents concurrently
        # with concurrent.futures.ThreadPoolExecutor(
        #     max_workers=os.cpu_count()
        # ) as executor:
        #     # Submit all agent tasks and store with their index
        #     future_to_agent = {
        #         executor.submit(
        #             agent.run, task=task, img=img, imgs=imgs
        #         ): agent
        #         for agent in self.agents
        #     }

        #     # Collect results and add to conversation in completion order
        #     for future in concurrent.futures.as_completed(
        #         future_to_agent
        #     ):
        #         agent = future_to_agent[future]
        #         output = future.result()
        #         self.conversation.add(role=agent.name, content=output)
        agent_outputs = run_agents_concurrently(
            agents=self.agents,
            task=task,
            img=img,
            return_agent_output_dict=True,
        )

        return agent_outputs

    def _run(
        self,
        task: str,
        img: Optional[str] = None,
    ):

        # self.conversation.add(role="User", content=task)

        # for i in range(self.layers):
        #     out = self.step(
        #         task=self.conversation.get_str(), img=img, imgs=imgs
        #     )
        #     task = out

        # out = self.aggregator_agent.run(
        #     task=self.conversation.get_str()
        # )

        # self.conversation.add(
        #     role=self.aggregator_agent.agent_name, content=out
        # )

        # out = history_output_formatter(
        #     conversation=self.conversation, type=self.output_type
        # )

        # return out

        self.conversation.add(role="User", content=task)

        full_context = self.conversation.get_str()

        for i in range(self.layers):
            # Pass the full context/history string to the step method
            step_output = self.step(task=full_context, img=img)

            # Log each agent's output with full context awareness
            for agent_name, agent_output in step_output.items():
                self.conversation.add(
                    role=agent_name, content=agent_output
                )

            # Update the full_context with the latest conversation history
            full_context = self.conversation.get_str()

        aggregator_output = self.aggregator_agent.run(
            task=self.conversation.get_str()
        )

        self.conversation.add(
            role=self.aggregator_agent.agent_name,
            content=aggregator_output,
        )

        return history_output_formatter(
            conversation=self.conversation, type=self.output_type
        )

    def run(
        self,
        task: str,
        img: Optional[str] = None,
    ):
        try:
            return self._run(task=task, img=img)
        except Exception as e:
            logger.error(f"Error running Mixture of Agents: {e}")
            return f"Error: {e}"

    def run_batched(self, tasks: List[str]) -> List[str]:
        """
        Run the mixture of agents for a batch of tasks.

        Args:
            tasks (List[str]): A list of tasks for the mixture of agents.

        Returns:
            List[str]: A list of responses from the mixture of agents.
        """
        return [self.run(task) for task in tasks]

    def run_concurrently(self, tasks: List[str]) -> List[str]:
        """
        Run the mixture of agents for a batch of tasks concurrently.
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            futures = [
                executor.submit(self.run, task) for task in tasks
            ]
            return [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]
