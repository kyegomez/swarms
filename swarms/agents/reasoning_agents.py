from typing import List, Literal

from swarms.agents.consistency_agent import SelfConsistencyAgent
from swarms.agents.flexion_agent import ReflexionAgent
from swarms.agents.gkp_agent import GKPAgent
from swarms.agents.i_agent import (
    IterativeReflectiveExpansion as IREAgent,
)
from swarms.agents.reasoning_duo import ReasoningDuo
from swarms.structs.output_types import OutputType
from swarms.agents.agent_judge import AgentJudge

agent_types = Literal[
    "reasoning-duo",
    "self-consistency",
    "ire",
    "reasoning-agent",
    "consistency-agent",
    "ire-agent",
    "ReflexionAgent",
    "GKPAgent",
    "AgentJudge",
]


class ReasoningAgentRouter:
    """
    A Reasoning Agent that can answer questions and assist with various tasks using different reasoning strategies.

    Attributes:
        agent_name (str): The name of the agent.
        description (str): A brief description of the agent's capabilities.
        model_name (str): The name of the model used for reasoning.
        system_prompt (str): The prompt that guides the agent's reasoning process.
        max_loops (int): The maximum number of loops for the reasoning process.
        swarm_type (agent_types): The type of reasoning swarm to use (e.g., reasoning duo, self-consistency, IRE).
        num_samples (int): The number of samples to generate for self-consistency agents.
        output_type (OutputType): The format of the output (e.g., dict, list).
    """

    def __init__(
        self,
        agent_name: str = "reasoning_agent",
        description: str = "A reasoning agent that can answer questions and help with tasks.",
        model_name: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful assistant that can answer questions and help with tasks.",
        max_loops: int = 1,
        swarm_type: agent_types = "reasoning_duo",
        num_samples: int = 1,
        output_type: OutputType = "dict",
        num_knowledge_items: int = 6,
        memory_capacity: int = 6,
    ):
        self.agent_name = agent_name
        self.description = description
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_loops = max_loops
        self.swarm_type = swarm_type
        self.num_samples = num_samples
        self.output_type = output_type
        self.num_knowledge_items = num_knowledge_items
        self.memory_capacity = memory_capacity

    def select_swarm(self):
        """
        Selects and initializes the appropriate reasoning swarm based on the specified swarm type.

        Returns:
            An instance of the selected reasoning swarm.
        """
        if (
            self.swarm_type == "reasoning-duo"
            or self.swarm_type == "reasoning-agent"
        ):
            return ReasoningDuo(
                agent_name=self.agent_name,
                agent_description=self.description,
                model_name=[self.model_name, self.model_name],
                system_prompt=self.system_prompt,
                output_type=self.output_type,
            )

        elif (
            self.swarm_type == "self-consistency"
            or self.swarm_type == "consistency-agent"
        ):
            return SelfConsistencyAgent(
                agent_name=self.agent_name,
                description=self.description,
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                max_loops=self.max_loops,
                num_samples=self.num_samples,
                output_type=self.output_type,
            )

        elif (
            self.swarm_type == "ire" or self.swarm_type == "ire-agent"
        ):
            return IREAgent(
                agent_name=self.agent_name,
                description=self.description,
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                max_loops=self.max_loops,
                max_iterations=self.num_samples,
                output_type=self.output_type,
            )

        elif self.swarm_type == "AgentJudge":
            return AgentJudge(
                agent_name=self.agent_name,
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                max_loops=self.max_loops,
            )

        elif self.swarm_type == "ReflexionAgent":
            return ReflexionAgent(
                agent_name=self.agent_name,
                system_prompt=self.system_prompt,
                model_name=self.model_name,
                max_loops=self.max_loops,
            )

        elif self.swarm_type == "GKPAgent":
            return GKPAgent(
                agent_name=self.agent_name,
                model_name=self.model_name,
                num_knowledge_items=self.num_knowledge_items,
            )
        else:
            raise ValueError(f"Invalid swarm type: {self.swarm_type}")

    def run(self, task: str, *args, **kwargs):
        """
        Executes the selected swarm's reasoning process on the given task.

        Args:
            task (str): The task or question to be processed by the reasoning agent.

        Returns:
            The result of the reasoning process.
        """
        swarm = self.select_swarm()
        return swarm.run(task=task)

    def batched_run(self, tasks: List[str], *args, **kwargs):
        """
        Executes the reasoning process on a batch of tasks.

        Args:
            tasks (List[str]): A list of tasks to be processed.

        Returns:
            List of results from the reasoning process for each task.
        """
        results = []
        for task in tasks:
            results.append(self.run(task, *args, **kwargs))
        return results
