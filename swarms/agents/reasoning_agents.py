"""
ReasoningAgentRouter: A flexible router for advanced reasoning agent swarms.

This module provides the ReasoningAgentRouter class, which enables dynamic selection and instantiation
of various advanced reasoning agent types (swarms) for complex problem-solving tasks. It supports
multiple reasoning strategies, including self-consistency, collaborative duo agents, iterative
reflection, knowledge prompting, and agent judging.

Key Features:
- Unified interface for multiple agent types (see `agent_types`)
- Caching of agent instances for efficiency and memory management
- Extensible factory-based architecture for easy addition of new agent types
- Batch and single-task execution
- Customizable agent configuration (model, prompt, memory, etc.)

Supported Agent Types:
    - "reasoning-duo" / "reasoning-agent": Dual collaborative agent system
    - "self-consistency" / "consistency-agent": Multiple independent solutions with consensus
    - "ire" / "ire-agent": Iterative Reflective Expansion agent
    - "ReflexionAgent": Reflexion agent with memory
    - "GKPAgent": Generated Knowledge Prompting agent
    - "AgentJudge": Agent judge for evaluation/critique

Example usage:
    >>> router = ReasoningAgentRouter(swarm_type="self-consistency", num_samples=3)
    >>> result = router.run("What is the capital of France?")
    >>> print(result)

    >>> # Batch mode
    >>> results = router.batched_run(["2+2?", "3+3?"])
    >>> print(results)

"""

import traceback
from typing import (
    List,
    Literal,
    Optional,
)

from swarms.agents.consistency_agent import SelfConsistencyAgent
from swarms.agents.flexion_agent import ReflexionAgent
from swarms.agents.gkp_agent import GKPAgent
from swarms.agents.i_agent import (
    IterativeReflectiveExpansion as IREAgent,
)
from swarms.agents.reasoning_duo import ReasoningDuo
from swarms.utils.output_types import OutputType
from swarms.agents.agent_judge import AgentJudge

#: Supported agent type literals for ReasoningAgentRouter
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


class ReasoningAgentExecutorError(Exception):
    """
    Exception raised when an error occurs during the execution of a reasoning agent.
    """

    pass


class ReasoningAgentInitializationError(Exception):
    """
    Exception raised when an error occurs during the initialization of a reasoning agent.
    """

    pass


class ReasoningAgentRouter:
    """
    A router for advanced reasoning agent swarms.

    The ReasoningAgentRouter enables dynamic selection, instantiation, and caching of various
    reasoning agent types ("swarms") for flexible, robust, and scalable problem-solving.

    Args:
        agent_name (str): Name identifier for the agent instance.
        description (str): Description of the agent's capabilities.
        model_name (str): The underlying language model to use.
        system_prompt (str): System prompt for the agent.
        max_loops (int): Maximum number of reasoning loops.
        swarm_type (agent_types): Type of reasoning swarm to use.
        num_samples (int): Number of samples for self-consistency or iterations.
        output_type (OutputType): Format of the output.
        num_knowledge_items (int): Number of knowledge items for GKP agent.
        memory_capacity (int): Memory capacity for agents that support it.
        eval (bool): Enable evaluation mode for self-consistency.
        random_models_on (bool): Enable random model selection for diversity.
        majority_voting_prompt (Optional[str]): Custom prompt for majority voting.

    Example:
        >>> router = ReasoningAgentRouter(swarm_type="reasoning-duo")
        >>> result = router.run("Explain quantum entanglement.")
        >>> print(result)
    """

    def __init__(
        self,
        agent_name: str = "reasoning_agent",
        description: str = "A reasoning agent that can answer questions and help with tasks.",
        model_name: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful assistant that can answer questions and help with tasks.",
        max_loops: int = 1,
        swarm_type: agent_types = "reasoning-duo",
        num_samples: int = 1,
        output_type: OutputType = "dict-all-except-first",
        num_knowledge_items: int = 6,
        memory_capacity: int = 6,
        eval: bool = False,
        random_models_on: bool = False,
        majority_voting_prompt: Optional[str] = None,
        reasoning_model_name: Optional[
            str
        ] = "claude-3-5-sonnet-20240620",
    ):
        """
        Initialize the ReasoningAgentRouter with the specified configuration.

        See class docstring for parameter details.
        """
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
        self.eval = eval
        self.random_models_on = random_models_on
        self.majority_voting_prompt = majority_voting_prompt
        self.reasoning_model_name = reasoning_model_name

        self.reliability_check()

    def reliability_check(self):

        if self.max_loops == 0:
            raise ReasoningAgentInitializationError(
                "ReasoningAgentRouter Error: Max loops must be greater than 0"
            )

        if self.model_name == "" or self.model_name is None:
            raise ReasoningAgentInitializationError(
                "ReasoningAgentRouter Error: Model name must be provided"
            )

        if self.swarm_type == "" or self.swarm_type is None:
            raise ReasoningAgentInitializationError(
                "ReasoningAgentRouter Error: Swarm type must be provided. This is the type of reasoning agent you want to use. For example, 'reasoning-duo' for a reasoning duo agent, 'self-consistency' for a self-consistency agent, 'ire' for an iterative reflective expansion agent, 'reasoning-agent' for a reasoning agent, 'consistency-agent' for a consistency agent, 'ire-agent' for an iterative reflective expansion agent, 'ReflexionAgent' for a reflexion agent, 'GKPAgent' for a generated knowledge prompting agent, 'AgentJudge' for an agent judge."
            )

        # Initialize the factory mapping dictionary
        self.agent_factories = self._initialize_agent_factories()

    def _initialize_agent_factories(self) -> None:
        """
        Initialize the agent factory mapping dictionary, mapping various agent types to their respective creation functions.

        This method replaces the original if-elif chain, making the code more maintainable and extensible.
        """
        agent_factories = {
            "reasoning-duo": self._create_reasoning_duo,
            "reasoning-agent": self._create_reasoning_duo,
            "self-consistency": self._create_consistency_agent,
            "consistency-agent": self._create_consistency_agent,
            "ire": self._create_ire_agent,
            "ire-agent": self._create_ire_agent,
            "AgentJudge": self._create_agent_judge,
            "ReflexionAgent": self._create_reflexion_agent,
            "GKPAgent": self._create_gkp_agent,
        }

        return agent_factories

    def _create_reasoning_duo(self):
        """
        Create an agent instance for the ReasoningDuo type.

        Returns:
            ReasoningDuo: An instance of the ReasoningDuo agent.
        """
        return ReasoningDuo(
            agent_name=self.agent_name,
            agent_description=self.description,
            model_name=[self.model_name, self.model_name],
            system_prompt=self.system_prompt,
            output_type=self.output_type,
            reasoning_model_name=self.reasoning_model_name,
            max_loops=self.max_loops,
        )

    def _create_consistency_agent(self):
        """
        Create an agent instance for the SelfConsistencyAgent type.

        Returns:
            SelfConsistencyAgent: An instance of the SelfConsistencyAgent.
        """
        return SelfConsistencyAgent(
            name=self.agent_name,
            description=self.description,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            max_loops=self.max_loops,
            num_samples=self.num_samples,
            output_type=self.output_type,
            eval=self.eval,
            random_models_on=self.random_models_on,
            majority_voting_prompt=self.majority_voting_prompt,
        )

    def _create_ire_agent(self):
        """
        Create an agent instance for the IREAgent type.

        Returns:
            IREAgent: An instance of the IterativeReflectiveExpansion agent.
        """
        return IREAgent(
            agent_name=self.agent_name,
            description=self.description,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            max_loops=self.max_loops,
            max_iterations=self.num_samples,
            output_type=self.output_type,
        )

    def _create_agent_judge(self):
        """
        Create an agent instance for the AgentJudge type.

        Returns:
            AgentJudge: An instance of the AgentJudge agent.
        """
        return AgentJudge(
            agent_name=self.agent_name,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            max_loops=self.max_loops,
        )

    def _create_reflexion_agent(self):
        """
        Create an agent instance for the ReflexionAgent type.

        Returns:
            ReflexionAgent: An instance of the ReflexionAgent.
        """
        return ReflexionAgent(
            agent_name=self.agent_name,
            system_prompt=self.system_prompt,
            model_name=self.model_name,
            max_loops=self.max_loops,
            memory_capacity=self.memory_capacity,
        )

    def _create_gkp_agent(self):
        """
        Create an agent instance for the GKPAgent type.

        Returns:
            GKPAgent: An instance of the GKPAgent.
        """
        return GKPAgent(
            agent_name=self.agent_name,
            model_name=self.model_name,
            num_knowledge_items=self.num_knowledge_items,
        )

    def select_swarm(self):
        """
        Select and initialize the appropriate reasoning swarm based on the specified swarm type.

        Returns:
            The selected reasoning swarm instance.

        Raises:
            ValueError: If the specified swarm type is invalid.
        """
        try:
            if self.swarm_type in self.agent_factories:
                return self.agent_factories[self.swarm_type]()
            else:
                raise ReasoningAgentInitializationError(
                    f"ReasoningAgentRouter Error: Invalid swarm type: {self.swarm_type}"
                )
        except Exception as e:
            raise ReasoningAgentInitializationError(
                f"ReasoningAgentRouter Error: {e} Traceback: {traceback.format_exc()} If the error persists, please check the agent's configuration and try again. If you would like support book a call with our team at https://cal.com/swarms"
            )

    def run(self, task: str, *args, **kwargs):
        """
        Execute the reasoning process of the selected swarm on a given task.

        Args:
            task (str): The task or question to be processed by the reasoning agent.
            *args: Additional positional arguments for the agent's run method.
            **kwargs: Additional keyword arguments for the agent's run method.

        Returns:
            The result of the reasoning process (format depends on agent and output_type).
        """
        try:
            swarm = self.select_swarm()
            return swarm.run(task=task, *args, **kwargs)
        except Exception as e:
            raise ReasoningAgentExecutorError(
                f"ReasoningAgentRouter Error: {e} Traceback: {traceback.format_exc()} If the error persists, please check the agent's configuration and try again. If you would like support book a call with our team at https://cal.com/swarms"
            )

    def batched_run(self, tasks: List[str], *args, **kwargs):
        """
        Execute the reasoning process on a batch of tasks.

        Args:
            tasks (List[str]): The list of tasks to process.
            *args: Additional positional arguments for the agent's run method.
            **kwargs: Additional keyword arguments for the agent's run method.

        Returns:
            A list of reasoning process results for each task.
        """
        results = []
        for task in tasks:
            results.append(self.run(task, *args, **kwargs))
        return results
