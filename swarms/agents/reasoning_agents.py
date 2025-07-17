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

See also:
    - docs/swarms/agents/reasoning_agent_router.md for detailed documentation and architecture diagrams.
    - consistency_example.py for a usage example with SelfConsistencyAgent.

"""

from typing import (
    List,
    Literal,
    Dict,
    Callable,
    Any,
    Tuple,
    Hashable,
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

    # Class variable to store cached agent instances
    _agent_cache: Dict[Tuple[Hashable, ...], Any] = {}

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

        # Initialize the factory mapping dictionary
        self._initialize_agent_factories()

    def _initialize_agent_factories(self) -> None:
        """
        Initialize the agent factory mapping dictionary, mapping various agent types to their respective creation functions.

        This method replaces the original if-elif chain, making the code more maintainable and extensible.
        """
        self.agent_factories: Dict[str, Callable[[], Any]] = {
            # ReasoningDuo factory method
            "reasoning-duo": self._create_reasoning_duo,
            "reasoning-agent": self._create_reasoning_duo,
            # SelfConsistencyAgent factory methods
            "self-consistency": self._create_consistency_agent,
            "consistency-agent": self._create_consistency_agent,
            # IREAgent factory methods
            "ire": self._create_ire_agent,
            "ire-agent": self._create_ire_agent,
            # Other agent type factory methods
            "AgentJudge": self._create_agent_judge,
            "ReflexionAgent": self._create_reflexion_agent,
            "GKPAgent": self._create_gkp_agent,
        }

    def _get_cache_key(self) -> Tuple[Hashable, ...]:
        """
        Generate a unique key for cache lookup.

        The key is based on all relevant configuration parameters of the agent.

        Returns:
            Tuple[Hashable, ...]: A hashable tuple to serve as the cache key.
        """
        return (
            self.swarm_type,
            self.agent_name,
            self.description,
            self.model_name,
            self.system_prompt,
            self.max_loops,
            self.num_samples,
            self.output_type,
            self.num_knowledge_items,
            self.memory_capacity,
            self.eval,
            self.random_models_on,
            self.majority_voting_prompt,
        )

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

        Uses a caching mechanism to return a cached instance if an agent with the same configuration already exists.

        Returns:
            The selected reasoning swarm instance.

        Raises:
            ValueError: If the specified swarm type is invalid.
        """
        # Generate cache key
        cache_key = self._get_cache_key()

        # Check if an instance with the same configuration already exists in the cache
        if cache_key in self.__class__._agent_cache:
            return self.__class__._agent_cache[cache_key]

        try:
            # Use the factory method to create a new instance
            agent = self.agent_factories[self.swarm_type]()

            # Add the newly created instance to the cache
            self.__class__._agent_cache[cache_key] = agent

            return agent
        except KeyError:
            # Keep the same error handling as the original code
            raise ValueError(f"Invalid swarm type: {self.swarm_type}")

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
        swarm = self.select_swarm()
        return swarm.run(task=task, *args, **kwargs)

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

    @classmethod
    def clear_cache(cls):
        """
        Clear the agent instance cache.

        Use this when you need to free memory or force the creation of new instances.
        """
        cls._agent_cache.clear()
