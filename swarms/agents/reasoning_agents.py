from typing import (
    List,
    Literal,
    Dict,
    Callable,
    Any,
    Tuple,
    Hashable,
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

    # Class variable to store cached agent instances
    _agent_cache: Dict[Tuple[Hashable, ...], Any] = {}

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

        # Added: Initialize the factory mapping dictionary

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
            Tuple[Hashable, ...]: A hashable tuple to serve as the cache key
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
        )

    def _create_reasoning_duo(self):
        """Create an agent instance for the ReasoningDuo type"""
        return ReasoningDuo(
            agent_name=self.agent_name,
            agent_description=self.description,
            model_name=[self.model_name, self.model_name],
            system_prompt=self.system_prompt,
            output_type=self.output_type,
        )

    def _create_consistency_agent(self):
        """Create an agent instance for the SelfConsistencyAgent type"""
        return SelfConsistencyAgent(
            agent_name=self.agent_name,
            description=self.description,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            max_loops=self.max_loops,
            num_samples=self.num_samples,
            output_type=self.output_type,
        )

    def _create_ire_agent(self):
        """Create an agent instance for the IREAgent type"""
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
        """Create an agent instance for the AgentJudge type"""
        return AgentJudge(
            agent_name=self.agent_name,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            max_loops=self.max_loops,
        )

    def _create_reflexion_agent(self):
        """Create an agent instance for the ReflexionAgent type"""
        return ReflexionAgent(
            agent_name=self.agent_name,
            system_prompt=self.system_prompt,
            model_name=self.model_name,
            max_loops=self.max_loops,
        )

    def _create_gkp_agent(self):
        """Create an agent instance for the GKPAgent type"""
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


        Returns:
            The result of the reasoning process.
        """
        swarm = self.select_swarm()
        return swarm.run(task=task)

    def batched_run(self, tasks: List[str], *args, **kwargs):
        """
        Execute the reasoning process on a batch of tasks.


        Args:
            tasks (List[str]): The list of tasks to process.


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
