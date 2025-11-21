"""MakeASwarm Framework - Combine swarm architectures and agent types to create new architectures."""

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import orjson

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.board_of_directors_swarm import BoardOfDirectorsSwarm
from swarms.structs.batched_grid_workflow import BatchedGridWorkflow
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms.structs.council_as_judge import CouncilAsAJudge
from swarms.structs.groupchat import GroupChat
from swarms.structs.heavy_swarm import HeavySwarm
from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms.structs.interactive_groupchat import InteractiveGroupChat
from swarms.structs.majority_voting import MajorityVoting
from swarms.structs.malt import MALT
from swarms.structs.mixture_of_agents import MixtureOfAgents
from swarms.structs.multi_agent_router import MultiAgentRouter
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="make_a_swarm")


class MakeASwarmError(Exception):
    pass


class CycleDetectionError(MakeASwarmError):
    pass


class ComponentNotFoundError(MakeASwarmError):
    pass


class TopologicalSorter:
    """Topological sort using Kahn's algorithm."""

    def __init__(self, graph: Dict[str, List[str]]) -> None:
        self.graph = graph

    def sort(self) -> List[List[str]]:
        all_nodes = set(self.graph.keys())
        for deps in self.graph.values():
            all_nodes.update(deps)

        in_degree = {node: len(self.graph.get(node, [])) for node in all_nodes}
        queue = deque([node for node in in_degree if in_degree[node] == 0])
        result: List[List[str]] = []
        processed = 0

        while queue:
            current_level: List[str] = []
            level_size = len(queue)

            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node)
                processed += 1

                for dependent in self.graph:
                    if node in self.graph[dependent]:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            queue.append(dependent)

            result.append(current_level)

        if processed != len(all_nodes):
            raise CycleDetectionError(
                "Cycle detected in execution order dependencies."
            )

        return result


class ComponentRegistry:
    """Registry for tracking swarm architectures, agents, and reasoning architectures by name."""

    def __init__(self) -> None:
        self.components: Dict[str, Any] = {}

    def add(self, name: str, component: Any) -> None:
        if not name:
            raise ValueError("Component name cannot be empty")
        if component is None:
            raise ValueError("Component cannot be None")
        if not (hasattr(component, 'run') and callable(getattr(component, 'run'))) and not callable(component):
            raise ValueError(f"Component must have a 'run' method or be callable. Got: {type(component)}")
        self.components[name] = component

    def get(self, name: str) -> Any:
        if name not in self.components:
            raise ComponentNotFoundError(
                f"Component '{name}' not found. Available: {list(self.components.keys())}"
            )
        return self.components[name]

    def has(self, name: str) -> bool:
        return name in self.components

    def list_all(self) -> List[str]:
        return list(self.components.keys())

    def clear(self) -> None:
        self.components.clear()


class MakeASwarm(BaseSwarm):
    """
    Framework for combining swarm architectures and creating new reasoning architectures.
    
    Combines different swarm types (HeavySwarm, BoardOfDirectors, GroupChat, etc.) as components.
    Creates nested architectures where swarms contain other swarms.
    Mixes agent types (ToT, CoT, Reflexion, etc.) to create new reasoning architectures.
    """

    def __init__(
        self,
        name: Optional[str] = "MakeASwarm",
        description: Optional[str] = "A custom swarm created with MakeASwarm",
        agents: Optional[List[Union[Agent, Callable]]] = None,
        max_loops: Optional[int] = 1,
        execution_mode: Literal["sequential", "concurrent", "dependency"] = "sequential",
        execution_order: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            agents=agents or [],
            max_loops=max_loops,
            *args,
            **kwargs,
        )

        # Follow agent.py pattern: set agent_name to match name (line 664 in agent.py: self.name = agent_name)
        self.agent_name = name

        self.component_registry = ComponentRegistry()
        self.execution_mode = execution_mode
        self.execution_order = execution_order
        self.nested_swarms: Dict[str, BaseSwarm] = {}
        self.workflow_graph: Dict[str, List[str]] = {}
        self.execution_results: Dict[str, Any] = {}
        self.output_type = kwargs.get("output_type", "dict")
        self._built = False

    def add_component(self, name: str, component: Any) -> None:
        """Add a swarm architecture, agent, or reasoning architecture to the registry."""
        self.component_registry.add(name, component)

    def create_agent(self, config: Dict[str, Any]) -> Any:
        """Create an agent or reasoning architecture from config. Supports all agent types."""
        agent_type = config.pop("agent_type", "Agent").lower()
        
        # Handle base Agent
        if agent_type == "agent" or agent_type == "base":
            if "agent_name" not in config:
                raise ValueError("agent_name is required in agent config")
            agent_name = config.pop("agent_name")
            system_prompt = config.pop("system_prompt", None)
            return Agent(agent_name=agent_name, system_prompt=system_prompt, **config)
        
        # Comprehensive agent factory
        agent_factory: Dict[str, Callable] = {
            "cotagent": lambda: self._create_cot_agent(config),
            "cot": lambda: self._create_cot_agent(config),
            "totagent": lambda: self._create_tot_agent(config),
            "tot": lambda: self._create_tot_agent(config),
            "gotagent": lambda: self._create_got_agent(config),
            "got": lambda: self._create_got_agent(config),
            "aerasigmaagent": lambda: self._create_aerasigma_agent(config),
            "aerasigma": lambda: self._create_aerasigma_agent(config),
            "selfconsistencyagent": lambda: self._create_self_consistency_agent(config),
            "selfconsistency": lambda: self._create_self_consistency_agent(config),
            "reflexionagent": lambda: self._create_reflexion_agent(config),
            "reflexion": lambda: self._create_reflexion_agent(config),
            "gkpagent": lambda: self._create_gkp_agent(config),
            "gkp": lambda: self._create_gkp_agent(config),
            "agentjudge": lambda: self._create_agent_judge(config),
            "judge": lambda: self._create_agent_judge(config),
            "crcaagent": lambda: self._create_crca_agent(config),
            "crca": lambda: self._create_crca_agent(config),
            "iterativereflectiveexpansion": lambda: self._create_ire_agent(config),
            "ire": lambda: self._create_ire_agent(config),
            "reasoningduo": lambda: self._create_reasoning_duo(config),
            "reasoning-duo": lambda: self._create_reasoning_duo(config),
            "reasoningagentrouter": lambda: self._create_reasoning_agent_router(config),
            "reasoning-router": lambda: self._create_reasoning_agent_router(config),
            "openaiassistant": lambda: self._create_openai_assistant(config),
            "openai": lambda: self._create_openai_assistant(config),
        }
        
        if agent_type not in agent_factory:
            # Try dynamic import as fallback
            return self._create_agent_dynamic(agent_type, config)
        
        try:
            return agent_factory[agent_type]()
        except Exception as e:
            raise ValueError(f"Failed to create {agent_type} agent: {str(e)}") from e
    
    def _create_cot_agent(self, config: Dict[str, Any]) -> Any:
        """Create a CoTAgent from config."""
        from swarms.agents import CoTAgent
        
        # CoTAgent uses 'name' or 'agent_name', 'description', 'model_name', etc.
        agent_name = config.pop("agent_name", config.pop("name", "cot-agent"))
        description = config.pop("description", None)
        model_name = config.pop("model_name", "gpt-4o")
        
        return CoTAgent(
            agent_name=agent_name,
            description=description,
            model_name=model_name,
            **config
        )
    
    def _create_tot_agent(self, config: Dict[str, Any]) -> Any:
        """Create a ToTAgent from config."""
        from swarms.agents import ToTAgent
        from swarms.agents.tree_of_thought_agent import ToTConfig
        
        # ToTAgent uses 'agent_name', 'model_name', 'config' (ToTConfig), etc.
        agent_name = config.pop("agent_name", config.pop("name", "tot-agent"))
        model_name = config.pop("model_name", "gpt-4o")
        
        # Handle ToTConfig if provided
        tot_config = config.pop("config", None)
        if isinstance(tot_config, dict):
            tot_config = ToTConfig(**tot_config)
        elif tot_config is None:
            tot_config = ToTConfig()
        
        return ToTAgent(
            agent_name=agent_name,
            model_name=model_name,
            config=tot_config,
            **config
        )
    
    def _create_got_agent(self, config: Dict[str, Any]) -> Any:
        """Create a GoTAgent from config."""
        from swarms.agents import GoTAgent
        
        agent_name = config.pop("agent_name", config.pop("name", "got-agent"))
        description = config.pop("description", None)
        model_name = config.pop("model_name", "gpt-4o")
        
        return GoTAgent(
            agent_name=agent_name,
            description=description,
            model_name=model_name,
            **config
        )
    
    def _create_aerasigma_agent(self, config: Dict[str, Any]) -> Any:
        """Create an AERASigmaAgent from config."""
        from swarms.agents import AERASigmaAgent
        
        agent_name = config.pop("agent_name", config.pop("name", "aerasigma-agent"))
        model_name = config.pop("model_name", "gpt-4o")
        
        return AERASigmaAgent(
            agent_name=agent_name,
            model_name=model_name,
            **config
        )
    
    def _create_self_consistency_agent(self, config: Dict[str, Any]) -> Any:
        """Create a SelfConsistencyAgent from config."""
        from swarms.agents import SelfConsistencyAgent
        
        agent_name = config.pop("agent_name", config.pop("name", "consistency-agent"))
        model_name = config.pop("model_name", "gpt-4o")
        num_samples = config.pop("num_samples", 3)
        
        return SelfConsistencyAgent(
            agent_name=agent_name,
            model_name=model_name,
            num_samples=num_samples,
            **config
        )
    
    def _create_reflexion_agent(self, config: Dict[str, Any]) -> Any:
        """Create a ReflexionAgent from config."""
        from swarms.agents import ReflexionAgent
        
        agent_name = config.pop("agent_name", config.pop("name", "reflexion-agent"))
        model_name = config.pop("model_name", "openai/o1")
        max_loops = config.pop("max_loops", 3)
        memory_capacity = config.pop("memory_capacity", 100)
        system_prompt = config.pop("system_prompt", None)
        
        return ReflexionAgent(
            agent_name=agent_name,
            model_name=model_name,
            max_loops=max_loops,
            memory_capacity=memory_capacity,
            system_prompt=system_prompt,
            **config
        )
    
    def _create_gkp_agent(self, config: Dict[str, Any]) -> Any:
        """Create a GKPAgent from config."""
        from swarms.agents import GKPAgent
        
        agent_name = config.pop("agent_name", config.pop("name", "gkp-agent"))
        model_name = config.pop("model_name", "gpt-4o")
        num_knowledge_items = config.pop("num_knowledge_items", 5)
        
        return GKPAgent(
            agent_name=agent_name,
            model_name=model_name,
            num_knowledge_items=num_knowledge_items,
            **config
        )
    
    def _create_agent_judge(self, config: Dict[str, Any]) -> Any:
        """Create an AgentJudge from config."""
        from swarms.agents import AgentJudge
        
        agent_name = config.pop("agent_name", config.pop("name", "judge-agent"))
        model_name = config.pop("model_name", "gpt-4o")
        
        return AgentJudge(
            agent_name=agent_name,
            model_name=model_name,
            **config
        )
    
    def _create_crca_agent(self, config: Dict[str, Any]) -> Any:
        """Create a CRCAAgent from config."""
        from swarms.agents import CRCAAgent
        
        agent_name = config.pop("agent_name", config.pop("name", "crca-agent"))
        model_name = config.pop("model_name", "gpt-4o")
        
        return CRCAAgent(
            agent_name=agent_name,
            model_name=model_name,
            **config
        )
    
    def _create_ire_agent(self, config: Dict[str, Any]) -> Any:
        """Create an IterativeReflectiveExpansion agent from config."""
        from swarms.agents import IterativeReflectiveExpansion
        
        agent_name = config.pop("agent_name", config.pop("name", "ire-agent"))
        model_name = config.pop("model_name", "gpt-4o")
        
        return IterativeReflectiveExpansion(
            agent_name=agent_name,
            model_name=model_name,
            **config
        )
    
    def _create_reasoning_duo(self, config: Dict[str, Any]) -> Any:
        """Create a ReasoningDuo from config."""
        from swarms.agents import ReasoningDuo
        
        agent_name = config.pop("agent_name", config.pop("name", "reasoning-duo"))
        model_name = config.pop("model_name", "gpt-4o-mini")
        reasoning_model_name = config.pop("reasoning_model_name", "claude-3-5-sonnet-20240620")
        system_prompt = config.pop("system_prompt", None)
        
        return ReasoningDuo(
            agent_name=agent_name,
            model_name=model_name,
            reasoning_model_name=reasoning_model_name,
            system_prompt=system_prompt,
            **config
        )
    
    def _create_reasoning_agent_router(self, config: Dict[str, Any]) -> Any:
        """Create a ReasoningAgentRouter from config."""
        from swarms.agents import ReasoningAgentRouter
        
        agent_name = config.pop("agent_name", config.pop("name", "reasoning-router"))
        model_name = config.pop("model_name", "gpt-4o")
        swarm_type = config.pop("swarm_type", "reasoning-duo")
        
        return ReasoningAgentRouter(
            agent_name=agent_name,
            model_name=model_name,
            swarm_type=swarm_type,
            **config
        )
    
    def _create_openai_assistant(self, config: Dict[str, Any]) -> Any:
        """Create an OpenAIAssistant from config."""
        from swarms.agents.openai_assistant import OpenAIAssistant
        
        agent_name = config.pop("agent_name", config.pop("name", "openai-assistant"))
        model_name = config.pop("model_name", "gpt-4o")
        
        return OpenAIAssistant(
            agent_name=agent_name,
            model_name=model_name,
            **config
        )
    
    def _create_agent_dynamic(self, agent_type: str, config: Dict[str, Any]) -> Any:
        """Dynamically import and create an agent type not in the factory."""
        try:
            # Try importing from swarms.agents
            from swarms.agents import __all__ as agent_exports
            
            # Normalize agent type name
            agent_class_name = agent_type.replace("-", "").replace("_", "")
            agent_class_name = "".join(word.capitalize() for word in agent_class_name.split())
            
            # Try to find matching class
            for export_name in agent_exports:
                if export_name.lower() == agent_type or export_name.lower() == agent_class_name.lower():
                    agent_class = getattr(__import__(f"swarms.agents.{export_name.lower()}", fromlist=[export_name]), export_name)
                    agent_name = config.pop("agent_name", config.pop("name", f"{agent_type}-agent"))
                    return agent_class(agent_name=agent_name, **config)
            
            raise ValueError(f"Agent type '{agent_type}' not found in swarms.agents")
        except Exception as e:
            raise ValueError(
                f"Failed to dynamically create agent type '{agent_type}': {str(e)}. "
                f"Please specify a valid agent_type or ensure the agent class is importable."
            ) from e

    def create_swarm(
        self, name: str, swarm_type: Union[str, type], config: Dict[str, Any]
    ) -> BaseSwarm:
        """Create a swarm architecture from configuration. Can be combined with other swarms."""
        if isinstance(swarm_type, type):
            try:
                return swarm_type(**config)
            except Exception as e:
                raise ValueError(
                    f"Failed to create custom swarm {swarm_type.__name__}: {str(e)}"
                ) from e

        swarm_factory: Dict[str, Callable] = {
            "SequentialWorkflow": lambda: SequentialWorkflow(**config),
            "ConcurrentWorkflow": lambda: ConcurrentWorkflow(**config),
            "GroupChat": lambda: GroupChat(**config),
            "HeavySwarm": lambda: HeavySwarm(**config),
            "HierarchicalSwarm": lambda: HierarchicalSwarm(**config),
            "HiearchicalSwarm": lambda: HierarchicalSwarm(**config),
            "MixtureOfAgents": lambda: MixtureOfAgents(**config),
            "MajorityVoting": lambda: MajorityVoting(**config),
            "MALT": lambda: MALT(**config),
            "CouncilAsAJudge": lambda: CouncilAsAJudge(**config),
            "InteractiveGroupChat": lambda: InteractiveGroupChat(**config),
            "MultiAgentRouter": lambda: MultiAgentRouter(**config),
            "BoardOfDirectorsSwarm": lambda: BoardOfDirectorsSwarm(**config),
            "BatchedGridWorkflow": lambda: BatchedGridWorkflow(**config),
            "AgentRearrange": lambda: self._create_agent_rearrange(config),
            "RoundRobinSwarm": lambda: self._create_round_robin_swarm(config),
            "SpreadSheetSwarm": lambda: self._create_spreadsheet_swarm(config),
            "SwarmRouter": lambda: self._create_swarm_router(config),
            "GraphWorkflow": lambda: self._create_graph_workflow(config),
            "AutoSwarmBuilder": lambda: self._create_auto_swarm_builder(config),
            "HybridHierarchicalClusterSwarm": lambda: self._create_hybrid_hierarchical_swarm(config),
            "AOP": lambda: self._create_aop(config),
            "SelfMoASeq": lambda: self._create_self_moa_seq(config),
            "SocialAlgorithms": lambda: self._create_social_algorithms(config),
        }

        if swarm_type not in swarm_factory:
            # Try dynamic import as fallback
            return self._create_swarm_dynamic(swarm_type, config)

        try:
            return swarm_factory[swarm_type]()
        except Exception as e:
            raise ValueError(f"Failed to create swarm {swarm_type}: {str(e)}") from e
    
    def _create_agent_rearrange(self, config: Dict[str, Any]) -> Any:
        """Create an AgentRearrange from config."""
        from swarms.structs.agent_rearrange import AgentRearrange
        return AgentRearrange(**config)
    
    def _create_round_robin_swarm(self, config: Dict[str, Any]) -> Any:
        """Create a RoundRobinSwarm from config."""
        from swarms.structs.round_robin import RoundRobinSwarm
        return RoundRobinSwarm(**config)
    
    def _create_spreadsheet_swarm(self, config: Dict[str, Any]) -> Any:
        """Create a SpreadSheetSwarm from config."""
        from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm
        return SpreadSheetSwarm(**config)
    
    def _create_swarm_router(self, config: Dict[str, Any]) -> Any:
        """Create a SwarmRouter from config."""
        from swarms.structs.swarm_router import SwarmRouter
        return SwarmRouter(**config)
    
    def _create_graph_workflow(self, config: Dict[str, Any]) -> Any:
        """Create a GraphWorkflow from config."""
        from swarms.structs.graph_workflow import GraphWorkflow
        return GraphWorkflow(**config)
    
    def _create_auto_swarm_builder(self, config: Dict[str, Any]) -> Any:
        """Create an AutoSwarmBuilder from config."""
        from swarms.structs.auto_swarm_builder import AutoSwarmBuilder
        return AutoSwarmBuilder(**config)
    
    def _create_hybrid_hierarchical_swarm(self, config: Dict[str, Any]) -> Any:
        """Create a HybridHierarchicalClusterSwarm from config."""
        from swarms.structs.hybrid_hiearchical_peer_swarm import HybridHierarchicalClusterSwarm
        return HybridHierarchicalClusterSwarm(**config)
    
    def _create_aop(self, config: Dict[str, Any]) -> Any:
        """Create an AOP from config."""
        from swarms.structs.aop import AOP
        return AOP(**config)
    
    def _create_self_moa_seq(self, config: Dict[str, Any]) -> Any:
        """Create a SelfMoASeq from config."""
        from swarms.structs.self_moa_seq import SelfMoASeq
        return SelfMoASeq(**config)
    
    def _create_social_algorithms(self, config: Dict[str, Any]) -> Any:
        """Create a SocialAlgorithms from config."""
        from swarms.structs.social_algorithms import SocialAlgorithms
        return SocialAlgorithms(**config)
    
    def _create_swarm_dynamic(self, swarm_type: str, config: Dict[str, Any]) -> Any:
        """Dynamically import and create a swarm type not in the factory."""
        try:
            # Try importing from swarms.structs
            from swarms.structs import __all__ as struct_exports
            
            # Normalize swarm type name
            swarm_class_name = swarm_type.replace("-", "").replace("_", "")
            
            # Try to find matching class
            for export_name in struct_exports:
                if export_name.lower() == swarm_type.lower() or export_name == swarm_type:
                    try:
                        swarm_class = getattr(__import__(f"swarms.structs.{export_name.lower()}", fromlist=[export_name]), export_name)
                        if hasattr(swarm_class, '__call__'):
                            return swarm_class(**config)
                    except (ImportError, AttributeError):
                        continue
            
            raise ValueError(f"Swarm type '{swarm_type}' not found in swarms.structs")
        except Exception as e:
            raise ValueError(
                f"Failed to dynamically create swarm type '{swarm_type}': {str(e)}. "
                f"Please specify a valid swarm_type or ensure the swarm class is importable."
            ) from e

    def set_execution_order(
        self, order: Union[List[str], Dict[str, List[str]]]
    ) -> None:
        """Set execution order for swarm architectures and components."""
        self.execution_order = order
        if isinstance(order, dict):
            self.workflow_graph = order

    def _resolve_dependencies(self) -> List[List[str]]:
        if not isinstance(self.execution_order, dict):
            raise ValueError("Dependency resolution requires execution_order to be a dict")
        return TopologicalSorter(self.workflow_graph).sort()

    def _execute_sequential(
        self, order: List[str], task: str, *args, **kwargs
    ) -> Dict[str, Any]:
        """Execute swarm architectures sequentially."""
        results: Dict[str, Any] = {}
        previous_result: Any = None

        for component_name in order:
            component = self.component_registry.get(component_name)
            # Component can be a swarm architecture, agent, or reasoning architecture
            if hasattr(component, 'run') and callable(getattr(component, 'run')):
                result = component.run(task, *args, **kwargs)
            elif callable(component):
                result = component(task, *args, **kwargs)
            else:
                raise ValueError(f"Unknown component type: {type(component)}. Component must have a 'run' method or be callable.")

            results[component_name] = result
            previous_result = result

            if previous_result and isinstance(previous_result, str):
                task = f"{task}\n\nPrevious result: {previous_result}"

        return results

    def _execute_concurrent(
        self, order: List[str], task: str, *args, **kwargs
    ) -> Dict[str, Any]:
        """Execute swarm architectures concurrently (in parallel)."""
        results: Dict[str, Any] = {}
        with ThreadPoolExecutor() as executor:
            futures = {}
            for component_name in order:
                component = self.component_registry.get(component_name)
                # Component: swarm architecture (HeavySwarm, BoardOfDirectors, etc.), agent, or reasoning architecture
                if hasattr(component, 'run') and callable(getattr(component, 'run')):
                    future = executor.submit(component.run, task, *args, **kwargs)
                elif callable(component):
                    future = executor.submit(component, task, *args, **kwargs)
                else:
                    raise ValueError(f"Unknown component type: {type(component)}. Component must have a 'run' method or be callable.")
                futures[future] = component_name

            for future in as_completed(futures):
                component_name = futures[future]
                try:
                    results[component_name] = future.result()
                except Exception as e:
                    results[component_name] = {"error": str(e)}

        return results

    def _execute_with_dependencies(
        self, task: str, *args, **kwargs
    ) -> Dict[str, Any]:
        """Execute swarm architectures based on dependency graph using topological sorting."""
        execution_levels = self._resolve_dependencies()
        results: Dict[str, Any] = {}

        for level in execution_levels:
            # Execute all swarm architectures at this dependency level concurrently (HeavySwarm, BoardOfDirectors, etc.)
            level_results = self._execute_concurrent(level, task, *args, **kwargs)
            results.update(level_results)

            if level_results:
                level_summary = "\n".join(
                    [f"{name}: {str(result)[:100]}" for name, result in level_results.items()]
                )
                task = f"{task}\n\nResults from previous level:\n{level_summary}"

        return results

    def build(self) -> None:
        """Build the final multi-architecture swarm structure."""
        if self._built:
            logger.warning("Swarm already built, rebuilding...")

        if self.execution_order is None:
            self.execution_order = self.component_registry.list_all()

        if isinstance(self.execution_order, list):
            for component_name in self.execution_order:
                if not self.component_registry.has(component_name):
                    raise ComponentNotFoundError(
                        f"Component '{component_name}' not found in registry"
                    )
        elif isinstance(self.execution_order, dict):
            all_components = set(self.execution_order.keys())
            for deps in self.execution_order.values():
                all_components.update(deps)

            for component_name in all_components:
                if not self.component_registry.has(component_name):
                    raise ComponentNotFoundError(
                        f"Component '{component_name}' not found in registry"
                    )

        # Include all runnable components (swarm architectures, agents, reasoning architectures, callables)
        self.agents = [
            comp
            for comp in self.component_registry.components.values()
            if (hasattr(comp, 'run') and callable(getattr(comp, 'run'))) or callable(comp)
        ]

        self._built = True

    def run(self, task: Optional[str] = None, *args, **kwargs) -> Any:
        """Execute the combined swarm architecture system."""
        if not self._built:
            self.build()

        if task is None:
            task = "Execute the workflow"

        if self.execution_mode == "sequential":
            if not isinstance(self.execution_order, list):
                raise ValueError("Sequential mode requires execution_order to be a list")
            self.execution_results = self._execute_sequential(
                self.execution_order, task, *args, **kwargs
            )
        elif self.execution_mode == "concurrent":
            order = (
                list(self.execution_order.keys())
                if isinstance(self.execution_order, dict)
                else self.execution_order
            )
            self.execution_results = self._execute_concurrent(order, task, *args, **kwargs)
        elif self.execution_mode == "dependency":
            self.execution_results = self._execute_with_dependencies(task, *args, **kwargs)
        else:
            raise ValueError(f"Invalid execution_mode: {self.execution_mode}")

        output_type = getattr(self, 'output_type', None)
        if output_type and "dict" in str(output_type):
            return self.execution_results
        elif len(self.execution_results) == 1:
            return list(self.execution_results.values())[0]
        else:
            return list(self.execution_results.values())

    def export_to_json(self, filepath: str) -> None:
        """Export the combined swarm architecture configuration to JSON and generate importable module."""
        config = {
            "name": self.name,
            "description": self.description,
            "execution_mode": self.execution_mode,
            "execution_order": self.execution_order,
            "max_loops": self.max_loops,
            "components": {},
        }

        for name, component in self.component_registry.components.items():
            # Check if it's an agent (base Agent or specialized agents)
            if hasattr(component, 'run') and not isinstance(component, BaseSwarm):
                # Try to determine agent type from class name
                agent_type = "Agent"
                if hasattr(component, '__class__'):
                    class_name = component.__class__.__name__
                    # Map all known agent types
                    agent_type_map = {
                        "CoTAgent": "CoTAgent",
                        "ToTAgent": "ToTAgent",
                        "GoTAgent": "GoTAgent",
                        "AERASigmaAgent": "AERASigmaAgent",
                        "EnhancedAERASigmaAgent": "AERASigmaAgent",
                        "SelfConsistencyAgent": "SelfConsistencyAgent",
                        "ReflexionAgent": "ReflexionAgent",
                        "GKPAgent": "GKPAgent",
                        "AgentJudge": "AgentJudge",
                        "CRCAAgent": "CRCAAgent",
                        "IterativeReflectiveExpansion": "IterativeReflectiveExpansion",
                        "ReasoningDuo": "ReasoningDuo",
                        "ReasoningAgentRouter": "ReasoningAgentRouter",
                        "OpenAIAssistant": "OpenAIAssistant",
                    }
                    agent_type = agent_type_map.get(class_name, "Agent")
                
                agent_config = {
                    "agent_type": agent_type.lower(),
                    "agent_name": getattr(component, "agent_name", getattr(component, "name", name)),
                    "model_name": getattr(component, "model_name", None),
                }
                
                # Add common optional fields
                common_fields = [
                    "system_prompt", "description", "global_system_prompt", 
                    "secondary_system_prompt", "max_loops", "output_type"
                ]
                for field in common_fields:
                    if hasattr(component, field):
                        value = getattr(component, field, None)
                        if value is not None:
                            agent_config[field] = value
                
                # Export agent-specific parameters
                if agent_type == "ToTAgent" and hasattr(component, "config"):
                    tot_config = getattr(component, "config", None)
                    if tot_config:
                        agent_config["config"] = {
                            "max_depth": getattr(tot_config, "max_depth", 3),
                            "branch_factor": getattr(tot_config, "branch_factor", 2),
                            "beam_width": getattr(tot_config, "beam_width", 3),
                            "search_strategy": getattr(tot_config, "search_strategy", "beam"),
                        }
                elif agent_type == "ReasoningDuo" and hasattr(component, "reasoning_model_name"):
                    agent_config["reasoning_model_name"] = getattr(component, "reasoning_model_name", None)
                elif agent_type == "ReflexionAgent" and hasattr(component, "memory_capacity"):
                    agent_config["memory_capacity"] = getattr(component, "memory_capacity", 100)
                elif agent_type == "GKPAgent" and hasattr(component, "num_knowledge_items"):
                    agent_config["num_knowledge_items"] = getattr(component, "num_knowledge_items", 5)
                elif agent_type == "SelfConsistencyAgent" and hasattr(component, "num_samples"):
                    agent_config["num_samples"] = getattr(component, "num_samples", 3)
                elif agent_type == "ReasoningAgentRouter" and hasattr(component, "swarm_type"):
                    agent_config["swarm_type"] = getattr(component, "swarm_type", "reasoning-duo")
                
                # Try to export any additional serializable attributes
                # Skip non-serializable objects (callables, LLMs, etc.)
                skip_attrs = {
                    'run', 'llm', 'agent', 'reasoner', 'controller', 'encoder', 
                    'expander', 'merger', 'conversation', 'memory', 'actor', 
                    'evaluator', 'reflector', '__dict__', '__class__', '__module__'
                }
                
                for attr_name in dir(component):
                    if attr_name.startswith('_') or attr_name in skip_attrs:
                        continue
                    if attr_name in agent_config:
                        continue
                    try:
                        attr_value = getattr(component, attr_name)
                        # Skip callables and complex objects
                        if callable(attr_value) or hasattr(attr_value, '__dict__'):
                            continue
                        # Try to serialize
                        try:
                            orjson.dumps(attr_value)
                            agent_config[attr_name] = attr_value
                        except (TypeError, ValueError):
                            continue
                    except (AttributeError, Exception):
                        continue
                
                config["components"][name] = {
                    "type": "agent",
                    "config": agent_config,
                }
            elif isinstance(component, BaseSwarm) or (hasattr(component, 'run') and hasattr(component, 'name')):
                # Export swarm configuration
                swarm_config = {
                    "name": getattr(component, "name", name),
                    "description": getattr(component, "description", ""),
                    "max_loops": getattr(component, "max_loops", 1),
                }
                
                # Export common swarm fields
                common_swarm_fields = [
                    "execution_mode", "output_type", "agents", "board_model_name",
                    "worker_model_name", "question_agent_model_name", "loops_per_agent"
                ]
                for field in common_swarm_fields:
                    if hasattr(component, field):
                        value = getattr(component, field, None)
                        if value is not None:
                            try:
                                # Try to serialize to check if it's JSON-serializable
                                orjson.dumps(value)
                                swarm_config[field] = value
                            except (TypeError, ValueError):
                                # Skip non-serializable values
                                continue
                
                # Try to export additional serializable attributes
                skip_attrs = {
                    'run', 'agents', 'components', 'component_registry', 
                    'execution_results', 'workflow_graph', 'nested_swarms',
                    '__dict__', '__class__', '__module__'
                }
                
                for attr_name in dir(component):
                    if attr_name.startswith('_') or attr_name in skip_attrs:
                        continue
                    if attr_name in swarm_config:
                        continue
                    try:
                        attr_value = getattr(component, attr_name)
                        # Skip callables and complex objects
                        if callable(attr_value) or hasattr(attr_value, '__dict__'):
                            continue
                        # Try to serialize
                        try:
                            orjson.dumps(attr_value)
                            swarm_config[attr_name] = attr_value
                        except (TypeError, ValueError):
                            continue
                    except (AttributeError, Exception):
                        continue
                
                config["components"][name] = {
                    "type": component.__class__.__name__,
                    "config": swarm_config,
                }

        try:
            json_bytes = orjson.dumps(config, option=orjson.OPT_INDENT_2)
            with open(filepath, "wb") as f:
                f.write(json_bytes)
            
            # Generate Python module for automatic import
            self._generate_import_module(filepath)
        except Exception as e:
            raise IOError(f"Failed to export configuration: {str(e)}") from e

    def load_from_json(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        try:
            with open(filepath, "rb") as f:
                config = orjson.loads(f.read())
        except Exception as e:
            raise ValueError(f"Failed to parse JSON file: {str(e)}") from e

        if "components" not in config:
            raise ValueError("Configuration missing 'components' field")

        self.name = config.get("name", self.name)
        self.description = config.get("description", self.description)
        self.execution_mode = config.get("execution_mode", self.execution_mode)
        self.execution_order = config.get("execution_order", self.execution_order)
        self.max_loops = config.get("max_loops", self.max_loops)

        for name, component_config in config["components"].items():
            comp_type = component_config.get("type")
            comp_config = component_config.get("config", {})
            
            # Make a copy to avoid mutating the original config
            comp_config = comp_config.copy() if isinstance(comp_config, dict) else comp_config

            try:
                if comp_type == "agent":
                    # Ensure agent_type is present and lowercase
                    if "agent_type" not in comp_config:
                        comp_config["agent_type"] = "agent"  # Default to base Agent
                    else:
                        comp_config["agent_type"] = str(comp_config["agent_type"]).lower()
                    self.add_component(name, self.create_agent(comp_config))
                else:
                    # Create swarm with the type name
                    self.add_component(name, self.create_swarm(name, comp_type, comp_config))
            except Exception as e:
                raise ValueError(
                    f"Failed to load component '{name}' of type '{comp_type}': {str(e)}"
                ) from e

        self.build()

    def _generate_import_module(self, json_filepath: str) -> None:
        """
        Generate a Python module file that allows importing the exported swarm/agent.
        
        When you export to 'my_swarm.json', this creates 'swarms/exports/my_swarm.py'
        which can be imported as: from swarms.exports import my_swarm
        
        Args:
            json_filepath: Path to the exported JSON file
            
        Raises:
            IOError: If module generation fails
        """
        try:
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(json_filepath))[0]
            # Sanitize filename to be a valid Python identifier
            module_name = "".join(c if c.isalnum() or c == "_" else "_" for c in base_name)
            if module_name and module_name[0].isdigit():
                module_name = "_" + module_name
            
            # Create exports directory if it doesn't exist
            # Get the swarms package directory (parent of structs)
            structs_dir = os.path.dirname(__file__)
            swarms_dir = os.path.dirname(structs_dir)
            exports_dir = os.path.join(swarms_dir, "exports")
            os.makedirs(exports_dir, exist_ok=True)
            
            # Create __init__.py if it doesn't exist
            init_file = os.path.join(exports_dir, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write('"""Auto-generated exports directory for MakeASwarm configurations."""\n')
            
            # Generate module file
            module_file = os.path.join(exports_dir, f"{module_name}.py")
            
            # Get path to JSON file (use relative path if possible, otherwise absolute)
            # Try to make it relative to the exports directory for portability
            try:
                # If JSON is in current directory or subdirectory, use relative path
                json_abs = os.path.abspath(json_filepath)
                exports_abs = os.path.abspath(exports_dir)
                try:
                    json_rel = os.path.relpath(json_abs, exports_abs)
                    # Use relative path if it doesn't go outside exports directory
                    if not json_rel.startswith(".."):
                        json_path = json_rel
                    else:
                        json_path = json_abs
                except ValueError:
                    # Different drives on Windows, use absolute
                    json_path = json_abs
            except Exception:
                json_path = os.path.abspath(json_filepath)
            
            # Generate module content
            module_content = f'''"""
Auto-generated module for imported swarm/agent: {self.name}

This module was automatically generated by MakeASwarm.export_to_json()
when exporting to: {json_filepath}

Usage:
    from swarms.exports import {module_name}
    result = {module_name}.swarm.run("your task here")
"""

from swarms.structs.make_a_swarm import MakeASwarm
import os as _os

# Load the swarm from JSON configuration
# Try relative path first, fall back to absolute
_json_path = _os.path.join(_os.path.dirname(__file__), r"{json_path}") if not _os.path.isabs(r"{json_path}") else r"{json_path}"
_swarm = MakeASwarm()
_swarm.load_from_json(_json_path)

# Export the swarm instance
swarm = _swarm

# Export metadata
name = "{self.name}"
description = {repr(self.description)}
config_path = _json_path

__all__ = ["swarm", "name", "description", "config_path"]
'''
            
            with open(module_file, "w", encoding="utf-8") as f:
                f.write(module_content)
            
            logger.info(
                f"Generated importable module: {module_file}\n"
                f"You can now import it as: from swarms.exports import {module_name}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to generate import module: {str(e)}")
            # Don't raise - JSON export succeeded, module generation is optional
