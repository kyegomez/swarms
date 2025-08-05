from swarms.structs.agent import Agent
from swarms.structs.agent_builder import AgentsBuilder
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder
from swarms.structs.base_structure import BaseStructure
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.batch_agent_execution import batch_agent_execution
from swarms.structs.board_of_directors_swarm import (
    BoardConfig,
    BoardConfigModel,
    BoardDecision,
    BoardDecisionType,
    BoardFeatureStatus,
    BoardMember,
    BoardMemberRole,
    BoardOfDirectorsSwarm,
    BoardOrder,
    BoardSpec,
    create_default_config_file,
    disable_board_feature,
    disable_verbose_logging,
    enable_board_feature,
    enable_verbose_logging,
    get_board_config,
    is_board_feature_enabled,
    set_board_model,
    set_board_size,
    set_decision_threshold,
)
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms.structs.conversation import Conversation
from swarms.structs.council_judge import CouncilAsAJudge
from swarms.structs.cron_job import CronJob
from swarms.structs.de_hallucination_swarm import DeHallucinationSwarm
from swarms.structs.deep_research_swarm import DeepResearchSwarm
from swarms.structs.graph_workflow import (
    Edge,
    GraphWorkflow,
    Node,
    NodeType,
)
from swarms.structs.groupchat import (
    GroupChat,
    expertise_based,
)
from swarms.structs.heavy_swarm import HeavySwarm
from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms.structs.hybrid_hiearchical_peer_swarm import (
    HybridHierarchicalClusterSwarm,
)
from swarms.structs.interactive_groupchat import (
    InteractiveGroupChat,
    priority_speaker,
    random_dynamic_speaker,
    random_speaker,
    round_robin_speaker,
)
from swarms.structs.ma_blocks import (
    aggregate,
    find_agent_by_name,
    run_agent,
)
from swarms.structs.majority_voting import (
    MajorityVoting,
    majority_voting,
    most_frequent,
    parse_code_completion,
)
from swarms.structs.malt import MALT
from swarms.structs.meme_agent_persona_generator import (
    MemeAgentGenerator,
)
from swarms.structs.mixture_of_agents import MixtureOfAgents
from swarms.structs.model_router import ModelRouter
from swarms.structs.multi_agent_exec import (
    get_agents_info,
    get_swarms_info,
    run_agent_with_timeout,
    run_agents_concurrently,
    run_agents_concurrently_async,
    run_agents_concurrently_multiprocess,
    run_agents_sequentially,
    run_agents_with_different_tasks,
    run_agents_with_resource_monitoring,
    run_agents_with_tasks_concurrently,
    run_single_agent,
)
from swarms.structs.multi_agent_router import MultiAgentRouter
from swarms.structs.rearrange import AgentRearrange, rearrange
from swarms.structs.round_robin import RoundRobinSwarm
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm
from swarms.structs.swarm_arange import SwarmRearrange
from swarms.structs.swarm_router import (
    SwarmRouter,
    SwarmType,
)
from swarms.structs.swarming_architectures import (
    broadcast,
    circular_swarm,
    exponential_swarm,
    fibonacci_swarm,
    geometric_swarm,
    grid_swarm,
    harmonic_swarm,
    linear_swarm,
    log_swarm,
    mesh_swarm,
    one_to_one,
    one_to_three,
    power_swarm,
    prime_swarm,
    pyramid_swarm,
    sigmoid_swarm,
    staircase_swarm,
    star_swarm,
)

# Standalone function for getting default board templates
def get_default_board_template(template_name: str = "standard") -> dict:
    """
    Get a default board template.
    
    This function provides predefined board templates for common use cases.
    Templates are cached for improved performance.
    
    Args:
        template_name: Name of the template to retrieve
        
    Returns:
        dict: Board template configuration
    """
    config = get_board_config()
    return config.get_default_board_template(template_name)

__all__ = [
    "Agent",
    "BaseStructure",
    "BaseSwarm",
    "BoardConfig",
    "BoardConfigModel",
    "BoardDecision",
    "BoardDecisionType",
    "BoardFeatureStatus",
    "BoardMember",
    "BoardMemberRole",
    "BoardOfDirectorsSwarm",
    "BoardOrder",
    "BoardSpec",
    "ConcurrentWorkflow",
    "Conversation",
    "GroupChat",
    "MajorityVoting",
    "majority_voting",
    "most_frequent",
    "parse_code_completion",
    "AgentRearrange",
    "rearrange",
    "RoundRobinSwarm",
    "SequentialWorkflow",
    "MixtureOfAgents",
    "GraphWorkflow",
    "Node",
    "NodeType",
    "Edge",
    "broadcast",
    "circular_swarm",
    "exponential_swarm",
    "fibonacci_swarm",
    "geometric_swarm",
    "grid_swarm",
    "harmonic_swarm",
    "linear_swarm",
    "log_swarm",
    "mesh_swarm",
    "one_to_one",
    "one_to_three",
    "power_swarm",
    "prime_swarm",
    "pyramid_swarm",
    "sigmoid_swarm",
    "staircase_swarm",
    "star_swarm",
    "SpreadSheetSwarm",
    "SwarmRouter",
    "SwarmType",
    "SwarmRearrange",
    "run_agents_concurrently",
    "run_agents_concurrently_async",
    "run_single_agent",
    "run_agents_concurrently_multiprocess",
    "run_agents_sequentially",
    "run_agents_with_different_tasks",
    "run_agent_with_timeout",
    "run_agents_with_resource_monitoring",
    "run_agents_with_tasks_concurrently",
    "GroupChat",
    "expertise_based",
    "MultiAgentRouter",
    "MemeAgentGenerator",
    "ModelRouter",
    "AgentsBuilder",
    "MALT",
    "DeHallucinationSwarm",
    "DeepResearchSwarm",
    "HybridHierarchicalClusterSwarm",
    "get_agents_info",
    "get_swarms_info",
    "AutoSwarmBuilder",
    "CouncilAsAJudge",
    "batch_agent_execution",
    "aggregate",
    "find_agent_by_name",
    "run_agent",
    "InteractiveGroupChat",
    "round_robin_speaker",
    "random_speaker",
    "priority_speaker",
    "random_dynamic_speaker",
    "HierarchicalSwarm",
    "HeavySwarm",
    "CronJob",
    "create_default_config_file",
    "disable_board_feature",
    "disable_verbose_logging",
    "enable_board_feature",
    "enable_verbose_logging",
    "get_board_config",
    "get_default_board_template",
    "is_board_feature_enabled",
    "set_board_model",
    "set_board_size",
    "set_decision_threshold",
]
