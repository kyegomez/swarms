import importlib
from typing import Any

_MODULE_MAP = {
    "A2AClient": "a2a_client",
    "A2AServer": "a2a_server",
    "AdvisorSwarm": "advisor_swarm",
    "Agent": "agent",
    "AgentLoader": "agent_loader",
    "AgentMailChannel": "agentmail_channel",
    "AgentRearrange": "agent_rearrange",
    "AutoAgentBuilder": "auto_agent_builder",
    "AutoSwarmBuilder": "auto_swarm_builder",
    "BatchedGridWorkflow": "batched_grid_workflow",
    "Broadcast": "broadcast",
    "CommsChannel": "comms_channel",
    "ConcurrentWorkflow": "concurrent_workflow",
    "Conversation": "conversation",
    "CouncilAsAJudge": "council_as_judge",
    "CronJob": "cron_job",
    "DebateWithJudge": "debate_with_judge",
    "DecisionModel": "decision_model",
    "Edge": "graph_workflow",
    "EvaluationReport": "planner_generator_evaluator",
    "FileChannel": "file_channel",
    "GraphWorkflow": "graph_workflow",
    "GroupChat": "groupchat",
    "HarnessResult": "planner_generator_evaluator",
    "HeavySwarm": "heavy_swarm",
    "HierarchicalSwarm": "hiearchical_swarm",
    "HttpChannel": "http_channel",
    "HybridHierarchicalClusterSwarm": "hybrid_hiearchical_peer_swarm",
    "LLMCouncil": "llm_council",
    "MCPDeployer": "mcp_deployer",
    "MajorityVoting": "majority_voting",
    "MixtureOfAgents": "mixture_of_agents",
    "ModelRouter": "model_router",
    "MultiAgentRouter": "multi_agent_router",
    "Node": "graph_workflow",
    "NodeType": "graph_workflow",
    "OneToOne": "one_to_one",
    "OneToThree": "one_to_three",
    "PlannerGeneratorEvaluator": "planner_generator_evaluator",
    "PlannerWorkerSwarm": "planner_worker_swarm",
    "QueueChannel": "queue_channel",
    "RESPOND_TOOL": "groupchat",
    "RoundRobinSwarm": "round_robin",
    "SelfMoASeq": "self_moa_seq",
    "SequentialWorkflow": "sequential_workflow",
    "SmtpChannel": "smtp_channel",
    "SocialAlgorithms": "social_algorithms",
    "SpreadSheetSwarm": "spreadsheet_swarm",
    "SqliteChannel": "sqlite_channel",
    "StepContract": "planner_generator_evaluator",
    "SubagentRegistry": "async_subagent",
    "SubagentTask": "async_subagent",
    "SwarmRouter": "swarm_router",
    "SwarmType": "swarm_router",
    "TaskStatus": "async_subagent",
    "aggregate": "ma_blocks",
    "batch_agent_execution": "batch_agent_execution",
    "batched_grid_agent_execution": "multi_agent_exec",
    "broadcast": "broadcast",
    "circular_swarm": "swarming_architectures",
    "deploy_as_mcp": "mcp_deployer",
    "find_agent_by_name": "ma_blocks",
    "get_agents_info": "multi_agent_exec",
    "get_available_models": "check_models",
    "get_decision_model_prices": "decision_model",
    "get_decision_models": "decision_model",
    "get_swarms_info": "multi_agent_exec",
    "grid_swarm": "swarming_architectures",
    "is_model_available": "check_models",
    "load_agent_from_markdown": "agent_loader",
    "load_agents_from_markdown": "agent_loader",
    "mesh_swarm": "swarming_architectures",
    "model_count": "check_models",
    "one_to_one": "one_to_one",
    "one_to_three": "one_to_three",
    "pyramid_swarm": "swarming_architectures",
    "rearrange": "agent_rearrange",
    "run_agent": "ma_blocks",
    "run_agent_async": "multi_agent_exec",
    "run_agents_concurrently": "multi_agent_exec",
    "run_agents_concurrently_async": "multi_agent_exec",
    "run_agents_concurrently_multiprocess": "multi_agent_exec",
    "run_agents_with_different_tasks": "multi_agent_exec",
    "run_single_agent": "multi_agent_exec",
    "star_swarm": "swarming_architectures",
}

__all__ = sorted(_MODULE_MAP.keys())


def __getattr__(name: str) -> Any:
    module_name = _MODULE_MAP.get(name)
    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(list(globals().keys()) + __all__)
