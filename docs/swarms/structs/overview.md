# Multi-Agent Architectures Overview

This page provides a comprehensive overview of all available multi-agent architectures in Swarms, their use cases, and functionality.

## Architecture Comparison

=== "Core Architectures"
    | Architecture | Use Case | Key Functionality | Documentation |
    |-------------|----------|-------------------|---------------|
    | MajorityVoting | Decision making through consensus | Combines multiple agent opinions and selects the most common answer | [Docs](majorityvoting.md) |
    | AgentRearrange | Optimizing agent order | Dynamically reorders agents based on task requirements | [Docs](agent_rearrange.md) |
    | RoundRobin | Equal task distribution | Cycles through agents in a fixed order | [Docs](round_robin_swarm.md) |
    | Mixture of Agents | Complex problem solving | Combines diverse expert agents for comprehensive analysis | [Docs](moa.md) |
    | GroupChat | Collaborative discussions | Simulates group discussions with multiple agents | [Docs](group_chat.md) |
    | AgentRegistry | Agent management | Central registry for managing and accessing agents | [Docs](agent_registry.md) |
    | SpreadSheetSwarm | Data processing | Collaborative data processing and analysis | [Docs](spreadsheet_swarm.md) |
    | ForestSwarm | Hierarchical decision making | Tree-like structure for complex decision processes | [Docs](forest_swarm.md) |
    | SwarmRouter | Task routing | Routes tasks to appropriate agents based on requirements | [Docs](swarm_router.md) |
    | TaskQueueSwarm | Task management | Manages and prioritizes tasks in a queue | [Docs](taskqueue_swarm.md) |
    | SwarmRearrange | Dynamic swarm optimization | Optimizes swarm configurations for specific tasks | [Docs](swarm_rearrange.md) |
    | MultiAgentRouter | Advanced task routing | Routes tasks to specialized agents based on capabilities | [Docs](multi_agent_router.md) |
    | AgentRouter | Embedding-based routing | Routes tasks to agents using semantic similarity on embeddings | [Docs](agent_router.md) |
    | MatrixSwarm | Parallel processing | Matrix-based organization for parallel task execution | [Docs](matrix_swarm.md) |
    | ModelRouter | Model selection | Routes tasks to appropriate AI models | [Docs](model_router.md) |
    | Deep Research Swarm | Research automation | Conducts comprehensive research across multiple domains | [Docs](deep_research_swarm.md) |
    | Swarm Matcher | Agent matching | Matches tasks with appropriate agent combinations | [Docs](swarm_matcher.md) |

=== "Workflow Architectures"
    | Architecture | Use Case | Key Functionality | Documentation |
    |-------------|----------|-------------------|---------------|
    | ConcurrentWorkflow | Parallel task execution | Executes multiple tasks simultaneously | [Docs](concurrentworkflow.md) |
    | SequentialWorkflow | Step-by-step processing | Executes tasks in a specific sequence | [Docs](sequential_workflow.md) |
    | GraphWorkflow | Complex task dependencies | Manages tasks with complex dependencies | [Docs](graph_workflow.md) |

=== "Hierarchical Architectures"
    | Architecture | Use Case | Key Functionality | Documentation |
    |-------------|----------|-------------------|---------------|
    | HierarchicalSwarm | Hierarchical task orchestration | Director agent coordinates specialized worker agents | [Docs](hierarchical_swarm.md) |
    | Hybrid Hierarchical-Cluster Swarm | Complex organization | Combines hierarchical and cluster-based organization | [Docs](hhcs.md) |
    | Auto Swarm Builder | Automated swarm creation | Automatically creates and configures swarms | [Docs](auto_swarm_builder.md) |

## Communication Structure

!!! note "Communication Protocols"
    The [Conversation](conversation.md) documentation details the communication protocols and structures used between agents in these architectures.

## Choosing the Right Architecture

When selecting a multi-agent architecture, consider the following factors:

!!! tip "Task Complexity"
    Simple tasks may only need basic architectures like RoundRobin, while complex tasks might require Hierarchical or Graph-based approaches.

!!! tip "Parallelization Needs"
    If tasks can be executed in parallel, consider ConcurrentWorkflow or MatrixSwarm.

!!! tip "Decision Making Requirements"
    For consensus-based decisions, MajorityVoting is ideal.

!!! tip "Resource Optimization"
    If you need to optimize agent usage, consider SwarmRouter or TaskQueueSwarm.

!!! tip "Dynamic Adaptation"
    For tasks requiring dynamic adaptation, consider SwarmRearrange or Auto Swarm Builder.

For more detailed information about each architecture, please refer to their respective documentation pages.
