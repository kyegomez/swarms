# Multi-Agent Architectures

Each multi-agent architecture type is designed for specific use cases and can be combined to create powerful multi-agent systems. Here's a comprehensive overview of each available swarm:

| Swarm Type           | Description                                                                  | Learn More |
|---------------------|------------------------------------------------------------------------------|------------|
| AgentRearrange      | Dynamically reorganizes agents to optimize task performance and efficiency. Optimizes agent performance by dynamically adjusting their roles and positions within the workflow. This architecture is particularly useful when the effectiveness of agents depends on their sequence or arrangement. | [Learn More](/swarms/structs/agent_rearrange) |
| MixtureOfAgents     | Creates diverse teams of specialized agents, each bringing unique capabilities to solve complex problems. Each agent contributes unique skills to achieve the overall goal, making it excel at tasks requiring multiple types of expertise or processing. | [Learn More](/swarms/structs/moa) |
| SpreadSheetSwarm    | Provides a structured approach to data management and operations, making it ideal for tasks involving data analysis, transformation, and systematic processing in a spreadsheet-like structure. | [Learn More](/swarms/structs/spreadsheet_swarm) |
| SequentialWorkflow  | Ensures strict process control by executing tasks in a predefined order. Perfect for workflows where each step depends on the completion of previous steps. | [Learn More](/swarms/structs/sequential_workflow) |
| ConcurrentWorkflow  | Maximizes efficiency by running independent tasks in parallel, significantly reducing overall processing time for complex operations. Ideal for independent tasks that can be processed simultaneously. | [Learn More](/swarms/structs/concurrentworkflow) |
| GroupChat           | Enables dynamic collaboration between agents through a chat-based interface, facilitating real-time information sharing and decision-making. | [Learn More](/swarms/structs/group_chat) |
| MultiAgentRouter    | Acts as an intelligent task dispatcher, ensuring optimal distribution of work across available agents based on their capabilities and current workload. | [Learn More](/swarms/structs/multi_agent_router) |
| AutoSwarmBuilder    | Simplifies swarm creation by automatically configuring agent architectures based on task requirements and performance metrics. | [Learn More](/swarms/structs/auto_swarm_builder) |
| HiearchicalSwarm    | Implements a structured approach to task management, with clear lines of authority and delegation across multiple agent levels. | [Learn More](/swarms/structs/multi_swarm_orchestration) |
| auto               | Provides intelligent swarm selection based on context, automatically choosing the most effective architecture for given tasks. | [Learn More](/swarms/concept/how_to_choose_swarms) |
| MajorityVoting     | Implements robust decision-making through consensus, particularly useful for tasks requiring collective intelligence or verification. | [Learn More](/swarms/structs/majorityvoting) |
| MALT              | Specialized framework for language-based tasks, optimizing agent collaboration for complex language processing operations. | [Learn More](/swarms/structs/malt) |

# Learn More

To learn more about Swarms architecture and how different swarm types work together, visit our comprehensive guides:

- [Introduction to Multi-Agent Architectures](/swarms/concept/swarm_architectures)

- [How to Choose the Right Multi-Agent Architecture](/swarms/concept/how_to_choose_swarms)

- [Framework Architecture Overview](/swarms/concept/framework_architecture)

- [Building Custom Swarms](/swarms/structs/custom_swarm)
