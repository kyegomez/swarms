# Multi-Agent Architectures

Each multi-agent architecture type is designed for specific use cases and can be combined to create powerful multi-agent systems. Below is an overview of each available swarm type:

| Swarm Type           | Description                                                                  | Learn More |
|----------------------|------------------------------------------------------------------------------|------------|
| AgentRearrange       | Dynamically reorganizes agents to optimize task performance and efficiency. Useful when agent effectiveness depends on their sequence or arrangement. | [Learn More](swarms_api.md#agentrearrange) |
| MixtureOfAgents      | Builds diverse teams of specialized agents, each contributing unique skills to solve complex problems. Excels at tasks requiring multiple types of expertise. | [Learn More](swarms_api.md#mixtureofagents) |
| SpreadSheetSwarm     | Provides a structured approach to data management and operations, ideal for tasks involving data analysis, transformation, and systematic processing in a spreadsheet-like structure. | [Learn More](swarms_api.md#spreadsheetswarm) |
| SequentialWorkflow   | Executes tasks in a strict, predefined order. Perfect for workflows where each step depends on the completion of the previous one. | [Learn More](swarms_api.md#sequentialworkflow) |
| ConcurrentWorkflow   | Runs independent tasks in parallel, significantly reducing processing time for complex operations. Ideal for tasks that can be processed simultaneously. | [Learn More](swarms_api.md#concurrentworkflow) |
| GroupChat            | Enables dynamic collaboration between agents through a chat-based interface, facilitating real-time information sharing and decision-making. | [Learn More](swarms_api.md#groupchat) |
| MultiAgentRouter     | Acts as an intelligent task dispatcher, distributing work across agents based on their capabilities and current workload. | [Learn More](swarms_api.md#multiagentrouter) |
| AutoSwarmBuilder     | Automatically configures agent architectures based on task requirements and performance metrics, simplifying swarm creation. | [Learn More](swarms_api.md#autoswarmbuilder) |
| HierarchicalSwarm    | Implements a structured, multi-level approach to task management, with clear lines of authority and delegation. | [Learn More](swarms_api.md#hierarchicalswarm) |
| Auto                | Intelligently selects the most effective swarm architecture for a given task based on context. | [Learn More](swarms_api.md#auto) |
| MajorityVoting       | Implements robust decision-making through consensus, ideal for tasks requiring collective intelligence or verification. | [Learn More](swarms_api_tools.md#majorityvoting) |
| MALT                | Specialized framework for language-based tasks, optimizing agent collaboration for complex language processing operations. | [Learn More](swarms_api_tools.md#malt) |

# Learn More

To explore Swarms architecture and how different swarm types work together, check out our comprehensive guides:

- [Introduction to Multi-Agent Architectures](/swarms/concept/swarm_architectures)
- [How to Choose the Right Multi-Agent Architecture](/swarms/concept/how_to_choose_swarms)
- [Framework Architecture Overview](/swarms/concept/framework_architecture)
- [Building Custom Swarms](/swarms/structs/custom_swarm)
