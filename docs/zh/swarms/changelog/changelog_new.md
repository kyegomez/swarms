# 🚀 Swarms 5.9.2 Release Notes


### 🎯 Major Features

#### Concurrent Agent Execution Suite
We're excited to introduce a comprehensive suite of agent execution methods to supercharge your multi-agent workflows:

- `run_agents_concurrently`: Execute multiple agents in parallel with optimal resource utilization
- `run_agents_concurrently_async`: Asynchronous execution for improved performance
- `run_single_agent`: Streamlined single agent execution
- `run_agents_concurrently_multiprocess`: Multi-process execution for CPU-intensive tasks
- `run_agents_sequentially`: Sequential execution with controlled flow
- `run_agents_with_different_tasks`: Assign different tasks to different agents
- `run_agent_with_timeout`: Time-bounded agent execution
- `run_agents_with_resource_monitoring`: Monitor and manage resource usage

### 📚 Documentation
- Comprehensive documentation added for all new execution methods
- Updated examples and usage patterns
- Enhanced API reference

### 🛠️ Improvements
- Tree swarm implementation fixes
- Workspace directory now automatically set to `agent_workspace`
- Improved error handling and stability

## Quick Start

```python
from swarms import Agent, run_agents_concurrently, run_agents_with_timeout, run_agents_with_different_tasks

# Initialize multiple agents
agents = [
    Agent(
        agent_name=f"Analysis-Agent-{i}",
        system_prompt="You are a financial analysis expert",
        llm=model,
        max_loops=1
    )
    for i in range(5)
]

# Run agents concurrently
task = "Analyze the impact of rising interest rates on tech stocks"
outputs = run_agents_concurrently(agents, task)

# Example with timeout
outputs_with_timeout = run_agents_with_timeout(
    agents=agents,
    task=task,
    timeout=30.0,
    batch_size=2
)

# Run different tasks
task_pairs = [
    (agents[0], "Analyze tech stocks"),
    (agents[1], "Analyze energy stocks"),
    (agents[2], "Analyze retail stocks")
]
different_outputs = run_agents_with_different_tasks(task_pairs)
```

## Installation
```bash
pip3 install -U swarms
```

## Coming Soon
- 🌟 Auto Swarm Builder: Automatically construct and configure entire swarms from a single task specification (in development)
- Auto Prompt Generator for thousands of agents (in development)

## Community
We believe in the power of community-driven development. Help us make Swarms better!

- ⭐ Star our repository: https://github.com/kyegomez/swarms
- 🔄 Fork the project and contribute your improvements
- 🤝 Join our growing community of contributors

## Bug Fixes
- Fixed Tree Swarm implementation issues
- Resolved workspace directory configuration problems
- General stability improvements

---

For detailed documentation and examples, visit our [GitHub repository](https://github.com/kyegomez/swarms).

Let's build the future of multi-agent systems together! 🚀