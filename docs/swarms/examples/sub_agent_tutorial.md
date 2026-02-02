# Sub-Agent Delegation Tutorial

This tutorial teaches you how to use sub-agent delegation to create autonomous agents that can break down complex tasks and distribute work across specialized sub-agents running in parallel.

## What are Sub-Agents?

Sub-agents are specialized agent instances created dynamically by a main "coordinator" agent to handle specific subtasks. They enable:

- **Parallel Task Execution**: Multiple sub-agents work simultaneously
- **Domain Specialization**: Each sub-agent can focus on a specific area
- **Dynamic Scaling**: Create as many sub-agents as needed for the task
- **Reusability**: Sub-agents are cached and can handle multiple assignments

## Prerequisites

```bash
pip install -U swarms
```

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

## Basic Concepts

### The Coordinator Agent

The coordinator (main agent) is responsible for:
1. Analyzing the main task
2. Creating specialized sub-agents
3. Delegating work to sub-agents
4. Aggregating results

### Sub-Agents

Sub-agents are created with:
- **Name**: Descriptive identifier
- **Description**: Role and capabilities
- **System Prompt** (optional): Custom instructions

## Quick Start Example

Here's a simple example that creates sub-agents for parallel research:

```python
from swarms.structs.agent import Agent

# Create the coordinator agent
coordinator = Agent(
    agent_name="Research-Coordinator",
    model_name="gpt-4o",
    max_loops="auto",  # Enable autonomous mode
    interactive=False,
    selected_tools="all",  # Enable all tools including sub-agent tools
)

# Define a task requiring parallel work
task = """
Research three topics in parallel:
1. Latest trends in artificial intelligence
2. Recent quantum computing breakthroughs  
3. Advances in renewable energy

Create a sub-agent for each topic, assign research tasks to them,
and compile a comprehensive summary of all findings.
"""

# Run the coordinator - it will automatically create and manage sub-agents
result = coordinator.run(task)
print(result)
```

## How It Works

When you run the coordinator agent with a task requiring delegation:

### Step 1: Agent Creates Sub-Agents

The coordinator automatically calls the `create_sub_agent` tool:

```python
# This happens automatically inside the agent
create_sub_agent({
    "agents": [
        {
            "agent_name": "AI-Research-Agent",
            "agent_description": "Expert in artificial intelligence research and trends",
            "system_prompt": "You are an AI research specialist..." # Optional
        },
        {
            "agent_name": "Quantum-Research-Agent",
            "agent_description": "Expert in quantum computing and related technologies"
        },
        {
            "agent_name": "Energy-Research-Agent",
            "agent_description": "Expert in renewable energy and sustainability"
        }
    ]
})
```

**Result:** Each sub-agent is created, cached, and assigned a unique ID (e.g., `sub-agent-a1b2c3d4`)

### Step 2: Agent Assigns Tasks

The coordinator then assigns work using the `assign_task` tool:

```python
# This happens automatically inside the agent
assign_task({
    "assignments": [
        {
            "agent_id": "sub-agent-a1b2c3d4",
            "task": "Research the latest trends in artificial intelligence",
            "task_id": "ai-research"
        },
        {
            "agent_id": "sub-agent-e5f6g7h8",
            "task": "Research recent quantum computing breakthroughs",
            "task_id": "quantum-research"
        },
        {
            "agent_id": "sub-agent-i9j0k1l2",
            "task": "Research advances in renewable energy",
            "task_id": "energy-research"
        }
    ],
    "wait_for_completion": true
})
```

**Result:** All three sub-agents execute their tasks in parallel using `asyncio`

### Step 3: Results Aggregation

The coordinator receives results from all sub-agents:

```
Completed 3 task assignment(s):

[AI-Research-Agent] Task ai-research:
Result: Recent AI trends include...

[Quantum-Research-Agent] Task quantum-research:
Result: Major quantum computing breakthroughs...

[Energy-Research-Agent] Task energy-research:
Result: Renewable energy advances...
```

## Tool Parameters Reference

### create_sub_agent

Creates one or more sub-agents for delegation.

**Parameters:**

```python
{
    "agents": [  # Array of sub-agent specifications
        {
            "agent_name": str,           # Required: Name of the sub-agent
            "agent_description": str,    # Required: Role and capabilities
            "system_prompt": str         # Optional: Custom instructions
        }
    ]
}
```

**Returns:** Success message with created agent IDs

### assign_task

Assigns tasks to sub-agents for execution.

**Parameters:**

```python
{
    "assignments": [  # Array of task assignments
        {
            "agent_id": str,      # Required: Sub-agent ID from create_sub_agent
            "task": str,          # Required: Task description
            "task_id": str        # Optional: Identifier for tracking
        }
    ],
    "wait_for_completion": bool  # Optional: Wait for results (default: true)
}
```

**Returns:** Results from all sub-agent executions

## Advanced Example

Here's a more complex example with custom system prompts:

```python
from swarms.structs.agent import Agent

# Create a sophisticated coordinator
coordinator = Agent(
    agent_name="Advanced-Research-Coordinator",
    agent_description="Manages complex multi-domain research projects",
    model_name="gpt-4o",
    max_loops="auto",
    interactive=False,
    verbose=True,  # See detailed execution
)

# Complex multi-stage task
task = """
Conduct a comprehensive analysis of the technology landscape:

Phase 1: Create specialized sub-agents
- Create an AI/ML expert agent with deep knowledge of machine learning
- Create a cybersecurity expert agent
- Create a cloud computing expert agent

Phase 2: Parallel research
- Assign each agent to research their domain's latest trends
- Request detailed analysis including challenges and opportunities

Phase 3: Integration
- Compile findings into a unified technology landscape report
- Identify cross-domain synergies and opportunities
"""

# Execute the multi-phase task
result = coordinator.run(task)
print(result)
```

## Use Cases

### 1. Research & Analysis

Break down research topics into parallel investigations:

```python
task = """
Research the impact of climate change on three sectors:
1. Agriculture
2. Healthcare  
3. Technology infrastructure

Create expert sub-agents for each sector and compile findings.
"""
```

### 2. Content Creation

Distribute content creation across specialized writers:

```python
task = """
Create a comprehensive guide on Python web development:
1. Backend development (Django/Flask expert)
2. Frontend integration (React expert)
3. DevOps and deployment (Infrastructure expert)

Create sub-agents for each area and combine into a cohesive guide.
"""
```

### 3. Data Processing

Parallel processing of different data sources:

```python
task = """
Analyze market data from three sources:
1. Social media sentiment (Twitter, Reddit)
2. News articles (Financial news)
3. Technical indicators (Stock charts)

Create specialized sub-agents for each data source.
"""
```

### 4. Software Development Tasks

Distribute development work:

```python
task = """
Plan and design a new feature:
1. Frontend design and UX (Design expert)
2. Backend API architecture (API expert)
3. Database schema (Database expert)
4. Testing strategy (QA expert)

Create sub-agents for each role and compile a comprehensive plan.
"""
```

## Best Practices

### 1. Clear Sub-Agent Descriptions

Provide detailed descriptions for better specialization:

```python
{
    "agent_name": "Financial-Analyst",
    "agent_description": "Expert financial analyst specializing in equity research, valuation models, and market trend analysis. Focuses on quantitative metrics and data-driven insights."
}
```

### 2. Use Custom System Prompts

Guide sub-agent behavior with specific instructions:

```python
{
    "agent_name": "Code-Reviewer",
    "agent_description": "Expert code reviewer focusing on best practices",
    "system_prompt": "You are a senior code reviewer. Focus on: code quality, security vulnerabilities, performance optimization, and maintainability. Provide actionable feedback with examples."
}
```

### 3. Task Clarity

Be specific in task assignments:

```python
{
    "agent_id": "sub-agent-123",
    "task": "Analyze the Q4 2025 financial performance of tech companies, focusing on revenue growth, profit margins, and R&D spending. Provide comparison with previous quarters.",
    "task_id": "q4-tech-analysis"
}
```

### 4. Enable Logging

Use `verbose=True` to monitor sub-agent activity:

```python
coordinator = Agent(
    agent_name="Coordinator",
    model_name="gpt-4o",
    max_loops="auto",
    verbose=True,  # See sub-agent creation and task assignment
)
```

## Troubleshooting

### Issue: Sub-agents not created

**Solution:** Ensure `selected_tools="all"` or include both tools explicitly:

```python
coordinator = Agent(
    agent_name="Coordinator",
    model_name="gpt-4o",
    max_loops="auto",
    selected_tools=["create_sub_agent", "assign_task", "complete_task", ...]
)
```

### Issue: Tasks not executing in parallel

**Solution:** Verify `wait_for_completion=True` (default) for synchronous results. Sub-agents always run concurrently using asyncio.

### Issue: Sub-agent not found error

**Solution:** Ensure you're using the correct agent ID returned from `create_sub_agent`. The coordinator handles this automatically.

## Performance Considerations

### Optimal Number of Sub-Agents

- **3-5 sub-agents**: Ideal for most tasks
- **5-10 sub-agents**: Good for complex multi-domain projects
- **10+ sub-agents**: May increase coordination overhead

### Resource Usage

Each sub-agent:
- Runs with `max_loops=1` by default
- Uses same LLM as parent agent
- Has independent memory and state
- Executes concurrently (not sequentially)

## Summary

Sub-agent delegation enables:

✅ **Parallel Execution** - Multiple tasks run simultaneously  
✅ **Specialization** - Domain-specific agents for better results  
✅ **Scalability** - Handle complex tasks by distributing work  
✅ **Automation** - Coordinator manages everything automatically  
✅ **Efficiency** - Faster completion through concurrent processing

## Next Steps

- Explore [Agent Handoffs Tutorial](./agent_handoff_tutorial.md) for cross-agent communication
- Learn about [Autonomous Agent Tools](../autonomous_looper_tools.md)
- See [Multi-Agent Architectures](../../swarms/concept/swarm_architectures.md) for more patterns

## Complete Working Example

```python
from swarms.structs.agent import Agent
import os

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Create coordinator agent
coordinator = Agent(
    agent_name="Research-Team-Coordinator",
    model_name="gpt-4o",
    max_loops="auto",
    interactive=False,
    verbose=True,
    selected_tools="all",
)

# Define comprehensive research task
task = """
Conduct a comprehensive technology market analysis:

1. Create three specialized sub-agents:
   - AI/ML Market Analyst
   - Cloud Computing Analyst  
   - Cybersecurity Analyst

2. Assign each sub-agent to research:
   - Current market size and growth rate
   - Key players and market leaders
   - Emerging trends and innovations
   - Future outlook for next 3 years

3. Compile all findings into a unified market analysis report
   with cross-domain insights and investment recommendations.
"""

# Execute the task
print("Starting research coordination...")
result = coordinator.run(task)

# Display results
print("\n" + "="*80)
print("FINAL REPORT")
print("="*80)
print(result)
```

Run this example to see sub-agent delegation in action!
