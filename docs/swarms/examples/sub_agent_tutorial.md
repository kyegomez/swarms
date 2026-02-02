# Sub-Agent Delegation Tutorial

This tutorial explains how to use sub-agent delegation with the autonomous agent. When `max_loops="auto"`, the main agent can create specialized sub-agents, assign tasks to them, and aggregate results. Sub-agents run concurrently via `asyncio` and are cached on the main agent for reuse.

## What are Sub-Agents?

Sub-agents are agent instances created at runtime by the coordinator via the `create_sub_agent` tool. They are stored in the main agent's `sub_agents` dictionary and can be used repeatedly via `assign_task`. They provide:

- **Parallel execution**: Multiple sub-agents run at once using `asyncio.to_thread` and `asyncio.gather`

- **Specialization**: Each sub-agent has its own `agent_name`, `agent_description`, and optional `system_prompt`

- **Caching**: Sub-agents are keyed by ID (e.g. `sub-agent-{uuid}`) and reused across 
assignments

- **Same LLM**: Sub-agents use the parent's `model_name` and run with `max_loops=1`, `print_on=False`

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

Each sub-agent is created with:

| Parameter          | Required? | Description                                                                                   |
|--------------------|-----------|-----------------------------------------------------------------------------------------------|
| `agent_name`       | Yes       | Descriptive identifier for the sub-agent                                                      |
| `agent_description`| Yes       | Role and capabilities of the sub-agent                                                        |
| `system_prompt`    | No        | Custom instructions for the sub-agent (if omitted, defaults to the agent description prompt)  |

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

**Result:** Each sub-agent is created, cached in `agent.sub_agents`, and assigned a unique ID of the form `sub-agent-{uuid.uuid4().hex[:8]}` (e.g. `sub-agent-a1b2c3d4`). The handler returns a success message listing created agents and their IDs.

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

**Result:** All sub-agents run concurrently via `asyncio.to_thread(sub_agent.run, task)` and `asyncio.gather`. When `wait_for_completion` is true (default), the tool returns formatted results; when false, it dispatches tasks and returns immediately (fire-and-forget).

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

The tools are defined in `swarms.structs.autonomous_loop_utils` and wired in the agent's autonomous planning tool handlers.

### create_sub_agent

Creates one or more sub-agents and caches them on the main agent's `sub_agents` dictionary.

**Top-level parameters:**

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| agents    | array  | Yes      | List of sub-agent specifications. Each item is an object with the fields below. |

**Fields for each item in `agents`:**

| Field               | Type   | Required | Description |
|---------------------|--------|----------|-------------|
| agent_name          | string | Yes      | Name of the sub-agent. |
| agent_description   | string | Yes      | Role and capabilities of the sub-agent. |
| system_prompt       | string | No       | Custom system prompt. If omitted, a default based on the agent description is used. |

**Handler behavior (`create_sub_agent_tool`):** For each spec, a new `Agent` is created with `id=sub-agent-{uuid.uuid4().hex[:8]}`, `agent_name`, `agent_description`, `system_prompt`, the parent's `model_name`, `max_loops=1`, and `print_on=False`. The sub-agent is stored in `agent.sub_agents[agent_id]` with keys `agent`, `name`, `description`, `system_prompt`, `created_at`. Returns a success message listing created agents and their IDs.

### assign_task

Assigns tasks to one or more sub-agents. Tasks are run asynchronously via `asyncio.to_thread(sub_agent.run, task)` and gathered with `asyncio.gather`.

**Top-level parameters:**

| Parameter           | Type    | Required | Description |
|---------------------|---------|----------|-------------|
| assignments         | array   | Yes      | List of task assignments. Each item is an object with the fields below. |
| wait_for_completion | boolean | No       | If true (default), wait for all tasks and return formatted results. If false, dispatch tasks and return immediately. |

**Fields for each item in `assignments`:**

| Field     | Type   | Required | Description |
|-----------|--------|----------|-------------|
| agent_id  | string | Yes      | ID of the sub-agent (from `create_sub_agent` output). |
| task      | string | Yes      | Task description for the sub-agent. |
| task_id   | string | No       | Identifier for this assignment; defaults to `task-{idx + 1}`. |

**Handler behavior (`assign_task_tool`):** Validates that `agent.sub_agents` exists and that each `agent_id` is in it. Runs each assignment concurrently. On success, each result has `status: "success"` and `result`; on error, `status: "error"` and `error`. When `wait_for_completion` is true, returns a formatted string of all results; otherwise returns a dispatch confirmation.

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

## Troubleshooting

### Sub-agents not created

Ensure `selected_tools="all"` or include `"create_sub_agent"` and `"assign_task"` in the selected tools list. If the agent returns "Error: Each agent must have agent_name and agent_description", every entry in the `agents` array must have both `agent_name` and `agent_description`.

### No sub-agents have been created

If `assign_task` returns "Error: No sub-agents have been created. Use create_sub_agent first.", call `create_sub_agent` before `assign_task`. The main agent must create and cache sub-agents before assigning work.

### Sub-agent not found

If you see "Error: Sub-agent with ID '...' not found. Available agents: [...]", the `agent_id` in the assignment does not match any key in `agent.sub_agents`. Use the exact IDs returned from `create_sub_agent` (e.g. `sub-agent-a1b2c3d4`).

### Tasks not waiting for completion

When `wait_for_completion` is true (default), the tool waits for all sub-agent runs and returns formatted results. When false, it dispatches tasks and returns "Dispatched N task(s) to sub-agents (async mode)" without waiting.

## Performance Considerations

### Optimal Number of Sub-Agents

| Number of Sub-Agents | Recommended Use                                |
|----------------------|------------------------------------------------|
| 3-5                  | Ideal for most tasks                           |
| 5-10                 | Good for complex multi-domain projects         |
| 10+                  | May increase coordination overhead             |


## Summary

Sub-agent delegation allows an autonomous agent to divide complex tasks among specialized sub-agents, improving efficiency and scalability. When enabled, the main agent can dynamically create sub-agents with defined roles and assign them tasks to work on in parallel. Sub-agents are managed and reused by the coordinator, and their results are gathered and synthesized for the overall task outcome. This approach is ideal for scenarios requiring concurrent execution, expertise in multiple domains, or efficient handling of multi-step problems.
