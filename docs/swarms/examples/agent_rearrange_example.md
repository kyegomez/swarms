# AgentRearrange Example

!!! abstract "Overview"
    Learn how to create flexible multi-agent workflows using `AgentRearrange`. Define custom flow patterns with sequential execution (`->`) and concurrent execution (`,`) to orchestrate agents in sophisticated workflows.

## Prerequisites

!!! info "Before You Begin"
    Make sure you have:
    - Python 3.7+ installed
    - A valid API key for your model provider
    - The Swarms package installed

## Installation

```bash
pip3 install -U swarms
```

## Environment Setup

!!! tip "API Key Configuration"
    Set your API key in the `.env` file:
    ```bash
    OPENAI_API_KEY="your-api-key-here"
    ```

## Code Implementation

### Import Required Modules

```python
from swarms import Agent, AgentRearrange
```

### Configure Agents

!!! example "Agent Configuration"
    Here's how to set up your specialized agents:

    ```python
    # Research Agent
    researcher = Agent(
        agent_name="Researcher",
        system_prompt="You are a research specialist. Gather information, analyze data, and provide comprehensive findings.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    # Writer Agent
    writer = Agent(
        agent_name="Writer",
        system_prompt="You are a professional writer. Create clear and engaging content based on research findings.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    # Editor Agent
    editor = Agent(
        agent_name="Editor",
        system_prompt="You are an expert editor. Review content for clarity, accuracy, and style.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )
    ```

### Initialize AgentRearrange

!!! example "Workflow Setup"
    Configure AgentRearrange with your agents and flow pattern:

    ```python
    # Sequential flow: Researcher -> Writer -> Editor
    flow = "Researcher -> Writer -> Editor"

    workflow = AgentRearrange(
        name="content-creation-workflow",
        agents=[researcher, writer, editor],
        flow=flow,
        max_loops=1,
    )
    ```

### Run the Workflow

!!! example "Execute the Workflow"
    Start the workflow:

    ```python
    result = workflow.run(
        "Research and write a comprehensive article about the impact of AI on healthcare"
    )
    print(result)
    ```

## Complete Example

!!! success "Full Implementation"
    Here's the complete code combined:

    ```python
    from swarms import Agent, AgentRearrange

    # Create agents
    researcher = Agent(
        agent_name="Researcher",
        system_prompt="You are a research specialist. Gather information, analyze data, and provide comprehensive findings.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    writer = Agent(
        agent_name="Writer",
        system_prompt="You are a professional writer. Create clear and engaging content based on research findings.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    editor = Agent(
        agent_name="Editor",
        system_prompt="You are an expert editor. Review content for clarity, accuracy, and style.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    # Define flow pattern
    flow = "Researcher -> Writer -> Editor"

    # Create workflow
    workflow = AgentRearrange(
        name="content-creation-workflow",
        agents=[researcher, writer, editor],
        flow=flow,
        max_loops=1,
    )

    # Execute workflow
    result = workflow.run(
        "Research and write a comprehensive article about the impact of AI on healthcare"
    )
    print(result)
    ```

## Flow Pattern Examples

!!! info "Flow Pattern Syntax"
    - **Sequential**: `"Agent1 -> Agent2 -> Agent3"` - Agents run one after another
    - **Parallel**: `"Agent1, Agent2 -> Agent3"` - Agent1 and Agent2 run simultaneously, then Agent3
    - **Mixed**: `"Agent1 -> Agent2, Agent3 -> Agent4"` - Combine sequential and parallel execution

## Configuration Options

!!! info "Key Parameters"
    | Parameter | Description | Default |
    |-----------|-------------|---------|
    | `agents` | List of Agent objects | Required |
    | `flow` | Flow pattern string defining execution order | Required |
    | `max_loops` | Maximum number of execution loops | 1 |
    | `team_awareness` | Enable sequential awareness for agents | False |

## Next Steps

!!! tip "What to Try Next"
    1. Experiment with parallel execution: `"Agent1, Agent2 -> Agent3"`
    2. Enable `team_awareness=True` for better agent coordination
    3. Try more complex flows combining sequential and parallel patterns
    4. Use SwarmRouter with `swarm_type="AgentRearrange"` for unified interface

## Troubleshooting

!!! warning "Common Issues"
    - Ensure agent names in flow match `agent_name` exactly
    - Check for typos in agent names
    - Verify all agents in flow are included in agents list
    - Enable verbose mode for debugging: `verbose=True`
