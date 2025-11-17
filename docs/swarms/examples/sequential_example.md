# Sequential Workflow Example

!!! abstract "Overview"
    Learn how to create a sequential workflow with multiple specialized AI agents using the Swarms framework. This example demonstrates how to set up a legal practice workflow with different types of legal agents working in sequence.

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
from swarms import Agent, SequentialWorkflow
```

### Configure Agents

!!! example "Legal Agent Configuration"
    Here's how to set up your specialized legal agents:

    ```python
    # Litigation Agent
    litigation_agent = Agent(
        agent_name="Alex Johnson",
        system_prompt="As a Litigator, you specialize in navigating the complexities of lawsuits. Your role involves analyzing intricate facts, constructing compelling arguments, and devising effective case strategies to achieve favorable outcomes for your clients.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    # Corporate Attorney Agent
    corporate_agent = Agent(
        agent_name="Emily Carter",
        system_prompt="As a Corporate Attorney, you provide expert legal advice on business law matters. You guide clients on corporate structure, governance, compliance, and transactions, ensuring their business operations align with legal requirements.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    # IP Attorney Agent
    ip_agent = Agent(
        agent_name="Michael Smith",
        system_prompt="As an IP Attorney, your expertise lies in protecting intellectual property rights. You handle various aspects of IP law, including patents, trademarks, copyrights, and trade secrets, helping clients safeguard their innovations.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )
    ```

### Initialize Sequential Workflow

!!! example "Workflow Setup"
    Configure the SequentialWorkflow with your agents:

    ```python
    swarm = SequentialWorkflow(
        agents=[litigation_agent, corporate_agent, ip_agent],
        name="litigation-practice",
        description="Handle all aspects of litigation with a focus on thorough legal analysis and effective case management.",
    )
    ```

### Run the Workflow

!!! example "Execute the Workflow"
    Start the sequential workflow:

    ```python
    swarm.run("Create a report on how to patent an all-new AI invention and what platforms to use and more.")
    ```

## Complete Example

!!! success "Full Implementation"
    Here's the complete code combined:

    ```python
    from swarms import Agent, SequentialWorkflow

    # Core Legal Agent Definitions with enhanced system prompts
    litigation_agent = Agent(
        agent_name="Alex Johnson",
        system_prompt="As a Litigator, you specialize in navigating the complexities of lawsuits. Your role involves analyzing intricate facts, constructing compelling arguments, and devising effective case strategies to achieve favorable outcomes for your clients.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    corporate_agent = Agent(
        agent_name="Emily Carter",
        system_prompt="As a Corporate Attorney, you provide expert legal advice on business law matters. You guide clients on corporate structure, governance, compliance, and transactions, ensuring their business operations align with legal requirements.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    ip_agent = Agent(
        agent_name="Michael Smith",
        system_prompt="As an IP Attorney, your expertise lies in protecting intellectual property rights. You handle various aspects of IP law, including patents, trademarks, copyrights, and trade secrets, helping clients safeguard their innovations.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    # Initialize and run the workflow
    swarm = SequentialWorkflow(
        agents=[litigation_agent, corporate_agent, ip_agent],
        name="litigation-practice",
        description="Handle all aspects of litigation with a focus on thorough legal analysis and effective case management.",
    )

    swarm.run("Create a report on how to patent an all-new AI invention and what platforms to use and more.")
    ```

## Agent Roles

!!! info "Specialized Legal Agents"
    | Agent | Role | Expertise |
    |-------|------|-----------|
    | Alex Johnson | Litigator | Lawsuit navigation, case strategy |
    | Emily Carter | Corporate Attorney | Business law, compliance |
    | Michael Smith | IP Attorney | Patents, trademarks, copyrights |

## Configuration Options

!!! info "Key Parameters"
    | Parameter | Description | Default |
    |-----------|-------------|---------|
    | `agent_name` | Human-readable name for the agent | Required |
    | `system_prompt` | Detailed role description and expertise | Required |
    | `model_name` | LLM model to use | "gpt-4o-mini" |
    | `max_loops` | Maximum number of processing loops | 1 |

## Next Steps

!!! tip "What to Try Next"
    1. Experiment with different agent roles and specializations
    2. Modify the system prompts to create different expertise areas
    3. Add more agents to the workflow for complex tasks
    4. Try different model configurations

## Troubleshooting

!!! warning "Common Issues"
    - Ensure your API key is correctly set in the `.env` file
    
    - Check that all required dependencies are installed
    
    - Verify that your model provider's API is accessible
    
    - Monitor agent responses for quality and relevance
