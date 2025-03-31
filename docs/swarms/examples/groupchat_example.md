# GroupChat Example

!!! abstract "Overview"
    Learn how to create and configure a group chat with multiple AI agents using the Swarms framework. This example demonstrates how to set up agents for expense analysis and budget advising.

## Prerequisites

!!! info "Before You Begin"
    Make sure you have:
    - Python 3.7+ installed
    - A valid API key for your model provider
    - The Swarms package installed

## Installation

```bash
pip install swarms
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
from dotenv import load_dotenv
import os
from swarms import Agent, GroupChat
```

### Configure Agents

!!! example "Agent Configuration"
    Here's how to set up your agents with specific roles:

    ```python
    # Expense Analysis Agent
    agent1 = Agent(
        agent_name="Expense-Analysis-Agent",
        description="You are an accounting agent specializing in analyzing potential expenses.",
        model_name="gpt-4o-mini",
        max_loops=1,
        autosave=False,
        dashboard=False,
        verbose=True,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        retry_attempts=1,
        context_length=200000,
        output_type="string",
        streaming_on=False,
        max_tokens=15000,
    )

    # Budget Adviser Agent
    agent2 = Agent(
        agent_name="Budget-Adviser-Agent",
        description="You are a budget adviser who provides insights on managing and optimizing expenses.",
        model_name="gpt-4o-mini",
        max_loops=1,
        autosave=False,
        dashboard=False,
        verbose=True,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        retry_attempts=1,
        context_length=200000,
        output_type="string",
        streaming_on=False,
        max_tokens=15000,
    )
    ```

### Initialize GroupChat

!!! example "GroupChat Setup"
    Configure the GroupChat with your agents:

    ```python
    agents = [agent1, agent2]

    chat = GroupChat(
        name="Expense Advisory",
        description="Accounting group focused on discussing potential expenses",
        agents=agents,
        max_loops=1,
        output_type="all",
    )
    ```

### Run the Chat

!!! example "Execute the Chat"
    Start the conversation between agents:

    ```python
    history = chat.run(
        "What potential expenses should we consider for the upcoming quarter? Please collaborate to outline a comprehensive list."
    )
    ```

## Complete Example

!!! success "Full Implementation"
    Here's the complete code combined:

    ```python
    from dotenv import load_dotenv
    import os
    from swarms import Agent, GroupChat

    if __name__ == "__main__":
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

        # Configure agents
        agent1 = Agent(
            agent_name="Expense-Analysis-Agent",
            description="You are an accounting agent specializing in analyzing potential expenses.",
            model_name="gpt-4o-mini",
            max_loops=1,
            autosave=False,
            dashboard=False,
            verbose=True,
            dynamic_temperature_enabled=True,
            user_name="swarms_corp",
            retry_attempts=1,
            context_length=200000,
            output_type="string",
            streaming_on=False,
            max_tokens=15000,
        )

        agent2 = Agent(
            agent_name="Budget-Adviser-Agent",
            description="You are a budget adviser who provides insights on managing and optimizing expenses.",
            model_name="gpt-4o-mini",
            max_loops=1,
            autosave=False,
            dashboard=False,
            verbose=True,
            dynamic_temperature_enabled=True,
            user_name="swarms_corp",
            retry_attempts=1,
            context_length=200000,
            output_type="string",
            streaming_on=False,
            max_tokens=15000,
        )

        # Initialize GroupChat
        agents = [agent1, agent2]
        chat = GroupChat(
            name="Expense Advisory",
            description="Accounting group focused on discussing potential expenses",
            agents=agents,
            max_loops=1,
            output_type="all",
        )

        # Run the chat
        history = chat.run(
            "What potential expenses should we consider for the upcoming quarter? Please collaborate to outline a comprehensive list."
        )
    ```

## Configuration Options

!!! info "Key Parameters"
    | Parameter | Description | Default |
    |-----------|-------------|---------|
    | `max_loops` | Maximum number of conversation loops | 1 |
    | `autosave` | Enable automatic saving of chat history | False |
    | `dashboard` | Enable dashboard visualization | False |
    | `verbose` | Enable detailed logging | True |
    | `dynamic_temperature_enabled` | Enable dynamic temperature adjustment | True |
    | `retry_attempts` | Number of retry attempts for failed operations | 1 |
    | `context_length` | Maximum context length for the model | 200000 |
    | `max_tokens` | Maximum tokens for model output | 15000 |

## Next Steps

!!! tip "What to Try Next"
    1. Experiment with different agent roles and descriptions
    2. Adjust the `max_loops` parameter to allow for longer conversations
    3. Enable the dashboard to visualize agent interactions
    4. Try different model configurations and parameters

## Troubleshooting

!!! warning "Common Issues"
    - Ensure your API key is correctly set in the `.env` file
    - Check that all required dependencies are installed
    - Verify that your model provider's API is accessible
    - Monitor the `verbose` output for detailed error messages

## Additional Resources

- [Swarms Documentation](https://docs.swarms.world)
- [API Reference](https://docs.swarms.world/api)
- [Examples Gallery](https://docs.swarms.world/examples)