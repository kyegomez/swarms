# Building Agents from a YAML File

The `create_agents_from_yaml` function enables the dynamic creation and execution of agents based on configurations defined in a YAML file. This function is designed to support enterprise use-cases, offering flexibility, reliability, and scalability for various agent-based workflows.

By allowing the user to define multiple agents and tasks in a YAML configuration file, this function streamlines the process of initializing and executing tasks through agents while supporting advanced features such as multi-agent orchestration, logging, error handling, and flexible return values.

---

# Key Features
- **Multi-Agent Creation**: Automatically create multiple agents based on a single YAML configuration file.
- **Task Execution**: Each agent can execute a predefined task if specified in the YAML configuration.
- **Logging with Loguru**: Integrated logging using `loguru` for robust, real-time tracking and error reporting.
- **Dynamic Return Types**: Offers flexible return values (agents, tasks, or both) based on user needs.
- **Error Handling**: Gracefully handles missing configurations, invalid inputs, and runtime errors.
- **Extensibility**: Supports additional positional (`*args`) and keyword arguments (`**kwargs`) to customize agent behavior.

---

# Function Signature

```python
def create_agents_from_yaml(yaml_file: str, return_type: str = "agents", *args, **kwargs)
```

### Parameters:

- **`yaml_file: str`**
  - Description: The path to the YAML file containing agent configurations.
  - Required: Yes
  - Example: `'agents_config.yaml'`

- **`return_type: str`**
  - Description: Determines the type of data the function should return.
  - Options:
    - `"agents"`: Returns a list of the created agents.
    - `"tasks"`: Returns a list of task results (outputs or errors).
    - `"both"`: Returns both the list of agents and the task results as a tuple.
  - Default: `"agents"`
  - Required: No

- **`*args` and `**kwargs`**:
  - Description: Additional arguments to customize agent behavior. These can be passed through to the underlying `Agent` or `OpenAIChat` class constructors.
  - Required: No
  - Example: Can be used to modify model configurations or agent behavior dynamically.

### Returns:
- **Based on `return_type`:**
  - `return_type="agents"`: Returns a list of initialized `Agent` objects.
  - `return_type="tasks"`: Returns a list of task results (success or error).
  - `return_type="both"`: Returns a tuple containing both the list of agents and task results.

---

# YAML Configuration Structure

The function relies on a YAML file for defining agents and tasks. Below is an example YAML configuration:

### Example YAML (agents_config.yaml):
```yaml
agents:
  - agent_name: "Financial-Analysis-Agent"
    # model:
    #   model_name: "gpt-4o-mini"
    #   temperature: 0.1
    #   max_tokens: 2000
    system_prompt: "Your full system prompt here"
    max_loops: 1
    autosave: true
    dashboard: false
    verbose: true
    dynamic_temperature_enabled: true
    saved_state_path: "finance_agent.json"
    user_name: "swarms_corp"
    retry_attempts: 1
    context_length: 200000
    return_step_meta: false
    output_type: "str"
    task: "How can I establish a ROTH IRA to buy stocks and get a tax break?"

  - agent_name: "Stock-Analysis-Agent"
    # model:
    #   model_name: "gpt-4o-mini"
    #   temperature: 0.2
    #   max_tokens: 1500
    system_prompt: "Your full system prompt here"
    max_loops: 2
    autosave: true
    dashboard: false
    verbose: true
    dynamic_temperature_enabled: false
    saved_state_path: "stock_agent.json"
    user_name: "stock_user"
    retry_attempts: 3
    context_length: 150000
    return_step_meta: true
    output_type: "json"
    task: "What is the best strategy for long-term stock investment?"
```

---

# Enterprise Use Cases

### 1. **Automating Financial Analysis**
  - An enterprise can use this function to create agents that analyze financial data in real-time. For example, an agent can be configured to provide financial advice based on the latest market trends, using predefined tasks in YAML to query the agent.

### 2. **Scalable Stock Analysis**
  - Multiple stock analysis agents can be created, each tasked with analyzing specific stocks or investment strategies. This setup can help enterprises handle large-scale financial modeling and stock analysis without manual intervention.

### 3. **Task Scheduling and Execution**
  - In enterprise operations, agents can be pre-configured with tasks such as risk assessment, regulatory compliance checks, or financial forecasting. The function automatically runs these tasks and returns actionable results or alerts.

---

### Full Code Example

```python
import os

from dotenv import load_dotenv
from loguru import logger
from swarm_models import OpenAIChat # any model from swarm_models

from swarms.agents.create_agents_from_yaml import (
    create_agents_from_yaml,
)

# Load environment variables
load_dotenv()

# Path to your YAML file
yaml_file = "agents.yaml"

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)


try:
    # Create agents and run tasks (using 'both' to return agents and task results)
    task_results = create_agents_from_yaml(
        model=model, yaml_file=yaml_file, return_type="tasks"
    )

    logger.info(f"Results from agents: {task_results}")
except Exception as e:
    logger.error(f"An error occurred: {e}")


```

---

# Error Handling

### Common Errors:
1. **Missing API Key**:
   - Error: `API key is missing for agent: <agent_name>`
   - Cause: The API key is either not provided in the YAML or not available as an environment variable.
   - Solution: Ensure the API key is either defined in the YAML configuration or set as an environment variable.

2. **Missing System Prompt**:
   - Error: `System prompt is missing for agent: <agent_name>`
   - Cause: The `system_prompt` field is not defined in the YAML configuration.
   - Solution: Define the system prompt field for each agent.

3. **Invalid `return_type`**:
   - Error: `Invalid return_type: <return_type>`
   - Cause: The `return_type` provided is not one of `"agents"`, `"tasks"`, or `"both"`.
   - Solution: Ensure that `return_type` is set to one of the valid options.

### Logging:
- The function integrates `loguru` logging to track all key actions:
  - File loading, agent creation, task execution, and errors are all logged.
  - Use this logging output to monitor operations and diagnose issues in production environments.

---

# Scalability and Extensibility

### Scalability:
The `create_agents_from_yaml` function is designed for use in high-scale enterprise environments:
- **Multi-Agent Creation**: Create and manage large numbers of agents simultaneously.
- **Parallel Task Execution**: Tasks can be executed in parallel for real-time analysis and decision-making across multiple business units.

### Extensibility:
- **Customizable Behavior**: Through `*args` and `**kwargs`, users can extend the functionality of the agents or models without altering the core YAML configuration.
- **Seamless Integration**: The function can be easily integrated with larger multi-agent systems and workflows, enabling rapid scaling across departments.

---

# Security Considerations

For enterprise deployments, consider the following security best practices:
1. **API Key Management**: Ensure that API keys are stored securely (e.g., using environment variables or secret management tools).
2. **Data Handling**: Be mindful of sensitive information within tasks or agent responses. Implement data sanitization where necessary.
3. **Task Validation**: Validate tasks in the YAML file to ensure they meet your organization's security and operational policies before execution.

---

# Conclusion

The `create_agents_from_yaml` function is a powerful tool for enterprises to dynamically create, manage, and execute tasks using AI-powered agents. With its flexible configuration, logging, and error handling, this function is ideal for scaling agent-based systems and automating complex workflows across industries. 

Integrating this function into your enterprise workflow will enhance efficiency, provide real-time insights, and reduce operational overhead.