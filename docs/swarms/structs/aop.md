# AOP (Agent Orchestration Protocol)

The Agent Orchestration Protocol (AOP) is a powerful framework for deploying multiple Swarms agents as tools in an MCP (Model Context Protocol) server. This enables you to create a distributed system where agents can be accessed as individual tools, making them available for use by other systems, applications, or clients.

AOP provides two main classes:

- **AOP**: Deploy agents as tools in a single MCP server
- **AOPCluster**: Connect to and manage multiple MCP servers

## Core Classes

### AgentToolConfig

Configuration dataclass for converting an agent to an MCP tool.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `tool_name` | `str` | Required | The name of the tool in the MCP server |
| `tool_description` | `str` | Required | Description of what the tool does |
| `input_schema` | `Dict[str, Any]` | Required | JSON schema for the tool's input parameters |
| `output_schema` | `Dict[str, Any]` | Required | JSON schema for the tool's output |
| `timeout` | `int` | `30` | Maximum time to wait for agent execution (seconds) |
| `max_retries` | `int` | `3` | Number of retries if agent execution fails |
| `verbose` | `bool` | `False` | Enable verbose logging for this tool |
| `traceback_enabled` | `bool` | `True` | Enable traceback logging for errors |

### AOP Class

Main class for deploying agents as tools in an MCP server.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `server_name` | `str` | `"AOP Cluster"` | Name for the MCP server |
| `description` | `str` | `"A cluster that enables you to deploy multiple agents as tools in an MCP server."` | Server description |
| `agents` | `List[Agent]` | `None` | Optional list of agents to add initially |
| `port` | `int` | `8000` | Port for the MCP server |
| `transport` | `str` | `"streamable-http"` | Transport type for the MCP server |
| `verbose` | `bool` | `False` | Enable verbose logging |
| `traceback_enabled` | `bool` | `True` | Enable traceback logging for errors |
| `host` | `str` | `"localhost"` | Host to bind the server to |
| `log_level` | `str` | `"INFO"` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

#### Methods

##### add_agent()

Add an agent to the MCP server as a tool.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `AgentType` | Required | The swarms Agent instance to deploy |
| `tool_name` | `str` | `None` | Name for the tool (defaults to agent.agent_name) |
| `tool_description` | `str` | `None` | Description of the tool (defaults to agent.agent_description) |
| `input_schema` | `Dict[str, Any]` | `None` | JSON schema for input parameters |
| `output_schema` | `Dict[str, Any]` | `None` | JSON schema for output |
| `timeout` | `int` | `30` | Maximum execution time in seconds |
| `max_retries` | `int` | `3` | Number of retries on failure |
| `verbose` | `bool` | `None` | Enable verbose logging for this tool |
| `traceback_enabled` | `bool` | `None` | Enable traceback logging for this tool |

**Returns:** `str` - The tool name that was registered

##### add_agents_batch()

Add multiple agents to the MCP server as tools in batch.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `List[Agent]` | Required | List of swarms Agent instances |
| `tool_names` | `List[str]` | `None` | Optional list of tool names |
| `tool_descriptions` | `List[str]` | `None` | Optional list of tool descriptions |
| `input_schemas` | `List[Dict[str, Any]]` | `None` | Optional list of input schemas |
| `output_schemas` | `List[Dict[str, Any]]` | `None` | Optional list of output schemas |
| `timeouts` | `List[int]` | `None` | Optional list of timeout values |
| `max_retries_list` | `List[int]` | `None` | Optional list of max retry values |
| `verbose_list` | `List[bool]` | `None` | Optional list of verbose settings |
| `traceback_enabled_list` | `List[bool]` | `None` | Optional list of traceback settings |

**Returns:** `List[str]` - List of tool names that were registered

##### remove_agent()

Remove an agent from the MCP server.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tool_name` | `str` | Name of the tool to remove |

**Returns:** `bool` - True if agent was removed, False if not found

##### list_agents()

Get a list of all registered agent tool names.

**Returns:** `List[str]` - List of tool names

##### get_agent_info()

Get information about a specific agent tool.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tool_name` | `str` | Name of the tool |

**Returns:** `Dict[str, Any]` - Agent information, or None if not found

##### start_server()

Start the MCP server.

##### run()

Run the MCP server (alias for start_server).

##### get_server_info()

Get information about the MCP server and registered tools.

**Returns:** `Dict[str, Any]` - Server information

### AOPCluster Class

Class for connecting to and managing multiple MCP servers.

#### AOPCluster Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `urls` | `List[str]` | Required | List of MCP server URLs to connect to |
| `transport` | `str` | `"streamable-http"` | Transport type for connections |

#### AOPCluster Methods

##### get_tools()

Get tools from all connected MCP servers.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_type` | `Literal["json", "dict", "str"]` | `"dict"` | Format of the output |

**Returns:** `List[Dict[str, Any]]` - List of available tools

##### find_tool_by_server_name()

Find a tool by its server name (function name).

| Parameter | Type | Description |
|-----------|------|-------------|
| `server_name` | `str` | The name of the tool/function to find |

**Returns:** `Dict[str, Any]` - Tool information, or None if not found

## Tool Parameters

All agent tools accept the following parameters:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task` | `str` | Yes | The main task or prompt to execute |
| `img` | `str` | No | Single image to be processed by the agent |
| `imgs` | `List[str]` | No | Multiple images to be processed by the agent |
| `correct_answer` | `str` | No | Correct answer for validation or comparison |

## Output Format

All agent tools return a standardized response format:

```json
{
  "result": "string",     // The agent's response to the task
  "success": "boolean",   // Whether the task was executed successfully
  "error": "string"       // Error message if execution failed (null if successful)
}
```

## Complete Examples

### Basic Server Setup

```python
from swarms import Agent
from swarms.structs.aop import AOP

# Create specialized agents
research_agent = Agent(
    agent_name="Research-Agent",
    agent_description="Expert in research, data collection, and information gathering",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt="""You are a research specialist. Your role is to:
    1. Gather comprehensive information on any given topic
    2. Analyze data from multiple sources
    3. Provide well-structured research findings
    4. Cite sources and maintain accuracy
    5. Present findings in a clear, organized manner
    
    Always provide detailed, factual information with proper context.""",
)

analysis_agent = Agent(
    agent_name="Analysis-Agent",
    agent_description="Expert in data analysis, pattern recognition, and generating insights",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt="""You are an analysis specialist. Your role is to:
    1. Analyze data and identify patterns
    2. Generate actionable insights
    3. Create visualizations and summaries
    4. Provide statistical analysis
    5. Make data-driven recommendations
    
    Focus on extracting meaningful insights from information.""",
)

# Create AOP instance
deployer = AOP(
    server_name="MyAgentServer",
    port=8000,
    verbose=True,
    log_level="INFO"
)

# Add agents individually
deployer.add_agent(research_agent)
deployer.add_agent(analysis_agent)

# Start the server
deployer.run()
```

### Batch Agent Addition

```python
from swarms import Agent
from swarms.structs.aop import AOP

# Create multiple agents
agents = [
    Agent(
        agent_name="Research-Agent",
        agent_description="Expert in research and data collection",
        model_name="anthropic/claude-sonnet-4-5",
        max_loops=1,
    ),
    Agent(
        agent_name="Writing-Agent", 
        agent_description="Expert in content creation and editing",
        model_name="anthropic/claude-sonnet-4-5",
        max_loops=1,
    ),
    Agent(
        agent_name="Code-Agent",
        agent_description="Expert in programming and code review", 
        model_name="anthropic/claude-sonnet-4-5",
        max_loops=1,
    ),
]

# Create AOP instance
deployer = AOP("MyAgentServer", verbose=True)

# Add all agents at once
tool_names = deployer.add_agents_batch(agents)

print(f"Added {len(tool_names)} agents: {tool_names}")

# Start the server
deployer.run()
```

### Advanced Configuration

```python
from swarms import Agent
from swarms.structs.aop import AOP

# Create agent with custom configuration
research_agent = Agent(
    agent_name="Research-Agent",
    agent_description="Expert in research and data collection",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
)

# Create AOP with custom settings
deployer = AOP(
    server_name="AdvancedAgentServer",
    port=8001,
    host="0.0.0.0",  # Allow external connections
    verbose=True,
    traceback_enabled=True,
    log_level="DEBUG"
)

# Add agent with custom tool configuration
deployer.add_agent(
    agent=research_agent,
    tool_name="custom_research_tool",
    tool_description="Custom research tool with extended capabilities",
    timeout=60,  # 60 second timeout
    max_retries=5,  # 5 retries
    verbose=True,
    traceback_enabled=True
)

# Add custom input/output schemas
custom_input_schema = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "The research task to execute"
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific sources to research"
        },
        "depth": {
            "type": "string",
            "enum": ["shallow", "medium", "deep"],
            "description": "Research depth level"
        }
    },
    "required": ["task"]
}

custom_output_schema = {
    "type": "object", 
    "properties": {
        "result": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "success": {"type": "boolean"},
        "error": {"type": "string"}
    },
    "required": ["result", "success"]
}

# Add another agent with custom schemas
analysis_agent = Agent(
    agent_name="Analysis-Agent",
    agent_description="Expert in data analysis",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
)

deployer.add_agent(
    agent=analysis_agent,
    tool_name="custom_analysis_tool",
    tool_description="Custom analysis tool",
    input_schema=custom_input_schema,
    output_schema=custom_output_schema,
    timeout=45,
    max_retries=3
)

# List all registered agents
print("Registered agents:", deployer.list_agents())

# Get server information
server_info = deployer.get_server_info()
print("Server info:", server_info)

# Start the server
deployer.run()
```

### AOPCluster Usage

```python
import json
from swarms.structs.aop import AOPCluster

# Connect to multiple MCP servers
cluster = AOPCluster(
    urls=[
        "http://localhost:8000/mcp",
        "http://localhost:8001/mcp", 
        "http://localhost:8002/mcp"
    ],
    transport="streamable-http"
)

# Get all available tools from all servers
all_tools = cluster.get_tools(output_type="dict")
print(f"Found {len(all_tools)} tools across all servers")

# Pretty print all tools
print(json.dumps(all_tools, indent=2))

# Find a specific tool by name
research_tool = cluster.find_tool_by_server_name("Research-Agent")
if research_tool:
    print("Found Research-Agent tool:")
    print(json.dumps(research_tool, indent=2))
else:
    print("Research-Agent tool not found")
```

### Tool Execution Examples

Once your AOP server is running, you can call the tools using MCP clients. Here are examples of how the tools would be called:

```python
# Example tool calls (these would be made by MCP clients)

# Basic task execution
result = research_tool(task="Research the latest AI trends in 2024")

# Task with single image
result = analysis_tool(
    task="Analyze this chart and provide insights",
    img="path/to/chart.png"
)

# Task with multiple images
result = writing_tool(
    task="Write a comprehensive report based on these images",
    imgs=["image1.jpg", "image2.jpg", "image3.jpg"]
)

# Task with validation
result = code_tool(
    task="Debug this Python function",
    correct_answer="Expected output: Hello World"
)

# The response format for all calls:
# {
#   "result": "The agent's response...",
#   "success": true,
#   "error": null
# }
```

## Error Handling

AOP provides comprehensive error handling:

- **Timeout Protection**: Each agent has configurable timeout limits
- **Retry Logic**: Automatic retries on failure with configurable retry counts
- **Detailed Logging**: Verbose logging with traceback information
- **Graceful Degradation**: Failed agents don't crash the entire server

## Best Practices

| Best Practice                | Description                                                        |
|------------------------------|--------------------------------------------------------------------|
| **Use Descriptive Names**    | Choose clear, descriptive tool names                               |
| **Set Appropriate Timeouts** | Configure timeouts based on expected task complexity               |
| **Enable Logging**           | Use verbose logging for debugging and monitoring                   |
| **Handle Errors**            | Always check the `success` field in tool responses                 |
| **Resource Management**      | Monitor server resources when running multiple agents              |
| **Security**                 | Use appropriate host/port settings for your deployment environment |

## Integration with Other Systems

AOP servers can be integrated with:

| Integration Target      | Description                                      |
|------------------------|--------------------------------------------------|
| **MCP Clients**        | Any MCP-compatible client                        |
| **Web Applications**   | Via HTTP transport                               |
| **Other Swarms**       | As part of larger multi-agent systems            |
| **External APIs**      | Through MCP protocol                             |

This makes AOP a powerful tool for creating distributed, scalable agent systems that can be easily integrated into existing workflows and applications.
