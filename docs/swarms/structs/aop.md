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
| `agents` | `any` | `None` | Optional list of agents to add initially |
| `port` | `int` | `8000` | Port for the MCP server |
| `transport` | `str` | `"streamable-http"` | Transport type for the MCP server |
| `verbose` | `bool` | `False` | Enable verbose logging |
| `traceback_enabled` | `bool` | `True` | Enable traceback logging for errors |
| `host` | `str` | `"localhost"` | Host to bind the server to |
| `log_level` | `str` | `"INFO"` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `*args` | `Any` | - | Additional positional arguments passed to FastMCP |
| `**kwargs` | `Any` | - | Additional keyword arguments passed to FastMCP |

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

##### _register_tool()

Register a single agent as an MCP tool (internal method).

| Parameter | Type | Description |
|-----------|------|-------------|
| `tool_name` | `str` | Name of the tool to register |
| `agent` | `AgentType` | The agent instance to register |

##### _execute_agent_with_timeout()

Execute an agent with a timeout and all run method parameters (internal method).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `AgentType` | Required | The agent to execute |
| `task` | `str` | Required | The task to execute |
| `timeout` | `int` | Required | Maximum execution time in seconds |
| `img` | `str` | `None` | Optional image to be processed by the agent |
| `imgs` | `List[str]` | `None` | Optional list of images to be processed by the agent |
| `correct_answer` | `str` | `None` | Optional correct answer for validation or comparison |

**Returns:** `str` - The agent's response

**Raises:** `TimeoutError` if execution exceeds timeout, `Exception` if agent execution fails

##### _get_agent_discovery_info()

Get discovery information for a specific agent (internal method).

| Parameter | Type | Description |
|-----------|------|-------------|
| `tool_name` | `str` | Name of the agent tool |

**Returns:** `Optional[Dict[str, Any]]` - Agent discovery information, or None if not found

## Discovery Tools

AOP automatically registers several discovery tools that allow agents to learn about each other and enable dynamic agent discovery within the cluster.

### discover_agents

Discover information about agents in the cluster including their name, description, system prompt (truncated to 200 chars), and tags.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_name` | `str` | `None` | Optional specific agent name to get info for. If None, returns info for all agents. |

**Returns:** `Dict[str, Any]` - Agent information for discovery

**Response Format:**

```json
{
  "success": true,
  "agents": [
    {
      "tool_name": "agent_name",
      "agent_name": "Agent Name",
      "description": "Agent description",
      "short_system_prompt": "Truncated system prompt...",
      "tags": ["tag1", "tag2"],
      "capabilities": ["capability1", "capability2"],
      "role": "worker",
      "model_name": "model_name",
      "max_loops": 1,
      "temperature": 0.5,
      "max_tokens": 4096
    }
  ]
}
```

### get_agent_details

Get detailed information about a single agent by name including configuration, capabilities, and metadata.

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_name` | `str` | Name of the agent to get information for. |

**Returns:** `Dict[str, Any]` - Detailed agent information

**Response Format:**

```json
{
  "success": true,
  "agent_info": {
    "tool_name": "agent_name",
    "agent_name": "Agent Name",
    "agent_description": "Agent description",
    "model_name": "model_name",
    "max_loops": 1,
    "tool_description": "Tool description",
    "timeout": 30,
    "max_retries": 3,
    "verbose": false,
    "traceback_enabled": true
  },
  "discovery_info": {
    "tool_name": "agent_name",
    "agent_name": "Agent Name",
    "description": "Agent description",
    "short_system_prompt": "Truncated system prompt...",
    "tags": ["tag1", "tag2"],
    "capabilities": ["capability1", "capability2"],
    "role": "worker",
    "model_name": "model_name",
    "max_loops": 1,
    "temperature": 0.5,
    "max_tokens": 4096
  }
}
```

### get_agents_info

Get detailed information about multiple agents by providing a list of agent names.

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_names` | `List[str]` | List of agent names to get information for. |

**Returns:** `Dict[str, Any]` - Detailed information for all requested agents

**Response Format:**

```json
{
  "success": true,
  "agents_info": [
    {
      "agent_name": "agent_name",
      "agent_info": { /* detailed agent info */ },
      "discovery_info": { /* discovery info */ }
    }
  ],
  "not_found": ["missing_agent"],
  "total_found": 1,
  "total_requested": 2
}
```

### list_agents

Get a simple list of all available agent names in the cluster.

**Returns:** `Dict[str, Any]` - List of agent names

**Response Format:**

```json
{
  "success": true,
  "agent_names": ["agent1", "agent2", "agent3"],
  "total_count": 3
}
```

### search_agents

Search for agents by name, description, tags, or capabilities using keyword matching.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | Search query string |
| `search_fields` | `List[str]` | `["name", "description", "tags", "capabilities"]` | Optional list of fields to search in. If None, searches all fields. |

**Returns:** `Dict[str, Any]` - Matching agents

**Response Format:**

```json
{
  "success": true,
  "matching_agents": [
    {
      "tool_name": "agent_name",
      "agent_name": "Agent Name",
      "description": "Agent description",
      "short_system_prompt": "Truncated system prompt...",
      "tags": ["tag1", "tag2"],
      "capabilities": ["capability1", "capability2"],
      "role": "worker",
      "model_name": "model_name",
      "max_loops": 1,
      "temperature": 0.5,
      "max_tokens": 4096
    }
  ],
  "total_matches": 1,
  "query": "search_term",
  "search_fields": ["name", "description", "tags", "capabilities"]
}
```

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

### Advanced Configuration with Tags and Capabilities

```python
from swarms import Agent
from swarms.structs.aop import AOP

# Create agent with custom configuration, tags, and capabilities
research_agent = Agent(
    agent_name="Research-Agent",
    agent_description="Expert in research and data collection",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    # Add tags and capabilities for better discovery
    tags=["research", "data-collection", "analysis"],
    capabilities=["web-search", "data-gathering", "report-generation"],
    role="researcher"
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

### Discovery Tools Examples

The AOP server automatically provides discovery tools that allow agents to learn about each other. Here are examples of how to use these tools:

```python
# Example discovery tool calls (these would be made by MCP clients)

# Discover all agents in the cluster
all_agents = discover_agents()
print(f"Found {len(all_agents['agents'])} agents in the cluster")

# Discover a specific agent
research_agent_info = discover_agents(agent_name="Research-Agent")
if research_agent_info['success']:
    agent = research_agent_info['agents'][0]
    print(f"Agent: {agent['agent_name']}")
    print(f"Description: {agent['description']}")
    print(f"Tags: {agent['tags']}")
    print(f"Capabilities: {agent['capabilities']}")

# Get detailed information about a specific agent
agent_details = get_agent_details(agent_name="Research-Agent")
if agent_details['success']:
    print("Agent Info:", agent_details['agent_info'])
    print("Discovery Info:", agent_details['discovery_info'])

# Get information about multiple agents
multiple_agents = get_agents_info(agent_names=["Research-Agent", "Analysis-Agent"])
print(f"Found {multiple_agents['total_found']} out of {multiple_agents['total_requested']} agents")
print("Not found:", multiple_agents['not_found'])

# List all available agents
agent_list = list_agents()
print(f"Available agents: {agent_list['agent_names']}")

# Search for agents by keyword
search_results = search_agents(query="research")
print(f"Found {search_results['total_matches']} agents matching 'research'")

# Search in specific fields only
tag_search = search_agents(
    query="data", 
    search_fields=["tags", "capabilities"]
)
print(f"Found {tag_search['total_matches']} agents with 'data' in tags or capabilities")
```

### Dynamic Agent Discovery Example

Here's a practical example of how agents can use discovery tools to find and collaborate with other agents:

```python
from swarms import Agent
from swarms.structs.aop import AOP

# Create a coordinator agent that can discover and use other agents
coordinator = Agent(
    agent_name="Coordinator-Agent",
    agent_description="Coordinates tasks between different specialized agents",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    tags=["coordination", "orchestration", "management"],
    capabilities=["agent-discovery", "task-distribution", "workflow-management"],
    role="coordinator"
)

# Create specialized agents
research_agent = Agent(
    agent_name="Research-Agent",
    agent_description="Expert in research and data collection",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    tags=["research", "data-collection", "analysis"],
    capabilities=["web-search", "data-gathering", "report-generation"],
    role="researcher"
)

analysis_agent = Agent(
    agent_name="Analysis-Agent",
    agent_description="Expert in data analysis and insights",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    tags=["analysis", "data-processing", "insights"],
    capabilities=["statistical-analysis", "pattern-recognition", "visualization"],
    role="analyst"
)

# Create AOP server
deployer = AOP(
    server_name="DynamicAgentCluster",
    port=8000,
    verbose=True
)

# Add all agents
deployer.add_agent(coordinator)
deployer.add_agent(research_agent)
deployer.add_agent(analysis_agent)

# The coordinator can now discover other agents and use them
# This would be done through MCP tool calls in practice
def coordinate_research_task(task_description):
    """
    Example of how the coordinator might use discovery tools
    """
    # 1. Discover available research agents
    research_agents = discover_agents()
    research_agents = [a for a in research_agents['agents'] if 'research' in a['tags']]
    
    # 2. Get detailed info about the best research agent
    if research_agents:
        best_agent = research_agents[0]
        agent_details = get_agent_details(agent_name=best_agent['agent_name'])
        
        # 3. Use the research agent for the task
        research_result = research_agent.run(task=task_description)
        
        # 4. Find analysis agents for processing the research
        analysis_agents = search_agents(query="analysis", search_fields=["tags"])
        if analysis_agents['matching_agents']:
            analysis_agent_name = analysis_agents['matching_agents'][0]['agent_name']
            analysis_result = analysis_agent.run(task=f"Analyze this research: {research_result}")
            
            return {
                "research_result": research_result,
                "analysis_result": analysis_result,
                "agents_used": [best_agent['agent_name'], analysis_agent_name]
            }
    
    return {"error": "No suitable agents found"}

# Start the server
deployer.run()
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
| **Use Tags and Capabilities** | Add meaningful tags and capabilities to agents for better discovery |
| **Define Agent Roles**       | Use the `role` attribute to categorize agents (coordinator, worker, etc.) |
| **Leverage Discovery Tools** | Use built-in discovery tools for dynamic agent collaboration |
| **Design for Scalability**   | Plan for adding/removing agents dynamically using discovery tools |

## Integration with Other Systems

AOP servers can be integrated with:

| Integration Target      | Description                                      |
|------------------------|--------------------------------------------------|
| **MCP Clients**        | Any MCP-compatible client                        |
| **Web Applications**   | Via HTTP transport                               |
| **Other Swarms**       | As part of larger multi-agent systems            |
| **External APIs**      | Through MCP protocol                             |

This makes AOP a powerful tool for creating distributed, scalable agent systems that can be easily integrated into existing workflows and applications.
