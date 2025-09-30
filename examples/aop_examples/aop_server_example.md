# AOP Server Setup Example

This example demonstrates how to set up an AOP (Agent Orchestration Protocol) server with multiple specialized agents.

## Complete Server Setup

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

writing_agent = Agent(
    agent_name="Writing-Agent",
    agent_description="Expert in content creation, editing, and communication",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt="""You are a writing specialist. Your role is to:
    1. Create engaging, well-structured content
    2. Edit and improve existing text
    3. Adapt tone and style for different audiences
    4. Ensure clarity and coherence
    5. Follow best practices in writing
    
    Always produce high-quality, professional content.""",
)

code_agent = Agent(
    agent_name="Code-Agent",
    agent_description="Expert in programming, code review, and software development",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt="""You are a coding specialist. Your role is to:
    1. Write clean, efficient code
    2. Debug and fix issues
    3. Review and optimize code
    4. Explain programming concepts
    5. Follow best practices and standards
    
    Always provide working, well-documented code.""",
)

financial_agent = Agent(
    agent_name="Financial-Agent",
    agent_description="Expert in financial analysis, market research, and investment insights",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt="""You are a financial specialist. Your role is to:
    1. Analyze financial data and markets
    2. Provide investment insights
    3. Assess risk and opportunities
    4. Create financial reports
    5. Explain complex financial concepts
    
    Always provide accurate, well-reasoned financial analysis.""",
)

# Create AOP instance
deployer = AOP(
    server_name="MyAgentServer",
    port=8000,
    verbose=True,
    log_level="INFO"
)

# Add all agents at once
agents = [
    research_agent,
    analysis_agent,
    writing_agent,
    code_agent,
    financial_agent,
]

tool_names = deployer.add_agents_batch(agents)
print(f"Added {len(tool_names)} agents: {tool_names}")

# Display server information
server_info = deployer.get_server_info()
print(f"Server: {server_info['server_name']}")
print(f"Total tools: {server_info['total_tools']}")
print(f"Available tools: {server_info['tools']}")

# Start the server
print("Starting AOP server...")
deployer.run()
```

## Running the Server

1. Save the code above to a file (e.g., `aop_server.py`)
2. Install required dependencies:
   ```bash
   pip install swarms
   ```
3. Run the server:
   ```bash
   python aop_server.py
   ```

The server will start on `http://localhost:8000` and make all agents available as MCP tools.

## Tool Usage Examples

Once the server is running, you can call the tools using MCP clients:

### Research Agent
```python
# Call the research agent
result = research_tool(task="Research the latest AI trends in 2024")
print(result)
```

### Analysis Agent with Image
```python
# Call the analysis agent with an image
result = analysis_tool(
    task="Analyze this chart and provide insights",
    img="path/to/chart.png"
)
print(result)
```

### Writing Agent with Multiple Images
```python
# Call the writing agent with multiple images
result = writing_tool(
    task="Write a comprehensive report based on these images",
    imgs=["image1.jpg", "image2.jpg", "image3.jpg"]
)
print(result)
```

### Code Agent with Validation
```python
# Call the code agent with expected output
result = code_tool(
    task="Debug this Python function",
    correct_answer="Expected output: Hello World"
)
print(result)
```

### Financial Agent
```python
# Call the financial agent
result = financial_tool(task="Analyze the current market trends for tech stocks")
print(result)
```

## Response Format

All tools return a standardized response:

```json
{
  "result": "The agent's response to the task",
  "success": true,
  "error": null
}
```

## Advanced Configuration

### Custom Timeouts and Retries

```python
# Add agent with custom configuration
deployer.add_agent(
    agent=research_agent,
    tool_name="custom_research_tool",
    tool_description="Research tool with extended timeout",
    timeout=120,  # 2 minutes
    max_retries=5,
    verbose=True
)
```

### Custom Input/Output Schemas

```python
# Define custom schemas
custom_input_schema = {
    "type": "object",
    "properties": {
        "task": {"type": "string", "description": "The research task"},
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

# Add agent with custom schemas
deployer.add_agent(
    agent=research_agent,
    tool_name="advanced_research_tool",
    input_schema=custom_input_schema,
    timeout=60
)
```

## Monitoring and Debugging

### Enable Verbose Logging

```python
deployer = AOP(
    server_name="DebugServer",
    verbose=True,
    traceback_enabled=True,
    log_level="DEBUG"
)
```

### Check Server Status

```python
# List all registered agents
agents = deployer.list_agents()
print(f"Registered agents: {agents}")

# Get detailed agent information
for agent_name in agents:
    info = deployer.get_agent_info(agent_name)
    print(f"Agent {agent_name}: {info}")

# Get server information
server_info = deployer.get_server_info()
print(f"Server info: {server_info}")
```

## Production Deployment

### External Access

```python
deployer = AOP(
    server_name="ProductionServer",
    host="0.0.0.0",  # Allow external connections
    port=8000,
    verbose=False,  # Disable verbose logging in production
    log_level="WARNING"
)
```

### Multiple Servers

```python
# Server 1: Research and Analysis
research_deployer = AOP("ResearchServer", port=8000)
research_deployer.add_agent(research_agent)
research_deployer.add_agent(analysis_agent)

# Server 2: Writing and Code
content_deployer = AOP("ContentServer", port=8001)
content_deployer.add_agent(writing_agent)
content_deployer.add_agent(code_agent)

# Server 3: Financial
finance_deployer = AOP("FinanceServer", port=8002)
finance_deployer.add_agent(financial_agent)

# Start all servers
import threading

threading.Thread(target=research_deployer.run).start()
threading.Thread(target=content_deployer.run).start()
threading.Thread(target=finance_deployer.run).start()
```

This example demonstrates a complete AOP server setup with multiple specialized agents, proper configuration, and production-ready deployment options.
