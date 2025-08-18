# Stagehand Browser Automation Integration for Swarms

This directory contains examples demonstrating how to integrate [Stagehand](https://github.com/browserbase/stagehand), an AI-powered browser automation framework, with the Swarms multi-agent framework.

## Overview

Stagehand provides natural language browser automation capabilities that can be seamlessly integrated into Swarms agents. This integration enables:

- 🌐 **Natural Language Web Automation**: Use simple commands like "click the submit button" or "extract product prices"
- 🤖 **Multi-Agent Browser Workflows**: Multiple agents can automate different websites simultaneously
- 🔧 **Flexible Integration Options**: Use as a wrapped agent, individual tools, or via MCP server
- 📊 **Complex Automation Scenarios**: E-commerce monitoring, competitive analysis, automated testing, and more

## Examples

### 1. Stagehand Wrapper Agent (`1_stagehand_wrapper_agent.py`)

The simplest integration - wraps Stagehand as a Swarms-compatible agent.

```python
from examples.stagehand.stagehand_wrapper_agent import StagehandAgent

# Create a browser automation agent
browser_agent = StagehandAgent(
    agent_name="WebScraperAgent",
    model_name="gpt-4o-mini",
    env="LOCAL",  # or "BROWSERBASE" for cloud execution
)

# Use natural language to control the browser
result = browser_agent.run(
    "Navigate to news.ycombinator.com and extract the top 5 story titles"
)
```

**Features:**
- Inherits from Swarms `Agent` base class
- Automatic browser lifecycle management
- Natural language task interpretation
- Support for both local (Playwright) and cloud (Browserbase) execution

### 2. Stagehand as Tools (`2_stagehand_tools_agent.py`)

Provides fine-grained control by exposing Stagehand methods as individual tools.

```python
from swarms import Agent
from examples.stagehand.stagehand_tools_agent import (
    NavigateTool, ActTool, ExtractTool, ObserveTool, ScreenshotTool
)

# Create agent with browser tools
browser_agent = Agent(
    agent_name="BrowserAutomationAgent",
    model_name="gpt-4o-mini",
    tools=[
        NavigateTool(),
        ActTool(),
        ExtractTool(),
        ObserveTool(),
        ScreenshotTool(),
    ],
)

# Agent can now use tools strategically
result = browser_agent.run(
    "Go to google.com, search for 'Python tutorials', and extract the first 3 results"
)
```

**Available Tools:**
- `NavigateTool`: Navigate to URLs
- `ActTool`: Perform actions (click, type, scroll)
- `ExtractTool`: Extract data from pages
- `ObserveTool`: Find elements on pages
- `ScreenshotTool`: Capture screenshots
- `CloseBrowserTool`: Clean up browser resources

### 3. Stagehand MCP Server (`3_stagehand_mcp_agent.py`)

Integrates with Stagehand's Model Context Protocol (MCP) server for standardized tool access.

```python
from examples.stagehand.stagehand_mcp_agent import StagehandMCPAgent

# Connect to Stagehand MCP server
mcp_agent = StagehandMCPAgent(
    agent_name="WebResearchAgent",
    mcp_server_url="http://localhost:3000/sse",
)

# Use MCP tools including multi-session management
result = mcp_agent.run("""
    Create 3 browser sessions and:
    1. Session 1: Check Python.org for latest version
    2. Session 2: Check PyPI for trending packages  
    3. Session 3: Check GitHub Python trending repos
    Compile a Python ecosystem status report.
""")
```

**MCP Features:**
- Automatic tool discovery
- Multi-session browser management
- Built-in screenshot resources
- Prompt templates for common tasks

### 4. Multi-Agent Workflows (`4_stagehand_multi_agent_workflow.py`)

Demonstrates complex multi-agent browser automation scenarios.

```python
from examples.stagehand.stagehand_multi_agent_workflow import (
    create_price_comparison_workflow,
    create_competitive_analysis_workflow,
    create_automated_testing_workflow,
    create_news_aggregation_workflow
)

# Price comparison across multiple e-commerce sites
price_workflow = create_price_comparison_workflow()
result = price_workflow.run(
    "Compare prices for iPhone 15 Pro on Amazon and eBay"
)

# Competitive analysis of multiple companies
competitive_workflow = create_competitive_analysis_workflow()
result = competitive_workflow.run(
    "Analyze OpenAI, Anthropic, and DeepMind websites and social media"
)
```

**Workflow Examples:**
- **E-commerce Monitoring**: Track prices across multiple sites
- **Competitive Analysis**: Research competitors' websites and social media
- **Automated Testing**: UI, form validation, and accessibility testing
- **News Aggregation**: Collect and analyze news from multiple sources

## Setup

### Prerequisites

1. **Install Swarms and Stagehand:**
```bash
pip install swarms stagehand
```

2. **Set up environment variables:**
```bash
# For local browser automation (using Playwright)
export OPENAI_API_KEY="your-openai-key"

# For cloud browser automation (using Browserbase)
export BROWSERBASE_API_KEY="your-browserbase-key"
export BROWSERBASE_PROJECT_ID="your-project-id"
```

3. **For MCP Server examples:**
```bash
# Install and run the Stagehand MCP server
cd stagehand-mcp-server
npm install
npm run build
npm start
```

## Use Cases

### E-commerce Automation
- Price monitoring and comparison
- Inventory tracking
- Automated purchasing workflows
- Review aggregation

### Research and Analysis
- Competitive intelligence gathering
- Market research automation
- Social media monitoring
- News and trend analysis

### Quality Assurance
- Automated UI testing
- Cross-browser compatibility testing
- Form validation testing
- Accessibility compliance checking

### Data Collection
- Web scraping at scale
- Real-time data monitoring
- Structured data extraction
- Screenshot documentation

## Best Practices

1. **Resource Management**: Always clean up browser instances when done
```python
browser_agent.cleanup()  # For wrapper agents
```

2. **Error Handling**: Stagehand includes self-healing capabilities, but wrap critical operations in try-except blocks

3. **Parallel Execution**: Use `ConcurrentWorkflow` for simultaneous browser automation across multiple sites

4. **Session Management**: For complex multi-page workflows, use the MCP server's session management capabilities

5. **Rate Limiting**: Be respectful of websites - add delays between requests when necessary

## Testing

Run the test suite to verify the integration:

```bash
pytest tests/stagehand/test_stagehand_integration.py -v
```

## Troubleshooting

### Common Issues

1. **Browser not starting**: Ensure Playwright is properly installed
```bash
playwright install
```

2. **MCP connection failed**: Verify the MCP server is running on the correct port

3. **Timeout errors**: Increase timeout in StagehandConfig or agent initialization

### Debug Mode

Enable verbose logging:
```python
agent = StagehandAgent(
    agent_name="DebugAgent",
    verbose=True,  # Enable detailed logging
)
```

## Contributing

We welcome contributions! Please:
1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Submit PRs with clear descriptions

## License

These examples are provided under the same license as the Swarms framework. Stagehand is licensed separately - see [Stagehand's repository](https://github.com/browserbase/stagehand) for details.