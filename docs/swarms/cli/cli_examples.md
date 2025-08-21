# Swarms CLI Examples

This document provides comprehensive examples of how to use the Swarms CLI for various scenarios. Each example includes the complete command, expected output, and explanation.

## Table of Contents

- [Basic Usage Examples](#basic-usage-examples)
- [Agent Management Examples](#agent-management-examples)
- [Multi-Agent Workflow Examples](#multi-agent-workflow-examples)
- [Configuration Examples](#configuration-examples)
- [Advanced Usage Examples](#advanced-usage-examples)
- [Troubleshooting Examples](#troubleshooting-examples)

## Basic Usage Examples

### 1. Getting Started

#### Check CLI Installation

```bash
swarms help
```

**Expected Output:**
```
  _________                                     
 /   _____/_  _  _______ _______  _____   ______
 \_____  \\ \/ \/ /\__  \\_  __ \/     \ /  ___/
 /        \\     /  / __ \|  | \/  Y Y  \\___ \ 
/_______  / \/\_/  (____  /__|  |__|_|  /____  >
        \/              \/            \/     \/                                

Available Commands
┌─────────────────┬─────────────────────────────────────────────────────────────┐
│ Command         │ Description                                                 │
├─────────────────┼─────────────────────────────────────────────────────────────┤
│ onboarding      │ Start the interactive onboarding process                   │
│ help            │ Display this help message                                  │
│ get-api-key     │ Retrieve your API key from the platform                   │
│ check-login     │ Verify login status and initialize cache                   │
│ run-agents      │ Execute agents from your YAML configuration                │
│ load-markdown   │ Load agents from markdown files with YAML frontmatter     │
│ agent           │ Create and run a custom agent with specified parameters   │
│ auto-upgrade    │ Update Swarms to the latest version                       │
│ book-call       │ Schedule a strategy session with our team                 │
│ autoswarm       │ Generate and execute an autonomous swarm                  │
└─────────────────┴─────────────────────────────────────────────────────────────┘
```

#### Start Onboarding Process
```bash
swarms onboarding
```

This will start an interactive setup process to configure your environment.

#### Get API Key

```bash
swarms get-api-key
```

**Expected Output:**
```
✓ API key page opened in your browser
```

#### Check Login Status

```bash
swarms check-login
```

**Expected Output:**
```
✓ Authentication verified
```

## Agent Management Examples

### 2. Creating Custom Agents

#### Basic Research Agent

```bash
swarms agent \
  --name "Research Assistant" \
  --description "AI research specialist for academic papers" \
  --system-prompt "You are an expert research assistant specializing in academic research. You help users find, analyze, and synthesize information from various sources. Always provide well-structured, evidence-based responses." \
  --task "Research the latest developments in quantum computing and provide a summary of key breakthroughs in the last 2 years" \
  --model-name "gpt-4" \
  --temperature 0.1 \
  --max-loops 3
```

**Expected Output:**
```
Creating custom agent: Research Assistant
[✓] Agent 'Research Assistant' completed the task successfully!

┌─────────────────────────────────────────────────────────────────────────────┐
│ Agent Execution Results                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Agent Name: Research Assistant                                            │
│ Model: gpt-4                                                              │
│ Task: Research the latest developments in quantum computing...            │
│ Result:                                                                   │
│ Recent breakthroughs in quantum computing include:                        │
│ 1. Google's 53-qubit Sycamore processor achieving quantum supremacy      │
│ 2. IBM's 433-qubit Osprey processor...                                   │
│ ...                                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Code Review Agent

```bash
swarms agent \
  --name "Code Reviewer" \
  --description "Expert code review assistant with security focus" \
  --system-prompt "You are a senior software engineer specializing in code review, security analysis, and best practices. Review code for bugs, security vulnerabilities, performance issues, and adherence to coding standards." \
  --task "Review this Python code for security vulnerabilities and suggest improvements: def process_user_input(data): return eval(data)" \
  --model-name "gpt-4" \
  --temperature 0.05 \
  --max-loops 2 \
  --verbose
```

#### Financial Analysis Agent

```bash
swarms agent \
  --name "Financial Analyst" \
  --description "Expert financial analyst for market research and investment advice" \
  --system-prompt "You are a certified financial analyst with expertise in market analysis, investment strategies, and risk assessment. Provide data-driven insights and recommendations based on current market conditions." \
  --task "Analyze the current state of the technology sector and provide investment recommendations for the next quarter" \
  --model-name "gpt-4" \
  --temperature 0.2 \
  --max-loops 2 \
  --output-type "json"
```

### 3. Advanced Agent Configuration

#### Agent with Dynamic Features

```bash
swarms agent \
  --name "Adaptive Writer" \
  --description "Content writer with dynamic temperature and context adjustment" \
  --system-prompt "You are a professional content writer who adapts writing style based on audience and context. You can write in various tones from formal to casual, and adjust complexity based on the target audience." \
  --task "Write a blog post about artificial intelligence for a general audience, explaining complex concepts in simple terms" \
  --model-name "gpt-4" \
  --dynamic-temperature-enabled \
  --dynamic-context-window \
  --context-length 8000 \
  --retry-attempts 3 \
  --return-step-meta \
  --autosave \
  --saved-state-path "./agent_states/"
```

#### Agent with MCP Integration

```bash
swarms agent \
  --name "MCP Agent" \
  --description "Agent with Model Context Protocol integration" \
  --system-prompt "You are a agent with access to external tools and data sources through MCP. Use these capabilities to provide comprehensive and up-to-date information." \
  --task "Search for recent news about climate change and summarize the key findings" \
  --model-name "gpt-4" \
  --mcp-url "https://api.example.com/mcp" \
  --temperature 0.1 \
  --max-loops 5
```

## Multi-Agent Workflow Examples

### 4. Running Agents from YAML Configuration

#### Create `research_team.yaml`

```yaml
agents:
  - name: "Data Collector"
    description: "Specialist in gathering and organizing data from various sources"
    model_name: "gpt-4"
    system_prompt: "You are a data collection specialist. Your role is to gather relevant information from multiple sources and organize it in a structured format."
    temperature: 0.1
    max_loops: 3
    
  - name: "Data Analyzer"
    description: "Expert in analyzing and interpreting complex datasets"
    model_name: "gpt-4"
    system_prompt: "You are a data analyst. Take the collected data and perform comprehensive analysis to identify patterns, trends, and insights."
    temperature: 0.2
    max_loops: 4
    
  - name: "Report Writer"
    description: "Professional writer who creates clear, compelling reports"
    model_name: "gpt-4"
    system_prompt: "You are a report writer. Take the analyzed data and create a comprehensive, well-structured report that communicates findings clearly."
    temperature: 0.3
    max_loops: 3
```

#### Execute the Team

```bash
swarms run-agents --yaml-file research_team.yaml
```

**Expected Output:**
```
Loading agents from research_team.yaml...
[✓] Agents completed their tasks successfully!

Results:
Data Collector: [Collected data from 15 sources...]
Data Analyzer: [Identified 3 key trends and 5 significant patterns...]
Report Writer: [Generated comprehensive 25-page report...]
```

### 5. Loading Agents from Markdown

#### Create `agents/researcher.md`

```markdown
---
name: Market Researcher
description: Expert in market research and competitive analysis
model_name: gpt-4
temperature: 0.1
max_loops: 3
---

You are an expert market researcher with 15+ years of experience in competitive analysis, market sizing, and trend identification. You specialize in technology markets and have deep knowledge of consumer behavior, pricing strategies, and market dynamics.

Your approach includes:
- Systematic data collection from multiple sources
- Quantitative and qualitative analysis
- Competitive landscape mapping
- Market opportunity identification
- Risk assessment and mitigation strategies
```

#### Create `agents/analyst.md`

```markdown
---
name: Business Analyst
description: Strategic business analyst focusing on growth opportunities
model_name: gpt-4
temperature: 0.2
max_loops: 4
---

You are a senior business analyst specializing in strategic planning and growth strategy. You excel at identifying market opportunities, analyzing competitive advantages, and developing actionable business recommendations.

Your expertise covers:
- Market opportunity analysis
- Competitive positioning
- Business model innovation
- Risk assessment
- Strategic planning frameworks
```

#### Load and Use Agents

```bash
swarms load-markdown --markdown-path ./agents/ --concurrent
```

**Expected Output:**
```
Loading agents from markdown: ./agents/
✓ Successfully loaded 2 agents!

┌─────────────────────────────────────────────────────────────────────────────┐
│ Loaded Agents                                                              │
├─────────────────┬──────────────┬───────────────────────────────────────────┤
│ Name            │ Model        │ Description                               │
├─────────────────┼──────────────┼───────────────────────────────────────────┤
│ Market Researcher│ gpt-4        │ Expert in market research and competitive │
│                 │              │ analysis                                  │
├─────────────────┼──────────────┼───────────────────────────────────────────┤
│ Business Analyst│ gpt-4        │ Strategic business analyst focusing on    │
│                 │              │ growth opportunities                      │
└─────────────────┴──────────────┴───────────────────────────────────────────┘

Ready to use 2 agents!
You can now use these agents in your code or run them interactively.
```

## Configuration Examples

### 6. YAML Configuration Templates

#### Simple Agent Configuration

```yaml
# simple_agent.yaml
agents:
  - name: "Simple Assistant"
    description: "Basic AI assistant for general tasks"
    model_name: "gpt-3.5-turbo"
    system_prompt: "You are a helpful AI assistant."
    temperature: 0.7
    max_loops: 1
```

#### Advanced Multi-Agent Configuration

```yaml
# advanced_team.yaml
agents:
  - name: "Project Manager"
    description: "Coordinates team activities and ensures project success"
    model_name: "gpt-4"
    system_prompt: "You are a senior project manager with expertise in agile methodologies, risk management, and team coordination."
    temperature: 0.1
    max_loops: 5
    auto_generate_prompt: true
    dynamic_temperature_enabled: true
    
  - name: "Technical Lead"
    description: "Provides technical guidance and architecture decisions"
    model_name: "gpt-4"
    system_prompt: "You are a technical lead with deep expertise in software architecture, system design, and technical decision-making."
    temperature: 0.2
    max_loops: 4
    context_length: 12000
    retry_attempts: 3
    
  - name: "Quality Assurance"
    description: "Ensures quality standards and testing coverage"
    model_name: "gpt-4"
    system_prompt: "You are a QA specialist focused on quality assurance, testing strategies, and process improvement."
    temperature: 0.1
    max_loops: 3
    return_step_meta: true
    dashboard: true
```

### 7. Markdown Configuration Templates

#### Research Agent Template

```markdown
---
name: Research Specialist
description: Academic research and literature review expert
model_name: gpt-4
temperature: 0.1
max_loops: 5
context_length: 16000
auto_generate_prompt: true
---

You are a research specialist with expertise in academic research methodologies, literature review, and scholarly writing. You excel at:

- Systematic literature reviews
- Research methodology design
- Data analysis and interpretation
- Academic writing and citation
- Research gap identification

Always provide evidence-based responses and cite relevant sources when possible.
```

#### Creative Writing Agent Template

```markdown
---
name: Creative Writer
description: Professional creative writer and storyteller
model_name: gpt-4
temperature: 0.8
max_loops: 3
dynamic_temperature_enabled: true
output_type: markdown
---

You are a creative writer with a passion for storytelling, character development, and engaging narratives. You specialize in:

- Fiction writing across multiple genres
- Character development and dialogue
- Plot structure and pacing
- Creative problem-solving
- Engaging opening hooks and satisfying conclusions

Your writing style is adaptable, engaging, and always focused on creating memorable experiences for readers.
```

## Advanced Usage Examples

### 8. Autonomous Swarm Generation

#### Simple Task
```bash
swarms autoswarm \
  --task "Create a weekly meal plan for a family of 4 with dietary restrictions" \
  --model "gpt-4"
```

#### Complex Research Task
```bash
swarms autoswarm \
  --task "Conduct a comprehensive analysis of the impact of artificial intelligence on job markets, including historical trends, current state, and future projections. Include case studies from different industries and recommendations for workforce adaptation." \
  --model "gpt-4"
```

### 9. Integration Examples

#### CI/CD Pipeline Integration
```yaml
# .github/workflows/swarms-test.yml
name: Swarms Agent Testing
on: [push, pull_request]

jobs:
  test-agents:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install swarms
             - name: Run Swarms Agents
         run: |
           swarms run-agents --yaml-file ci_agents.yaml
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

#### Shell Script Integration
```bash
#!/bin/bash
# run_daily_analysis.sh

echo "Starting daily market analysis..."

# Run market research agent
swarms agent \
  --name "Daily Market Analyzer" \
  --description "Daily market analysis and reporting" \
  --system-prompt "You are a market analyst providing daily market insights." \
  --task "Analyze today's market movements and provide key insights" \
  --model-name "gpt-4" \
  --temperature 0.1

# Run risk assessment agent
swarms agent \
  --name "Risk Assessor" \
  --description "Risk assessment and mitigation specialist" \
  --system-prompt "You are a risk management expert." \
  --task "Assess current market risks and suggest mitigation strategies" \
  --model-name "gpt-4" \
  --temperature 0.2

echo "Daily analysis complete!"
```

## Troubleshooting Examples

### 10. Common Error Scenarios

#### Missing API Key
```bash
swarms agent \
  --name "Test Agent" \
  --description "Test" \
  --system-prompt "Test" \
  --task "Test"
```

**Expected Error:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Error                                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Failed to create or run agent: No API key found                           │
└─────────────────────────────────────────────────────────────────────────────┘

Please check:
1. Your API keys are set correctly
2. The model name is valid
3. All required parameters are provided
4. Your system prompt is properly formatted
```

**Resolution:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

#### Invalid YAML Configuration
```bash
swarms run-agents --yaml-file invalid.yaml
```

**Expected Error:**
```
┌─────────────────────────────────────────────────────────────────────────────┘
│ Configuration Error                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Error parsing YAML: Invalid YAML syntax                                   │
└─────────────────────────────────────────────────────────────────────────────┘

Please check your agents.yaml file format.
```

#### File Not Found
```bash
swarms load-markdown --markdown-path ./nonexistent/
```

**Expected Error:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ File Error                                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Markdown file/directory not found: ./nonexistent/                         │
└─────────────────────────────────────────────────────────────────────────────┘

Please make sure the path exists and you're in the correct directory.
```

### 11. Debug Mode Usage

#### Enable Verbose Output
```bash
swarms agent \
  --name "Debug Agent" \
  --description "Agent for debugging" \
  --system-prompt "You are a debugging assistant." \
  --task "Help debug this issue" \
  --model-name "gpt-4" \
  --verbose
```

This will provide detailed output including:
- Step-by-step execution details
- API call information
- Internal state changes
- Performance metrics

## Best Practices

### 12. Performance Optimization

#### Use Concurrent Processing
```bash
# For multiple markdown files
swarms load-markdown \
  --markdown-path ./large_agent_directory/ \
  --concurrent
```

#### Optimize Model Selection
```bash
# For simple tasks
--model-name "gpt-3.5-turbo" --temperature 0.1

# For complex reasoning
--model-name "gpt-4" --temperature 0.1 --max-loops 5
```

#### Context Length Management
```bash
# For long documents
--context-length 16000 --dynamic-context-window

# For concise responses
--context-length 4000 --max-loops 2
```

### 13. Security Considerations

#### Environment Variable Usage
```bash
# Secure API key management
export OPENAI_API_KEY="your-secure-key"
export ANTHROPIC_API_KEY="your-secure-key"

# Use in CLI
swarms agent [options]
```

#### File Permissions
```bash
# Secure configuration files
chmod 600 agents.yaml
chmod 600 .env
```

## Summary

The Swarms CLI provides a powerful and flexible interface for managing AI agents and multi-agent workflows. These examples demonstrate:

| Feature                | Description                                             |
|------------------------|---------------------------------------------------------|
| **Basic Usage**        | Getting started with the CLI                            |
| **Agent Management**   | Creating and configuring custom agents                  |
| **Multi-Agent Workflows** | Coordinating multiple agents                        |
| **Configuration**      | YAML and markdown configuration formats                 |
| **Advanced Features**  | Dynamic configuration and MCP integration               |
| **Troubleshooting**    | Common issues and solutions                             |
| **Best Practices**     | Performance and security considerations                 |

For more information, refer to the [CLI Reference](cli_reference.md) documentation.
