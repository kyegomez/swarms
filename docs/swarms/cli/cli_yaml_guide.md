# CLI YAML Configuration Guide: Run Agents from Config Files

Run multiple agents from YAML configuration files for reproducible, version-controlled agent deployments.

## Basic YAML Configuration

### Step 1: Create YAML Config File

Create a file named `agents.yaml`:

```yaml
agents:
  - name: "Research-Agent"
    description: "AI research specialist"
    model_name: "gpt-4o-mini"
    system_prompt: |
      You are an expert researcher. 
      Provide comprehensive, well-structured research summaries.
      Include key insights and data points.
    temperature: 0.3
    max_loops: 2
    task: "Research current trends in renewable energy"

  - name: "Analysis-Agent"
    description: "Data analysis specialist"
    model_name: "gpt-4o-mini"
    system_prompt: |
      You are a data analyst.
      Provide detailed statistical analysis and insights.
      Use data-driven reasoning.
    temperature: 0.2
    max_loops: 3
    task: "Analyze market opportunities in the EV sector"
```

### Step 2: Run Agents from YAML

```bash
swarms run-agents --yaml-file agents.yaml
```

### Step 3: View Results

Results are displayed in the terminal with formatted output for each agent.

---

## Complete YAML Schema

### Agent Configuration Options

```yaml
agents:
  - name: "Agent-Name"                    # Required: Agent identifier
    description: "Agent description"       # Required: What the agent does
    model_name: "gpt-4o-mini"             # Model to use
    system_prompt: "Your instructions"     # Agent's system prompt
    temperature: 0.5                       # Creativity (0.0-2.0)
    max_loops: 3                          # Maximum execution loops
    task: "Task to execute"               # Task for this agent
    
    # Optional settings
    context_length: 8192                  # Context window size
    streaming_on: true                    # Enable streaming
    verbose: true                         # Verbose output
    autosave: true                        # Auto-save state
    saved_state_path: "./states/agent.json"  # State file path
    output_type: "json"                   # Output format
    retry_attempts: 3                     # Retries on failure
```

---

## Use Case Examples

### Multi-Agent Research Pipeline

```yaml
# research_pipeline.yaml
agents:
  - name: "Data-Collector"
    description: "Collects and organizes research data"
    model_name: "gpt-4o-mini"
    system_prompt: |
      You are a research data collector.
      Gather comprehensive information on the given topic.
      Organize findings into structured categories.
    temperature: 0.3
    max_loops: 2
    task: "Collect data on AI applications in healthcare"

  - name: "Trend-Analyst"
    description: "Analyzes trends and patterns"
    model_name: "gpt-4o-mini"
    system_prompt: |
      You are a trend analyst.
      Identify emerging patterns and trends from data.
      Provide statistical insights and projections.
    temperature: 0.2
    max_loops: 2
    task: "Analyze AI healthcare adoption trends from 2020-2024"

  - name: "Report-Writer"
    description: "Creates comprehensive reports"
    model_name: "gpt-4"
    system_prompt: |
      You are a professional report writer.
      Create comprehensive, well-structured reports.
      Include executive summaries and key recommendations.
    temperature: 0.4
    max_loops: 1
    task: "Write an executive summary on AI in healthcare"
```

Run:

```bash
swarms run-agents --yaml-file research_pipeline.yaml
```

### Financial Analysis Team

```yaml
# financial_team.yaml
agents:
  - name: "Market-Analyst"
    description: "Analyzes market conditions"
    model_name: "gpt-4"
    system_prompt: |
      You are a CFA-certified market analyst.
      Provide detailed market analysis with technical indicators.
      Include risk assessments and market outlook.
    temperature: 0.2
    max_loops: 3
    task: "Analyze current S&P 500 market conditions"

  - name: "Risk-Assessor"
    description: "Evaluates investment risks"
    model_name: "gpt-4"
    system_prompt: |
      You are a risk management specialist.
      Evaluate investment risks and provide mitigation strategies.
      Use quantitative risk metrics.
    temperature: 0.1
    max_loops: 2
    task: "Assess risks in current tech sector investments"

  - name: "Portfolio-Advisor"
    description: "Provides portfolio recommendations"
    model_name: "gpt-4"
    system_prompt: |
      You are a portfolio advisor.
      Provide asset allocation recommendations.
      Consider risk tolerance and market conditions.
    temperature: 0.3
    max_loops: 2
    task: "Recommend portfolio adjustments for Q4 2024"
```

### Content Creation Pipeline

```yaml
# content_pipeline.yaml
agents:
  - name: "Topic-Researcher"
    description: "Researches content topics"
    model_name: "gpt-4o-mini"
    system_prompt: |
      You are a content researcher.
      Research topics thoroughly and identify key angles.
      Find unique perspectives and data points.
    temperature: 0.4
    max_loops: 2
    task: "Research content angles for 'Future of Remote Work'"

  - name: "Content-Writer"
    description: "Writes engaging content"
    model_name: "gpt-4"
    system_prompt: |
      You are a professional content writer.
      Write engaging, SEO-friendly content.
      Use clear structure with headers and bullet points.
    temperature: 0.7
    max_loops: 2
    task: "Write a blog post about remote work trends"

  - name: "Editor"
    description: "Edits and polishes content"
    model_name: "gpt-4o-mini"
    system_prompt: |
      You are a professional editor.
      Review content for clarity, grammar, and style.
      Suggest improvements and optimize for readability.
    temperature: 0.2
    max_loops: 1
    task: "Edit and polish the blog post for publication"
```

---

## Advanced Configuration

### Environment Variables in YAML

You can reference environment variables:

```yaml
agents:
  - name: "API-Agent"
    description: "Agent with API access"
    model_name: "${MODEL_NAME:-gpt-4o-mini}"  # Default if not set
    system_prompt: "You are an API integration specialist."
    task: "Test API integration"
```

### Multiple Config Files

Organize agents by purpose:

```bash
# Run different configurations
swarms run-agents --yaml-file research_agents.yaml
swarms run-agents --yaml-file analysis_agents.yaml
swarms run-agents --yaml-file reporting_agents.yaml
```

### Pipeline Script

```bash
#!/bin/bash
# run_pipeline.sh

echo "Starting research pipeline..."
swarms run-agents --yaml-file configs/research.yaml

echo "Starting analysis pipeline..."
swarms run-agents --yaml-file configs/analysis.yaml

echo "Starting reporting pipeline..."
swarms run-agents --yaml-file configs/reporting.yaml

echo "Pipeline complete!"
```

---

## Markdown Configuration

### Alternative: Load from Markdown

Create agents using markdown with YAML frontmatter:

```markdown
---
name: Research Agent
description: AI research specialist
model_name: gpt-4o-mini
temperature: 0.3
max_loops: 2
---

You are an expert researcher specializing in technology trends.
Provide comprehensive research summaries with:
- Key findings and insights
- Data points and statistics
- Recommendations and implications

Always cite sources when available and maintain objectivity.
```

Load from markdown:

```bash
# Load single file
swarms load-markdown --markdown-path ./agents/research_agent.md

# Load directory (concurrent processing)
swarms load-markdown --markdown-path ./agents/ --concurrent
```

---

## Best Practices

!!! tip "Configuration Management"
    - Version control your YAML files
    - Use descriptive agent names
    - Document purpose in descriptions

!!! tip "Template Organization"
    ```
    configs/
    ├── research/
    │   ├── tech_research.yaml
    │   └── market_research.yaml
    ├── analysis/
    │   ├── financial_analysis.yaml
    │   └── data_analysis.yaml
    └── production/
        └── prod_agents.yaml
    ```

!!! tip "Testing Configurations"
    - Test with `--verbose` flag first
    - Use lower `max_loops` for testing
    - Start with `gpt-4o-mini` for cost efficiency

!!! warning "Common Pitfalls"
    - Ensure proper YAML indentation (2 spaces)
    - Quote strings with special characters
    - Use `|` for multi-line prompts

---

## Next Steps

- [CLI Agent Guide](./cli_agent_guide.md) - Create agents from command line
- [CLI Multi-Agent Guide](../examples/cli_multi_agent_quickstart.md) - LLM Council and Heavy Swarm
- [CLI Reference](./cli_reference.md) - Complete command documentation

