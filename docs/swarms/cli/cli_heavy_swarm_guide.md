# CLI Heavy Swarm Guide: Comprehensive Task Analysis

Run Heavy Swarm from command line for complex task decomposition and comprehensive analysis with specialized agents.

## Overview

Heavy Swarm follows a structured workflow:

1. **Task Decomposition**: Breaks down tasks into specialized questions
2. **Parallel Execution**: Executes specialized agents in parallel
3. **Result Synthesis**: Integrates and synthesizes results
4. **Comprehensive Reporting**: Generates detailed final reports

---

## Basic Usage

### Step 1: Run a Simple Analysis

```bash
swarms heavy-swarm --task "Analyze the current state of quantum computing"
```

### Step 2: Customize with Options

```bash
swarms heavy-swarm \
    --task "Research renewable energy market trends" \
    --loops-per-agent 2 \
    --verbose
```

### Step 3: Use Custom Models

```bash
swarms heavy-swarm \
    --task "Analyze cryptocurrency regulation globally" \
    --question-agent-model-name gpt-4 \
    --worker-model-name gpt-4 \
    --loops-per-agent 3 \
    --verbose
```

---

## Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--task` | **Required** | The task to analyze |
| `--loops-per-agent` | 1 | Execution loops per agent |
| `--question-agent-model-name` | gpt-4o-mini | Model for question generation |
| `--worker-model-name` | gpt-4o-mini | Model for worker agents |
| `--random-loops-per-agent` | False | Randomize loops (1-10) |
| `--verbose` | False | Enable detailed output |

---

## Specialized Agents

Heavy Swarm includes specialized agents for different aspects:

| Agent | Role | Focus |
|-------|------|-------|
| **Question Agent** | Decomposes tasks | Generates targeted questions |
| **Research Agent** | Gathers information | Fast, trustworthy research |
| **Analysis Agent** | Processes data | Statistical analysis, insights |
| **Writing Agent** | Creates reports | Clear, structured documentation |

---

## Use Case Examples

### Market Research

```bash
swarms heavy-swarm \
    --task "Comprehensive market analysis of the electric vehicle industry in North America" \
    --loops-per-agent 3 \
    --question-agent-model-name gpt-4 \
    --worker-model-name gpt-4 \
    --verbose
```

### Technology Assessment

```bash
swarms heavy-swarm \
    --task "Evaluate the technical feasibility and ROI of implementing AI-powered customer service automation" \
    --loops-per-agent 2 \
    --verbose
```

### Competitive Analysis

```bash
swarms heavy-swarm \
    --task "Analyze competitive landscape for cloud computing services: AWS vs Azure vs Google Cloud" \
    --loops-per-agent 2 \
    --question-agent-model-name gpt-4 \
    --verbose
```

### Investment Research

```bash
swarms heavy-swarm \
    --task "Research investment opportunities in AI infrastructure companies for 2024-2025" \
    --loops-per-agent 3 \
    --worker-model-name gpt-4 \
    --verbose
```

### Policy Analysis

```bash
swarms heavy-swarm \
    --task "Analyze the impact of proposed AI regulations on tech startups in the United States" \
    --loops-per-agent 2 \
    --verbose
```

### Due Diligence

```bash
swarms heavy-swarm \
    --task "Conduct technology due diligence for acquiring a fintech startup focusing on payment processing" \
    --loops-per-agent 3 \
    --question-agent-model-name gpt-4 \
    --worker-model-name gpt-4 \
    --verbose
```

---

## Workflow Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Task                                │
│  "Analyze the impact of AI on healthcare"                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Question Agent                              │
│  Decomposes task into specialized questions:                     │
│  - What are current AI applications in healthcare?              │
│  - What are the regulatory challenges?                          │
│  - What is the market size and growth?                          │
│  - What are the key players and competitors?                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Research   │  Analysis   │  Research   │   Writing   │
│   Agent 1   │   Agent     │   Agent 2   │    Agent    │
└─────────────┴─────────────┴─────────────┴─────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Synthesis & Integration                     │
│              Combines all agent outputs                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Comprehensive Report                          │
│  - Executive Summary                                            │
│  - Detailed Findings                                            │
│  - Analysis & Insights                                          │
│  - Recommendations                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Scripting Examples

### Research Pipeline

```bash
#!/bin/bash
# research_pipeline.sh

TOPICS=(
    "AI in manufacturing"
    "Blockchain in supply chain"
    "Edge computing in IoT"
)

for topic in "${TOPICS[@]}"; do
    echo "Researching: $topic"
    OUTPUT_FILE="research_$(echo $topic | tr ' ' '_').txt"
    
    swarms heavy-swarm \
        --task "Comprehensive analysis of $topic: market size, key players, trends, and opportunities" \
        --loops-per-agent 2 \
        --verbose > "$OUTPUT_FILE"
    
    echo "Saved to: $OUTPUT_FILE"
done
```

### Daily Market Analysis

```bash
#!/bin/bash
# daily_market.sh

DATE=$(date +%Y-%m-%d)
OUTPUT_FILE="market_analysis_$DATE.txt"

echo "Daily Market Analysis - $DATE" > $OUTPUT_FILE
echo "==============================" >> $OUTPUT_FILE

swarms heavy-swarm \
    --task "Analyze today's key market movements, notable news, and outlook for tomorrow. Focus on tech, healthcare, and energy sectors." \
    --loops-per-agent 2 \
    --question-agent-model-name gpt-4 \
    --worker-model-name gpt-4 \
    --verbose >> $OUTPUT_FILE

echo "Analysis complete: $OUTPUT_FILE"
```

### CI/CD Integration

```yaml
# .github/workflows/heavy-swarm-research.yml
name: Weekly Heavy Swarm Research

on:
  schedule:
    - cron: '0 6 * * 1'  # Every Monday at 6 AM

jobs:
  research:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install Swarms
        run: pip install swarms
      
      - name: Run Heavy Swarm Research
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          swarms heavy-swarm \
            --task "Weekly technology trends and market analysis report" \
            --loops-per-agent 3 \
            --question-agent-model-name gpt-4 \
            --worker-model-name gpt-4 \
            --verbose > weekly_research.txt
      
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: weekly-research
          path: weekly_research.txt
```

---

## Configuration Recommendations

### Quick Analysis (Cost-Effective)

```bash
swarms heavy-swarm \
    --task "Quick overview of [topic]" \
    --loops-per-agent 1 \
    --question-agent-model-name gpt-4o-mini \
    --worker-model-name gpt-4o-mini
```

### Standard Research

```bash
swarms heavy-swarm \
    --task "Detailed analysis of [topic]" \
    --loops-per-agent 2 \
    --verbose
```

### Deep Dive (Comprehensive)

```bash
swarms heavy-swarm \
    --task "Comprehensive research on [topic]" \
    --loops-per-agent 3 \
    --question-agent-model-name gpt-4 \
    --worker-model-name gpt-4 \
    --verbose
```

### Exploratory (Variable Depth)

```bash
swarms heavy-swarm \
    --task "Explore [topic] with varying depth" \
    --random-loops-per-agent \
    --verbose
```

---

## Output Processing

### Save to File

```bash
swarms heavy-swarm --task "Your task" > report.txt 2>&1
```

### Extract Sections

```bash
# Get executive summary
swarms heavy-swarm --task "Your task" | grep -A 50 "Executive Summary"

# Get recommendations
swarms heavy-swarm --task "Your task" | grep -A 20 "Recommendations"
```

### Timestamp Output

```bash
swarms heavy-swarm --task "Your task" | while read line; do
    echo "[$(date '+%H:%M:%S')] $line"
done
```

---

## Best Practices

!!! tip "Task Formulation"
    - Be specific about what you want analyzed
    - Include scope and constraints
    - Specify desired output format

!!! tip "Loop Configuration"
    - Use `--loops-per-agent 1` for quick overviews
    - Use `--loops-per-agent 2-3` for detailed analysis
    - Higher loops = more comprehensive but slower

!!! tip "Model Selection"
    - Use `gpt-4o-mini` for cost-effective analysis
    - Use `gpt-4` for complex, nuanced topics
    - Match model to task complexity

!!! warning "Performance Notes"
    - Deep analysis (3+ loops) may take several minutes
    - Higher loops increase API costs
    - Use `--verbose` to monitor progress

---

## Comparison: LLM Council vs Heavy Swarm

| Feature | LLM Council | Heavy Swarm |
|---------|-------------|-------------|
| **Focus** | Collaborative decision-making | Comprehensive task analysis |
| **Workflow** | Parallel responses + peer review | Task decomposition + parallel research |
| **Best For** | Questions with multiple viewpoints | Complex research and analysis tasks |
| **Output** | Synthesized consensus | Detailed research report |
| **Speed** | Faster | More thorough but slower |

---

## Next Steps

- [CLI LLM Council Guide](./cli_llm_council_guide.md) - Collaborative decisions
- [CLI Reference](./cli_reference.md) - Complete command documentation
- [Heavy Swarm Python API](../structs/heavy_swarm.md) - Programmatic usage

