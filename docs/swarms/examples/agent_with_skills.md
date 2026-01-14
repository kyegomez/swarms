# Basic Agent Skills Usage

This guide shows you how to use Agent Skills with Swarms agents in the simplest way possible.

## Prerequisites

- Swarms installed: `pip3 install -U swarms`
- An LLM API key (OpenAI, Anthropic, etc.)
- Example skills from the Swarms repository

## Quick Start

### Step 1: Basic Agent (Without Skills)

First, let's see a basic agent:

```python
from swarms import Agent

agent = Agent(
    agent_name="Research Agent",
    model_name="gpt-4o",
    max_loops=1
)

response = agent.run("How do I perform financial analysis?")
print(response)
```

This gives a generic response.

### Step 2: Agent With Skills

Now let's add the financial-analysis skill:

```python
from pathlib import Path
from swarms import Agent

# Path to example skills (adjust for your setup)
skills_path = "./example_skills"

agent = Agent(
    agent_name="Financial Analyst",
    model_name="gpt-4o",
    skills_dir=skills_path,  # ← Add this parameter
    max_loops=1
)

response = agent.run(
    "Explain the key steps in performing a DCF valuation"
)
print(response)
```

Now the agent follows the structured DCF methodology from the financial-analysis skill!

## Complete Example

```python
"""
Agent Skills - Basic Usage Example

This example shows how to use Agent Skills to specialize an agent
with the financial-analysis skill from the example_skills directory.
"""

from pathlib import Path
from swarms import Agent

# Get path to example_skills directory
repo_root = Path(__file__).parent.parent.parent
skills_path = repo_root / "example_skills"

# Create agent with skills
agent = Agent(
    agent_name="Financial Analyst",
    model_name="gpt-4o",
    max_loops=1,
    skills_dir=str(skills_path),
)

print("=" * 70)
print("Agent with Financial Analysis Skill")
print("=" * 70)

# Test 1: DCF Valuation
print("\n1. DCF Valuation Framework:\n")
response = agent.run(
    "Provide a framework for performing a DCF valuation on Apple Inc."
)
print(response)

# Test 2: Financial Ratios
print("\n" + "=" * 70)
print("\n2. Financial Ratio Analysis:\n")
response = agent.run(
    "What financial ratios should I analyze for a tech company?"
)
print(response)

# Test 3: Investment Recommendation
print("\n" + "=" * 70)
print("\n3. Investment Recommendation Structure:\n")
response = agent.run(
    "How should I structure an investment recommendation?"
)
print(response)

print("\n" + "=" * 70)
```

## What Skills Do

When you add `skills_dir`, the agent:

1. **Loads** all SKILL.md files from subdirectories
2. **Parses** YAML frontmatter (name, description)
3. **Injects** full skill content into system prompt
4. **Follows** the skill's methodology automatically

## Output Comparison

### Without Skills
```
Generic response about financial analysis...
```

### With Financial Analysis Skill
```
### 1. Data Collection and Verification
- Gather historical financial statements (income statement, balance sheet, cash flow)
- Verify data sources for accuracy
- Identify anomalies or missing data

### 2. Financial Ratio Analysis
Calculate and analyze key ratios:
- Profitability: EBITDA margin, net profit margin, ROE, ROA
- Liquidity: Current ratio, quick ratio
- Leverage: Debt-to-equity, interest coverage
...
```

The agent follows the exact methodology defined in the skill!

## Checking Loaded Skills

You can verify which skills are loaded:

```python
# Check loaded skills
print(f"Loaded {len(agent.skills_metadata)} skills:")
for skill in agent.skills_metadata:
    print(f"  - {skill['name']}: {skill['description'][:60]}...")
```

Output:
```
Loaded 3 skills:
  - financial-analysis: Perform comprehensive financial analysis...
  - code-review: Perform comprehensive code reviews focusing on...
  - data-visualization: Create effective data visualizations...
```

## Directory Structure

The example_skills directory looks like this:

```
example_skills/
├── financial-analysis/
│   └── SKILL.md          # Financial analysis methodology
├── code-review/
│   └── SKILL.md          # Code review checklist
└── data-visualization/
    └── SKILL.md          # Data viz best practices
```

## Using Different Skills

### Financial Analysis
```python
agent = Agent(
    agent_name="Finance Expert",
    model_name="gpt-4o",
    skills_dir="./example_skills"
)

agent.run("Perform a DCF analysis")  # Uses financial-analysis skill
```

### Code Review
```python
agent = Agent(
    agent_name="Code Reviewer",
    model_name="gpt-4o",
    skills_dir="./example_skills"
)

agent.run("Review this Python code")  # Uses code-review skill
```

### Data Visualization
```python
agent = Agent(
    agent_name="Data Viz Expert",
    model_name="gpt-4o",
    skills_dir="./example_skills"
)

agent.run("Best chart for trends?")  # Uses data-visualization skill
```

## Key Takeaways

1. **One Parameter**: Just add `skills_dir="./path/to/skills"`
2. **Automatic Loading**: Skills are loaded and injected at initialization
3. **Intelligent Use**: Agent applies skills when relevant to the task
4. **Multiple Skills**: Can load multiple skills from the same directory

## Next Steps

- [Create your own custom skills](/swarms/examples/agent_with_custom_skill/)
- [Use multiple skills together](/swarms/examples/agent_with_multiple_skills/)
- [Explore the SKILL.md format](/swarms/agents/agent_skills/)

## Troubleshooting

### Skills Not Loading

**Problem**: No skills loaded

**Solution**:
```bash
# Check directory exists
ls -R ./example_skills

# Verify SKILL.md files
cat ./example_skills/financial-analysis/SKILL.md
```

### Path Issues

**Problem**: `Skills directory not found`

**Solution**:
```python
import os
print(os.path.abspath("./example_skills"))  # Check full path
```

Use absolute paths or `Path` from `pathlib`:
```python
from pathlib import Path
skills_path = Path(__file__).parent / "example_skills"
```

### No Visible Change

**Problem**: Agent doesn't seem to use skills

**Solution**:
Enable verbose mode to see skill loading:
```python
agent = Agent(
    skills_dir="./example_skills",
    verbose=True  # See loading logs
)
```

## Complete Working Example

Save this as `test_skills.py`:

```python
from swarms import Agent

# Use relative path to example_skills
agent = Agent(
    agent_name="Financial Analyst",
    model_name="gpt-4o",
    skills_dir="../../example_skills",  # Adjust path as needed
    max_loops=1
)

# Ask a financial analysis question
response = agent.run(
    "What are the steps to perform a DCF valuation?"
)

print(response)
```

Run it:
```bash
python3 test_skills.py
```

You should see the agent follow the structured DCF methodology from the skill!

## Resources

- [Agent Skills Documentation](/swarms/agents/agent_skills/)
- [Example Skills Source](https://github.com/kyegomez/swarms/tree/master/example_skills)
- [More Examples](https://github.com/kyegomez/swarms/tree/master/examples/single_agent)
