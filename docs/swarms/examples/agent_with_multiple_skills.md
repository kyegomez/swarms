# Using Multiple Agent Skills

This guide shows how to use multiple skills with a single agent and have the agent intelligently apply the right expertise based on the task.

## Overview

When you load multiple skills, the agent sees all of them in its system prompt and automatically applies the relevant guidance based on what you ask it to do.

## Quick Example

```python
from swarms import Agent

# Agent with access to all 3 example skills
agent = Agent(
    agent_name="Multi-Skilled Agent",
    model_name="gpt-4o",
    skills_dir="./example_skills",  # Contains 3 skills
    max_loops=1
)

# Financial task - uses financial-analysis skill
agent.run("Explain DCF valuation")

# Code task - uses code-review skill
agent.run("Review this code for bugs")

# Visualization task - uses data-visualization skill
agent.run("Best chart for trends?")
```

## Complete Working Example

Save this as `multi_skills_demo.py`:

```python
"""
Multiple Skills Example

Shows how an agent intelligently uses different skills
based on the task at hand.
"""

from pathlib import Path
from swarms import Agent

# Get path to example_skills directory
repo_root = Path(__file__).parent.parent.parent
skills_path = repo_root / "example_skills"

# Create agent with multiple skills
agent = Agent(
    agent_name="Multi-Skilled Agent",
    model_name="gpt-4o",
    max_loops=1,
    skills_dir=str(skills_path),
)

print("=" * 70)
print("Multi-Skilled Agent Demo")
print("=" * 70)

# Check what skills are loaded
print(f"\nLoaded {len(agent.skills_metadata)} skills:")
for skill in agent.skills_metadata:
    print(f"  - {skill['name']}: {skill['description'][:60]}...")

# Task 1: Financial Analysis
print("\n" + "=" * 70)
print("Task 1: Financial Analysis (uses financial-analysis skill)")
print("=" * 70)
response = agent.run(
    "What are the key steps in performing a DCF valuation?"
)
print(response[:500] + "...\n")

# Task 2: Code Review
print("=" * 70)
print("Task 2: Code Review (uses code-review skill)")
print("=" * 70)
code = """
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    return execute_query(query)
"""
response = agent.run(
    f"Review this code for security issues:\n{code}"
)
print(response[:500] + "...\n")

# Task 3: Data Visualization
print("=" * 70)
print("Task 3: Data Visualization (uses data-visualization skill)")
print("=" * 70)
response = agent.run(
    "What's the best chart type for showing quarterly sales trends over 3 years?"
)
print(response[:500] + "...\n")

# Task 4: Mixed Context
print("=" * 70)
print("Task 4: Mixed Context (uses multiple skills)")
print("=" * 70)
response = agent.run(
    "Create a financial dashboard that shows revenue trends with proper visualization"
)
print(response[:500] + "...\n")

print("=" * 70)
print("\n✓ Agent successfully used different skills for different tasks!")
```

Run it:
```bash
python3 multi_skills_demo.py
```

## How It Works

### Skill Loading

When you specify `skills_dir`, all skills in subdirectories are loaded:

```
example_skills/
├── financial-analysis/SKILL.md  → Loaded
├── code-review/SKILL.md         → Loaded
└── data-visualization/SKILL.md  → Loaded
```

### Skill Injection

All skills are appended to the system prompt:

```
[Your original system_prompt]

# Available Skills

## financial-analysis
**Description**: Perform comprehensive financial analysis...
[Full financial-analysis instructions]

---

## code-review
**Description**: Perform comprehensive code reviews...
[Full code-review instructions]

---

## data-visualization
**Description**: Create effective data visualizations...
[Full data-visualization instructions]
```

### Intelligent Application

The agent reads all skills and applies relevant guidance based on context:

- **Financial question** → Follows financial-analysis methodology
- **Code question** → Follows code-review checklist
- **Visualization question** → Follows data-visualization principles
- **Mixed question** → Combines relevant skills

## Real-World Use Cases

### 1. Full-Stack Development Agent

```python
# Skills for different aspects of development
skills/
├── frontend-review/
│   └── SKILL.md        # React, CSS, accessibility
├── backend-review/
│   └── SKILL.md        # API design, database, security
├── testing/
│   └── SKILL.md        # Unit tests, integration tests
└── documentation/
    └── SKILL.md        # Code docs, API docs, README

agent = Agent(
    agent_name="Full-Stack Reviewer",
    model_name="gpt-4o",
    skills_dir="./skills"
)

# Uses appropriate skill based on question
agent.run("Review my React component")      # → frontend-review
agent.run("Review my API endpoint")         # → backend-review
agent.run("Write tests for this function")  # → testing
```

### 2. Business Analyst Agent

```python
skills/
├── market-research/
│   └── SKILL.md
├── financial-modeling/
│   └── SKILL.md
├── competitor-analysis/
│   └── SKILL.md
└── strategic-planning/
    └── SKILL.md

agent = Agent(
    agent_name="Business Analyst",
    model_name="gpt-4o",
    skills_dir="./skills"
)

agent.run("Analyze market size")           # → market-research
agent.run("Build revenue projection")      # → financial-modeling
agent.run("Compare to competitors")        # → competitor-analysis
agent.run("Recommend strategy")            # → strategic-planning
```

### 3. Research Assistant Agent

```python
skills/
├── literature-review/
│   └── SKILL.md
├── data-analysis/
│   └── SKILL.md
├── paper-writing/
│   └── SKILL.md
└── citation-formatting/
    └── SKILL.md

agent = Agent(
    agent_name="Research Assistant",
    model_name="gpt-4o",
    skills_dir="./skills"
)

agent.run("Summarize these papers")        # → literature-review
agent.run("Analyze this dataset")          # → data-analysis
agent.run("Write methods section")         # → paper-writing
agent.run("Format citations in APA")       # → citation-formatting
```

## Skill Combinations

Sometimes tasks naturally combine multiple skills:

```python
# Task that uses both financial-analysis and data-visualization
response = agent.run(
    """
    Analyze our quarterly revenue and create a visualization
    showing the trends with key financial metrics
    """
)

# The agent will:
# 1. Use financial-analysis skill for metrics/ratios
# 2. Use data-visualization skill for chart selection
# 3. Combine both for comprehensive output
```

## Best Practices

### 1. Organize Skills by Domain

```
skills/
├── technical/
│   ├── code-review/
│   ├── architecture/
│   └── testing/
├── business/
│   ├── finance/
│   ├── marketing/
│   └── sales/
└── communication/
    ├── support/
    └── documentation/
```

### 2. Avoid Redundancy

Don't create overlapping skills:

```yaml
# Bad: Two similar skills
skills/
├── code-quality/
└── code-review/

# Good: One comprehensive skill
skills/
└── code-review/  # Covers quality, security, performance
```

### 3. Keep Skills Focused

```yaml
# Bad: Skill that does everything
mega-skill/
  └── SKILL.md  # 10 pages covering everything

# Good: Focused skills
skills/
├── frontend/
├── backend/
└── testing/
```

### 4. Name Skills Clearly

```yaml
# Good names
customer-support/
financial-analysis/
code-review/

# Bad names
skill-1/
helper/
misc/
```

## Checking Loaded Skills

Always verify which skills are loaded:

```python
print(f"Loaded {len(agent.skills_metadata)} skills:")
for skill in agent.skills_metadata:
    print(f"\n{skill['name']}")
    print(f"  Description: {skill['description']}")
    print(f"  Path: {skill['path']}")
```

## Performance Considerations

### Context Window

Each skill adds to the system prompt. Monitor token usage:

```python
from swarms.utils.litellm_tokenizer import count_tokens

# Count tokens in system prompt
token_count = count_tokens(agent.system_prompt, agent.model_name)
print(f"System prompt uses {token_count} tokens")
```

### Guidelines

- **1-5 skills**: Optimal for most use cases
- **5-10 skills**: Monitor token usage
- **10+ skills**: Consider splitting into specialized agents

### Optimization

If you have many skills, create specialized agents:

```python
# Instead of one agent with 20 skills
big_agent = Agent(skills_dir="./all_skills")  # 20 skills

# Use specialized agents
frontend_agent = Agent(skills_dir="./frontend_skills")  # 3 skills
backend_agent = Agent(skills_dir="./backend_skills")    # 4 skills
data_agent = Agent(skills_dir="./data_skills")          # 3 skills
```

## Dynamic Skill Loading

You can change skills at runtime by creating new agents:

```python
# Start with general skills
agent1 = Agent(
    agent_name="General Agent",
    skills_dir="./general_skills"
)

# Switch to specialized skills for specific task
agent2 = Agent(
    agent_name="Finance Expert",
    skills_dir="./finance_skills"
)
```

## Skill Conflicts

What if skills have contradictory guidance?

```yaml
# skill-a/SKILL.md
"Always use verbose output"

# skill-b/SKILL.md
"Keep responses concise"
```

**Solution**: The agent will use its judgment based on context. For better control:

1. **Prioritize in descriptions**:
   ```yaml
   description: "Primary code review skill - use unless specified otherwise"
   ```

2. **Be specific in tasks**:
   ```python
   agent.run("Review code using security-focused approach")
   ```

3. **Use separate agents** for conflicting methodologies

## Testing Multiple Skills

Test that each skill activates correctly:

```python
test_cases = {
    "financial": "Perform DCF analysis",
    "code": "Review this code for bugs",
    "visualization": "Best chart for trends?",
}

for skill_type, task in test_cases.items():
    print(f"\n=== Testing {skill_type} skill ===")
    response = agent.run(task)

    # Check if skill was used
    if skill_type in response.lower():
        print(f"✓ {skill_type} skill activated")
    else:
        print(f"✗ {skill_type} skill may not have activated")
```

## Advanced: Conditional Skills

Load skills based on environment or user:

```python
import os

# Development environment - load all skills
if os.getenv("ENV") == "development":
    skills_dir = "./all_skills"
else:
    # Production - load only approved skills
    skills_dir = "./production_skills"

agent = Agent(
    agent_name="Adaptive Agent",
    skills_dir=skills_dir
)
```

## Next Steps

- [Explore the main Agent Skills documentation](/swarms/agents/agent_skills/)
- [Create your own custom skills](/swarms/examples/agent_with_custom_skill/)
- [View example skills source code](https://github.com/kyegomez/swarms/tree/master/example_skills)

## Resources

- [Agent Skills Documentation](/swarms/agents/agent_skills/)
- [Basic Skills Usage](/swarms/examples/agent_with_skills/)
- [Creating Custom Skills](/swarms/examples/agent_with_custom_skill/)
- [GitHub Examples](https://github.com/kyegomez/swarms/tree/master/examples/single_agent)
