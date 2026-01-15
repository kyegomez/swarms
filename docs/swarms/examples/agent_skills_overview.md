# Agent Skills

Agent Skills let you specialize AI agents using simple markdown files. Add domain expertise to your agents without writing code - just create `SKILL.md` files in a directory.

## How It Works

1. **Create Skills**: Write `SKILL.md` files with instructions for specific tasks
2. **Load Directory**: Point your agent to a skills folder
3. **Automatic Application**: Agent uses relevant skills based on your prompts

## Performance Benefits

Skills dramatically improve agent performance by:

- **Domain Expertise**: Agents follow proven methodologies instead of generic responses
- **Consistency**: Same approach every time for similar tasks
- **Specialization**: Focus on specific domains rather than being general-purpose
- **Rapid Iteration**: Edit markdown files instead of retraining models

## Quick Example

```python
from swarms import Agent

# Without skills - generic response
basic_agent = Agent(agent_name="Assistant", model_name="gpt-4o")
basic_response = basic_agent.run("How do I analyze company financials?")
# → Generic explanation

# With skills - specialized response
skilled_agent = Agent(
    agent_name="Financial Analyst",
    model_name="gpt-4o",
    skills_dir="./skills"  # Contains financial-analysis skill
)
skilled_response = skilled_agent.run("How do I analyze company financials?")
# → Structured DCF methodology with specific steps
```

## Skill Schema

Skills use a simple markdown format with YAML frontmatter:

```markdown
---
name: financial-analysis
description: Perform comprehensive financial analysis including DCF modeling and ratio analysis
---

# Financial Analysis Skill

When performing financial analysis, follow these systematic steps:

## Core Methodology

### 1. Data Collection
- Gather income statement, balance sheet, cash flow
- Verify data accuracy and completeness

### 2. Financial Ratios
Calculate key ratios:
- EBITDA margin = (EBITDA / Revenue) × 100
- Current ratio = Current Assets / Current Liabilities

### 3. Valuation Models
- DCF: Project cash flows and discount to present value
- Comparables: Compare to similar companies

## Guidelines
- Use conservative assumptions when uncertain
- Cross-validate with multiple methods
- Clearly document all assumptions
```

### Required Fields

| Field       | Type   | Description                          |
|-------------|--------|--------------------------------------|
| `name`      | string | Unique skill identifier              |
| `description` | string | What the skill does and when to use it |

### Directory Structure

```text
skills/
├── financial-analysis/
│   └── SKILL.md
├── code-review/
│   └── SKILL.md
└── data-visualization/
    └── SKILL.md
```

## Usage

```python
from swarms import Agent

# Basic usage - load all skills from directory
agent = Agent(
    agent_name="Specialist",
    model_name="gpt-4o",
    skills_dir="./skills"  # Points to folder with SKILL.md files
)

# Agent automatically uses relevant skills
response = agent.run("Analyze this company's financial statements")
```

## Built-in Examples

| Skill                | What it does                                | Example Prompt                  |
|----------------------|---------------------------------------------|---------------------------------|
| **financial-analysis** | DCF valuation, ratio analysis, financial modeling | "Perform DCF analysis on Tesla" |
| **code-review**      | Security checks, performance optimization, best practices | "Review this Python code for issues" |
| **data-visualization** | Chart selection, design principles, storytelling | "Best chart for showing sales trends" |

## Creating Custom Skills

1. Create a folder: `mkdir my-skills/customer-support`
2. Add `SKILL.md`:

```markdown
---
name: customer-support
description: Handle customer inquiries with empathy and efficiency
---

# Customer Support Skill

## Approach
1. Acknowledge the issue
2. Ask clarifying questions
3. Provide clear solutions
4. Offer follow-up help

## Tone
- Professional yet friendly
- Patient and understanding
- Solution-oriented
```

1. Use with agent:

```python
agent = Agent(
    agent_name="Support Agent",
    model_name="gpt-4o",
    skills_dir="./my-skills"
)
```

## Compatibility

Agent Skills follow [Anthropic's Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) standard, ensuring compatibility with Claude Code and other compliant tools.

Skills created for Swarms work with Claude Code, and vice versa.

## Resources

- [Anthropic Agent Skills Specification](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [Code Examples](https://github.com/kyegomez/swarms/tree/master/examples/single_agent/agent_skill_examples)
