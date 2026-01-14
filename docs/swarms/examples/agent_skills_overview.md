# Agent Skills Overview

Agent Skills is a simple, powerful way to specialize agents using markdown files. This guide covers everything you need to know to use skills in your Swarms agents.

## What are Agent Skills?

Agent Skills are modular capabilities defined in `SKILL.md` files that guide agent behavior. They enable you to:

- **Specialize agents** without modifying code
- **Share expertise** across your team through skill libraries
- **Maintain consistency** in how agents perform tasks
- **Rapidly iterate** by editing markdown instead of rebuilding prompts

## Quick Example

```python
from swarms import Agent

agent = Agent(
    agent_name="Financial Analyst",
    model_name="gpt-4o",
    skills_dir="./example_skills",  # ← Just add this parameter!
    max_loops=1
)

# Agent automatically follows financial-analysis skill instructions
response = agent.run("Perform a DCF valuation for Tesla")
```

## SKILL.md Format

Skills use a simple structure:

```yaml
---
name: my-skill
description: What this skill does
---

# Skill Instructions

Detailed guidance for the agent...

## Guidelines
- Key principles
- Best practices

## Examples
- Example use case 1
- Example use case 2
```

## Directory Structure

```
skills/
├── financial-analysis/
│   └── SKILL.md
├── code-review/
│   └── SKILL.md
└── customer-support/
    └── SKILL.md
```

## Core Concepts

### 1. Skill Loading

When you specify `skills_dir`, Swarms:
1. Scans for subdirectories containing `SKILL.md`
2. Parses YAML frontmatter (name, description)
3. Loads full markdown content
4. Injects into agent's system prompt

### 2. Skill Activation

Skills are **always active** once loaded. The agent sees all skill instructions and applies them when relevant to the task.

### 3. Multiple Skills

Agents can have multiple skills loaded simultaneously. They'll intelligently apply the right guidance based on the task.

## Example Skills Included

Swarms includes 3 production-ready skills:

### Financial Analysis
```bash
example_skills/financial-analysis/SKILL.md
```
- DCF valuation methodology
- Financial ratio analysis
- Sensitivity analysis frameworks
- Investment recommendations

### Code Review
```bash
example_skills/code-review/SKILL.md
```
- Security vulnerability detection
- Performance optimization checks
- Best practices enforcement
- Maintainability assessment

### Data Visualization
```bash
example_skills/data-visualization/SKILL.md
```
- Chart type selection
- Design principles
- Color best practices
- Storytelling frameworks

## Common Use Cases

### 1. Domain Expertise

Add specialized knowledge:
```python
# Financial analysis agent
agent = Agent(
    agent_name="Finance Expert",
    skills_dir="./skills/finance"
)
```

### 2. Process Enforcement

Ensure consistent methodologies:
```python
# Code review with company standards
agent = Agent(
    agent_name="Code Reviewer",
    skills_dir="./skills/code-standards"
)
```

### 3. Communication Styles

Define tone and formatting:
```python
# Customer support with brand voice
agent = Agent(
    agent_name="Support Agent",
    skills_dir="./skills/support"
)
```

## Next Steps

Explore the example guides:

1. **[Basic Skills Usage](/swarms/examples/agent_with_skills/)** - Start here for your first skill
2. **[Creating Custom Skills](/swarms/examples/agent_with_custom_skill/)** - Build your own skills
3. **[Multiple Skills](/swarms/examples/agent_with_multiple_skills/)** - Use multiple skills together

## Key Benefits

- ✅ **Simple**: Just markdown files, no code needed
- ✅ **Portable**: Standard format works across platforms
- ✅ **Version Control**: Skills are just files - track with git
- ✅ **Reusable**: Share skills across your organization
- ✅ **Claude Code Compatible**: Works with Anthropic's ecosystem

## Compatibility

Swarms Agent Skills implementation follows [Anthropic's Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) standard, ensuring compatibility with Claude Code and other tools that use the same format.

### What This Means

- Skills created for Swarms work with Claude Code
- Skills from Claude Code work with Swarms
- Same `SKILL.md` format across platforms
- Part of an emerging ecosystem standard

## Resources

- [Main Agent Skills Documentation](/swarms/agents/agent_skills/)
- [Anthropic Agent Skills Blog](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [Official GitHub Repository](https://github.com/anthropics/skills)
- [Code Examples](https://github.com/kyegomez/swarms/tree/master/examples/single_agent)
