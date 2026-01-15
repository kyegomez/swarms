# Agent Skills

## Overview

Agent Skills is a lightweight, markdown-based format for defining modular, reusable agent capabilities. Introduced by Anthropic in December 2024, Agent Skills enables you to specialize agents without modifying code by loading skill definitions from simple `SKILL.md` files.

Swarms implements full support for the Agent Skills standard, making it **compatible with Anthropic's Claude Code** and other tools that use the same format.

## Key Benefits

- **Modular**: Skills are separate files, easy to create, share, and maintain
- **Reusable**: Build skill libraries for common tasks across your organization
- **Portable**: Standard format works across platforms and tools
- **Simple**: Just markdown files with YAML frontmatter - no code needed
- **Compatible**: Works with Anthropic's Agent Skills ecosystem

## SKILL.md Format

Skills are defined using a simple structure:

```yaml
---
name: skill-name
description: A clear description of what this skill does
---

# Skill Name

Instructions that guide the agent's behavior when this skill is active.

## Guidelines
- Guideline 1
- Guideline 2

## Examples
- Example usage 1
- Example usage 2
```

### Required Components

1. **YAML Frontmatter**:
   - `name`: Unique identifier for the skill (lowercase, hyphens for spaces)
   - `description`: Clear description of what the skill does and when to use it

2. **Markdown Content**: Detailed instructions, guidelines, examples, and methodologies

## Quick Start

### Basic Usage

```python
from swarms import Agent

# Create agent with skills
agent = Agent(
    agent_name="Financial Analyst",
    model_name="gpt-4o",
    skills_dir="./example_skills",  # Point to your skills directory
    max_loops=1
)

# Agent automatically follows skill instructions
response = agent.run("Perform a DCF valuation for Apple")
```

### Directory Structure

```
example_skills/
├── financial-analysis/
│   └── SKILL.md
├── code-review/
│   └── SKILL.md
└── data-visualization/
    └── SKILL.md
```

## Creating Custom Skills

### Step 1: Create Directory

```bash
mkdir -p ./my_skills/customer-support
```

### Step 2: Create SKILL.md

Create `./my_skills/customer-support/SKILL.md`:

```yaml
---
name: customer-support
description: Handle customer support inquiries with empathy and efficiency
---

# Customer Support Skill

When responding to customer support inquiries:

## Approach
1. Acknowledge the customer's issue
2. Ask clarifying questions if needed
3. Provide clear, step-by-step solutions
4. Offer additional help

## Tone
- Professional yet friendly
- Patient and understanding
- Solution-oriented
```

### Step 3: Use Your Skill

```python
agent = Agent(
    agent_name="Support Agent",
    model_name="gpt-4o",
    skills_dir="./my_skills"
)

response = agent.run("Customer says: I was charged twice!")
```

## How Skills Work

When you specify `skills_dir`:

1. **Loading**: Swarms scans the directory for subdirectories containing `SKILL.md` files
2. **Parsing**: YAML frontmatter is parsed to extract name and description
3. **Injection**: Full skill content is appended to the agent's system prompt
4. **Activation**: Agent automatically follows skill instructions when relevant to the task

### System Prompt Integration

Skills are injected into the system prompt like this:

```
[Your original system_prompt]

# Available Skills

## financial-analysis

**Description**: Perform comprehensive financial analysis...

# Financial Analysis Skill

[Full instructions from SKILL.md]

---

## code-review

**Description**: Perform comprehensive code reviews...

[Full instructions from SKILL.md]
```

## Example Skills Included

Swarms comes with 3 production-ready example skills:

### 1. Financial Analysis
**Location**: `example_skills/financial-analysis/SKILL.md`

Provides systematic approach for:
- DCF valuation modeling
- Financial ratio analysis
- Sensitivity analysis
- Investment recommendations

### 2. Code Review
**Location**: `example_skills/code-review/SKILL.md`

Covers:
- Security vulnerability detection
- Performance optimization
- Best practices enforcement
- Maintainability assessment

### 3. Data Visualization
**Location**: `example_skills/data-visualization/SKILL.md`

Includes:
- Chart type selection
- Design principles
- Color best practices
- Storytelling with data

## Multiple Skills

Agents can access multiple skills simultaneously:

```python
agent = Agent(
    agent_name="Multi-Skilled Agent",
    model_name="gpt-4o",
    skills_dir="./example_skills",  # Contains multiple skills
)

# Financial task - uses financial-analysis skill
agent.run("What's the DCF formula?")

# Code task - uses code-review skill
agent.run("Review this code for security issues")

# Visualization task - uses data-visualization skill
agent.run("Best chart for quarterly trends?")
```

## Claude Code Compatibility

Swarms Agent Skills implementation is **fully compatible** with [Anthropic's Agent Skills](https://github.com/anthropics/skills) standard.

### What This Means

- **Portable Skills**: Skills created for Swarms work with Claude Code and vice versa
- **Standard Format**: Uses the same SKILL.md format as Claude Code
- **Ecosystem Compatible**: Can share skills across tools that implement the standard
- **Future-Proof**: Aligned with Anthropic's roadmap for agent development

### Differences from Claude API Implementation

| Aspect | Swarms | Claude API |
|--------|--------|------------|
| **Loading** | All content loaded into system prompt | Progressive disclosure (3-tier) |
| **Activation** | Automatic based on content | Container-based execution |
| **LLM Support** | Any LLM (OpenAI, Anthropic, Groq, etc.) | Claude models only |
| **Format** | Identical SKILL.md format | Identical SKILL.md format |

Both implementations use the same SKILL.md format, making skills portable between Swarms and Claude Code.

## Advanced Features

### Checking Loaded Skills

```python
# See what skills are loaded
print(f"Loaded {len(agent.skills_metadata)} skills:")
for skill in agent.skills_metadata:
    print(f"- {skill['name']}: {skill['description']}")
```

### Programmatic Skill Access

```python
# Load full skill content
skill_content = agent.load_full_skill("financial-analysis")
print(skill_content)
```

### Git Configuration

Skills directories are automatically ignored by git to prevent committing private/proprietary skills:

**Ignored patterns** (won't be committed):
- `/skills/`
- `/my_skills/`
- `/custom_skills/`
- `/*_skills/`

**Exception**: `example_skills/` is kept in the repository for reference.

## Best Practices

### Writing Effective Skills

**✅ DO:**
- Write clear, specific instructions
- Include concrete examples
- Use bullet points and sections
- Focus on methodology ("how") not just goals ("what")
- Test skills with real tasks

**❌ DON'T:**
- Be vague or general
- Write walls of text without structure
- Assume prior knowledge
- Forget to include examples

### Organizing Skills

```
skills/
├── domain/
│   ├── finance/
│   │   └── SKILL.md
│   └── legal/
│       └── SKILL.md
├── process/
│   ├── code-review/
│   │   └── SKILL.md
│   └── testing/
│       └── SKILL.md
└── communication/
    └── support/
        └── SKILL.md
```

### Skill Naming Conventions

- Use lowercase with hyphens: `financial-analysis`, not `FinancialAnalysis`
- Be descriptive: `customer-support` not `cs`
- Avoid special characters except hyphens
- Keep names under 64 characters

## Examples

See the [Agent Skills Examples](/swarms/examples/agent_skills_overview/) section for:
- Basic usage tutorial
- Creating custom skills
- Using multiple skills
- Real-world applications

## References

- [Anthropic Agent Skills Blog Post](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [Official GitHub Repository](https://github.com/anthropics/skills)
- [Agent Skills Documentation](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
- [Swarms Examples](https://github.com/kyegomez/swarms/tree/master/examples)

## Troubleshooting

### Skills Not Loading

**Problem**: Agent doesn't seem to use skills

**Solution**:
1. Check directory structure: `ls -R ./example_skills`
2. Verify YAML frontmatter syntax (must start/end with `---`)
3. Enable verbose mode: `Agent(skills_dir="./skills", verbose=True)`

### YAML Parsing Errors

**Problem**: `Failed to load skill from...`

**Solution**:
- Check YAML syntax (use spaces, not tabs)
- Ensure frontmatter starts with `---` on line 1
- Validate YAML: `python -c "import yaml; yaml.safe_load(open('SKILL.md').read())"`

## API Reference

### Agent Parameters

```python
Agent(
    skills_dir: Optional[str] = None,  # Path to skills directory
    ...
)
```

### Methods

```python
agent.load_skills_metadata(skills_dir: str) -> List[Dict[str, str]]
# Load skill metadata from SKILL.md files

agent.load_full_skill(skill_name: str) -> Optional[str]
# Load complete skill content for programmatic access
```

## Roadmap

Future enhancements planned:

- **Skill Dependencies**: Allow skills to reference other skills
- **Skill Versioning**: Track and manage skill versions
- **Skill Marketplace**: Share skills via Swarms Marketplace
- **Dynamic Loading**: Load skills mid-conversation based on task
- **Skill Analytics**: Track which skills are used most often

## Community

Share your skills and learn from others:

- [Swarms Discord](https://discord.gg/EamjgSaEQf) - #agent-skills channel
- [GitHub Discussions](https://github.com/kyegomez/swarms/discussions)
- [Swarms Marketplace](https://swarms.world) - Coming soon: skill sharing

## License

Agent Skills implementation in Swarms is MIT licensed, compatible with Anthropic's open standard.
