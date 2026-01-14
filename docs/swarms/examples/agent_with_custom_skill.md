# Creating Custom Agent Skills

This guide shows you how to create your own custom skills from scratch and use them with Swarms agents.

## Why Create Custom Skills?

Custom skills let you:
- Encode your organization's methodologies
- Ensure consistency across agents
- Share domain expertise via files
- Rapidly iterate without code changes

## Quick Start

Here's the complete process in 3 steps:

### Step 1: Create Directory Structure

```bash
mkdir -p ./my_skills/customer-support
```

### Step 2: Create SKILL.md File

Create `./my_skills/customer-support/SKILL.md`:

```yaml
---
name: customer-support
description: Handle customer support inquiries with empathy, clarity, and efficiency
---

# Customer Support Skill

When responding to customer support inquiries, follow these guidelines:

## Approach
1. **Acknowledge**: Start by acknowledging the customer's issue
2. **Clarify**: Ask questions if anything is unclear
3. **Solve**: Provide clear, step-by-step solutions
4. **Follow-up**: Offer additional help if needed

## Tone
- Professional yet friendly
- Patient and understanding
- Solution-oriented
- Clear and concise

## Example Response Structure
"Thank you for reaching out. I understand [restate issue]. Let me help you with that.

Here's what you can do:
1. [Step one]
2. [Step two]
3. [Step three]

Is there anything else I can help you with?"
```

### Step 3: Use Your Skill

```python
from swarms import Agent

agent = Agent(
    agent_name="Support Agent",
    model_name="gpt-4o",
    skills_dir="./my_skills",
    max_loops=1
)

# Test it
response = agent.run(
    "Customer says: I was charged twice for my subscription!"
)
print(response)
```

Done! Your agent now follows your custom support methodology.

## Complete Working Example

Save this as `create_custom_skill.py`:

```python
"""
Custom Agent Skills Example

This example creates a custom customer-support skill from scratch
and uses it with an agent.
"""

import os
from swarms import Agent

# Step 1: Create skills directory structure
os.makedirs("./my_custom_skills/customer-support", exist_ok=True)

# Step 2: Create the SKILL.md file
skill_content = """---
name: customer-support
description: Handle customer support inquiries with empathy, clarity, and efficiency
---

# Customer Support Skill

When responding to customer support inquiries, follow these guidelines:

## Approach
1. **Acknowledge**: Start by acknowledging the customer's issue
2. **Clarify**: Ask questions if anything is unclear
3. **Solve**: Provide clear, step-by-step solutions
4. **Follow-up**: Offer additional help if needed

## Tone
- Professional yet friendly
- Patient and understanding
- Solution-oriented
- Clear and concise

## Response Template
"Thank you for reaching out. I understand [restate issue]. Let me help you with that.

Here's what you can do:
1. [Step one]
2. [Step two]
3. [Step three]

Is there anything else I can help you with?"

## Common Scenarios

### Billing Issues
- Acknowledge concern about charges
- Verify account details
- Explain what happened
- Provide immediate resolution
- Confirm satisfaction

### Technical Problems
- Express empathy for inconvenience
- Ask for specific error details
- Provide troubleshooting steps
- Offer alternative solutions
- Schedule follow-up if needed
"""

# Write the skill file
with open("./my_custom_skills/customer-support/SKILL.md", "w") as f:
    f.write(skill_content)

print("✓ Created custom skill at: ./my_custom_skills/customer-support/SKILL.md\n")

# Step 3: Create agent with your custom skill
agent = Agent(
    agent_name="Support Agent",
    model_name="gpt-4o",
    max_loops=1,
    skills_dir="./my_custom_skills",
)

print("=" * 70)
print("Testing Custom Customer Support Skill")
print("=" * 70)

# Test Case 1: Billing Issue
print("\n1. Billing Issue:\n")
response = agent.run(
    "Customer complaint: I was charged twice for my subscription but only "
    "received one confirmation email."
)
print(response)

# Test Case 2: Technical Problem
print("\n" + "=" * 70)
print("\n2. Technical Problem:\n")
response = agent.run(
    "Customer says: The app keeps crashing when I try to upload files."
)
print(response)

print("\n" + "=" * 70)
print("\n✓ Custom skill working! Check how the responses follow your guidelines.")

# Cleanup (optional)
# import shutil
# shutil.rmtree("./my_custom_skills")
```

Run it:
```bash
python3 create_custom_skill.py
```

## Skill Template

Use this template for new skills:

```yaml
---
name: your-skill-name
description: One-line description of what this skill does and when to use it
---

# Your Skill Name

Brief introduction explaining the purpose and scope of this skill.

## When to Use This Skill
- Use case 1
- Use case 2
- Use case 3

## Core Methodology

### Step 1: [First Step Name]
- What to do
- Key considerations
- Expected outcomes

### Step 2: [Second Step Name]
- What to do
- Key considerations
- Expected outcomes

## Guidelines
- Important guideline 1
- Important guideline 2
- Important guideline 3

## Best Practices
- Best practice 1
- Best practice 2
- Best practice 3

## Common Pitfalls to Avoid
- Pitfall 1
- Pitfall 2
- Pitfall 3

## Examples

### Example 1: [Scenario Name]
**Input**: [Description of input]
**Expected Output**: [Description of expected output]

### Example 2: [Scenario Name]
**Input**: [Description of input]
**Expected Output**: [Description of expected output]

## Quality Checklist
- [ ] Item 1
- [ ] Item 2
- [ ] Item 3
```

## Real-World Examples

### 1. Legal Document Review

```yaml
---
name: legal-review
description: Review legal documents for compliance and risk assessment
---

# Legal Document Review Skill

## Review Framework

### 1. Initial Assessment
- Document type identification
- Jurisdiction verification
- Key parties identification

### 2. Compliance Check
- Regulatory requirements
- Industry standards
- Internal policies

### 3. Risk Analysis
- Identify potential liabilities
- Flag ambiguous terms
- Note missing clauses

### 4. Recommendations
- Required modifications
- Optional improvements
- Escalation criteria

## Output Format
Provide structured findings with:
- Summary of findings
- Compliance status
- Risk level (Low/Medium/High)
- Recommended actions
```

### 2. Code Architecture Review

```yaml
---
name: architecture-review
description: Evaluate software architecture decisions and design patterns
---

# Architecture Review Skill

## Review Criteria

### 1. Scalability
- Can it handle 10x growth?
- Horizontal vs vertical scaling
- Database design

### 2. Maintainability
- Code organization
- Documentation quality
- Test coverage

### 3. Performance
- Response times
- Resource utilization
- Bottlenecks

### 4. Security
- Authentication/authorization
- Data encryption
- Input validation

## Rating System
Use 1-5 scale for each criteria:
1. Major concerns
2. Significant issues
3. Acceptable
4. Good
5. Excellent
```

### 3. Research Paper Analysis

```yaml
---
name: research-analysis
description: Systematically analyze academic research papers
---

# Research Paper Analysis Skill

## Analysis Framework

### 1. Paper Overview
- Research question
- Methodology
- Key findings

### 2. Methodology Evaluation
- Study design appropriateness
- Sample size adequacy
- Statistical methods

### 3. Results Assessment
- Clarity of findings
- Statistical significance
- Practical significance

### 4. Critical Evaluation
- Limitations
- Biases
- Alternative explanations

### 5. Implications
- Theoretical contributions
- Practical applications
- Future research directions
```

## Tips for Writing Effective Skills

### ✅ DO

**Be Specific**
```yaml
# Good
"Calculate EBITDA margin using: (EBITDA / Revenue) × 100"

# Bad
"Calculate relevant metrics"
```

**Include Examples**
```yaml
# Good
"Example: For revenue of $1M and EBITDA of $300K:
EBITDA Margin = ($300K / $1M) × 100 = 30%"

# Bad
"Show the calculation"
```

**Use Structure**
```yaml
# Good
## Step 1: Data Gathering
- Income statement
- Balance sheet
- Cash flow statement

## Step 2: Ratio Calculation
...

# Bad
First get data then calculate ratios and make sure to check everything...
```

### ❌ DON'T

**Be Vague**
```yaml
# Bad
"Do a good analysis"

# Good
"Perform DCF analysis using the following steps..."
```

**Overload with Information**
```yaml
# Bad
[3 pages of dense text]

# Good
[Organized sections with bullets and clear steps]
```

**Forget the Context**
```yaml
# Bad
"Use the formula"  # What formula?

# Good
"Use the DCF formula: PV = CF / (1 + r)^n where..."
```

## Testing Your Skill

After creating a skill, test it thoroughly:

```python
# Test multiple scenarios
test_cases = [
    "Simple case",
    "Complex case",
    "Edge case",
    "Error case"
]

for test in test_cases:
    print(f"\nTest: {test}")
    response = agent.run(test)
    print(response)
```

## Organizing Multiple Skills

```
skills/
├── domain/
│   ├── finance/
│   │   └── SKILL.md
│   └── legal/
│       └── SKILL.md
├── process/
│   ├── review/
│   │   └── SKILL.md
│   └── analysis/
│       └── SKILL.md
└── communication/
    ├── support/
    │   └── SKILL.md
    └── marketing/
        └── SKILL.md
```

## Version Control

Skills are just files - track them with git:

```bash
# Initialize repo
git init

# Add skills
git add skills/
git commit -m "Add customer support skill"

# Update skill
vim skills/customer-support/SKILL.md
git commit -am "Update support response template"

# Tag versions
git tag v1.0.0
```

## Sharing Skills

Skills are portable! Share them:

1. **Via Git**:
   ```bash
   git clone https://github.com/your-org/skills.git
   ```

2. **Via Files**:
   ```bash
   zip -r customer-support-skill.zip customer-support/
   ```

3. **Via Docs**:
   Document in your team wiki with example usage

## Next Steps

- [Use multiple skills together](/swarms/examples/agent_with_multiple_skills/)
- [Explore advanced patterns](/swarms/agents/agent_skills/)
- [See more examples](https://github.com/kyegomez/swarms/tree/master/example_skills)

## Resources

- [Agent Skills Documentation](/swarms/agents/agent_skills/)
- [SKILL.md Format Specification](/swarms/agents/agent_skills/#skillmd-format)
- [Anthropic Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
