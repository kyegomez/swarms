"""
Custom Agent Skills Example

This example shows how to create your own custom skill and use it with an agent.

Steps:
1. Create a skills directory
2. Create a skill folder with a SKILL.md file
3. Point your agent to the skills directory
"""

import os
from swarms import Agent

# Step 1: Create skills directory structure
os.makedirs("./my_custom_skills/customer-support", exist_ok=True)

# Step 2: Create a SKILL.md file
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

## Example Response Structure
"Thank you for reaching out. I understand [restate issue]. Let me help you with that.

Here's what you can do:
1. [Step one]
2. [Step two]
3. [Step three]

Is there anything else I can help you with?"
"""

# Write the skill file
with open("./my_custom_skills/customer-support/SKILL.md", "w") as f:
    f.write(skill_content)

# Step 3: Create agent with your custom skill
agent = Agent(
    agent_name="Support Agent",
    model_name="gpt-4o",
    max_loops=1,
    skills_dir="./my_custom_skills",
)

# Test the agent - it will automatically follow the customer-support skill
response = agent.run(
    "Customer complaint: I was charged twice for my subscription but only received one confirmation email."
)

print(response)

# Cleanup (optional)
# import shutil
# shutil.rmtree("./my_custom_skills")
