# Agent Marketplace Publishing: 3-Step Quickstart Guide

Publish your agents directly to the Swarms Marketplace with minimal configuration. Share your specialized agents with the community and monetize your creations.

## Overview

| Feature | Description |
|---------|-------------|
| **Direct Publishing** | Publish agents with a single flag |
| **Minimal Configuration** | Just add use cases, tags, and capabilities |
| **Automatic Integration** | Seamlessly integrates with marketplace API |
| **Monetization Ready** | Set pricing for your agents |

---

## Step 1: Get Your API Key

Before publishing, you need a Swarms API key:

1. Visit [swarms.world/platform/api-keys](https://swarms.world/platform/api-keys)
2. Create an account or sign in
3. Generate an API key
4. Set the environment variable:

```bash
export SWARMS_API_KEY="your-api-key-here"
```

Or add to your `.env` file:

```
SWARMS_API_KEY=your-api-key-here
```

---

## Step 2: Configure Your Agent

Create an agent with publishing configuration:

```python
from swarms import Agent

# Create your specialized agent
my_agent = Agent(
    agent_name="Market-Analysis-Agent",
    agent_description="Expert market analyst specializing in cryptocurrency and stock analysis",
    model_name="gpt-4o-mini",
    system_prompt="""You are an expert market analyst specializing in:
    - Cryptocurrency market analysis
    - Stock market trends
    - Risk assessment
    - Portfolio recommendations
    
    Provide data-driven insights with confidence levels.""",
    max_loops=1,
    
    # Publishing configuration
    publish_to_marketplace=True,
    
    # Required: Define use cases
    use_cases=[
        {
            "title": "Cryptocurrency Analysis",
            "description": "Analyze crypto market trends and provide investment insights"
        },
        {
            "title": "Stock Screening",
            "description": "Screen stocks based on technical and fundamental criteria"
        },
        {
            "title": "Portfolio Review",
            "description": "Review and optimize investment portfolios"
        }
    ],
    
    # Required: Tags and capabilities
    tags=["finance", "crypto", "stocks", "analysis"],
    capabilities=["market-analysis", "risk-assessment", "portfolio-optimization"]
)
```

---

## Step 3: Run to Publish

Simply run the agent to trigger publishing:

```python
# Running the agent automatically publishes it
result = my_agent.run("Analyze Bitcoin's current market position")

print(result)
print("\n✅ Agent published to marketplace!")
```

---

## Complete Example

Here's a complete working example:

```python
import os
from swarms import Agent

# Ensure API key is set
if not os.getenv("SWARMS_API_KEY"):
    raise ValueError("Please set SWARMS_API_KEY environment variable")

# Step 1: Create a specialized medical analysis agent
medical_agent = Agent(
    agent_name="Blood-Data-Analysis-Agent",
    agent_description="Explains and contextualizes common blood test panels with structured insights",
    model_name="gpt-4o-mini",
    max_loops=1,
    
    system_prompt="""You are a clinical laboratory data analyst assistant focused on hematology and basic metabolic panels.

Your goals:
1) Interpret common blood test panels (CBC, CMP/BMP, lipid panel, HbA1c, thyroid panels)
2) Provide structured findings: out-of-range markers, degree of deviation, clinical significance
3) Identify potential confounders (e.g., hemolysis, fasting status, medications)
4) Suggest safe, non-diagnostic next steps

Reliability and safety:
- This is not medical advice. Do not diagnose or treat.
- Use cautious language with confidence levels (low/medium/high)
- Highlight red-flag combinations that warrant urgent clinical evaluation""",

    # Step 2: Publishing configuration
    publish_to_marketplace=True,
    
    tags=["lab", "hematology", "metabolic", "education"],
    capabilities=[
        "panel-interpretation",
        "risk-flagging", 
        "guideline-citation"
    ],
    
    use_cases=[
        {
            "title": "Blood Analysis",
            "description": "Analyze blood samples and summarize notable findings."
        },
        {
            "title": "Patient Lab Monitoring",
            "description": "Track lab results over time and flag key trends."
        },
        {
            "title": "Pre-surgery Lab Check",
            "description": "Review preoperative labs to highlight risks."
        }
    ],
)

# Step 3: Run the agent (this publishes it to the marketplace)
result = medical_agent.run(
    task="Analyze this blood sample: Hematology and Basic Metabolic Panel"
)

print(result)
```

---

## Required Fields for Publishing

| Field | Type | Description |
|-------|------|-------------|
| `publish_to_marketplace` | `bool` | Set to `True` to enable publishing |
| `use_cases` | `List[Dict]` | List of use case dictionaries with `title` and `description` |
| `tags` | `List[str]` | Keywords for discovery |
| `capabilities` | `List[str]` | Agent capabilities for matching |

### Use Case Format

```python
use_cases = [
    {
        "title": "Use Case Title",
        "description": "Detailed description of what the agent does for this use case"
    },
    # Add more use cases...
]
```

---

## Optional: Programmatic Publishing

You can also publish prompts/agents directly using the utility function:

```python
from swarms.utils.swarms_marketplace_utils import add_prompt_to_marketplace

response = add_prompt_to_marketplace(
    name="My Custom Agent",
    prompt="Your detailed system prompt here...",
    description="What this agent does",
    use_cases=[
        {"title": "Use Case 1", "description": "Description 1"},
        {"title": "Use Case 2", "description": "Description 2"}
    ],
    tags="tag1, tag2, tag3",
    category="research",
    is_free=True,          # Set to False for paid agents
    price_usd=0.0          # Set price if not free
)

print(response)
```

---

## Marketplace Categories

| Category | Description |
|----------|-------------|
| `research` | Research and analysis agents |
| `content` | Content generation agents |
| `coding` | Programming and development agents |
| `finance` | Financial analysis agents |
| `healthcare` | Medical and health-related agents |
| `education` | Educational and tutoring agents |
| `legal` | Legal research and analysis agents |

---

## Best Practices

!!! tip "Publishing Best Practices"
    - **Clear Descriptions**: Write detailed, accurate agent descriptions
    - **Multiple Use Cases**: Provide 3-5 distinct use cases
    - **Relevant Tags**: Use specific, searchable keywords
    - **Test First**: Thoroughly test your agent before publishing
    - **System Prompt Quality**: Ensure your system prompt is well-crafted

!!! warning "Important Notes"
    - `use_cases` is **required** when `publish_to_marketplace=True`
    - Both `tags` and `capabilities` should be provided for discoverability
    - The agent must have a valid `SWARMS_API_KEY` set in the environment

---

## Monetization

To create a paid agent:

```python
from swarms.utils.swarms_marketplace_utils import add_prompt_to_marketplace

response = add_prompt_to_marketplace(
    name="Premium Analysis Agent",
    prompt="Your premium agent prompt...",
    description="Advanced analysis capabilities",
    use_cases=[...],
    tags="premium, advanced",
    category="finance",
    is_free=False,         # Paid agent
    price_usd=9.99         # Price per use
)
```

---

## Next Steps

- Visit [Swarms Marketplace](https://swarms.world) to browse published agents
- Learn about [Marketplace Documentation](../swarms_platform/share_and_discover.md)
- Explore [Monetization Options](../swarms_platform/monetize.md)
- See [API Key Management](../swarms_platform/apikeys.md)

