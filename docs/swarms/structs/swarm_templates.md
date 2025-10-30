# SwarmTemplates

A comprehensive library of pre-built, production-ready swarm templates for common use cases. Each template includes professionally designed agents with optimized prompts and proven orchestration patterns.

## Overview

SwarmTemplates provides instant access to enterprise-grade multi-agent systems that solve real-world business problems. Instead of building agent teams from scratch, you can leverage battle-tested templates that have been optimized for specific use cases.

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Instant Deployment** | Get production-ready swarms in seconds, not hours |
| **Best Practices Built-in** | Templates use proven prompts and orchestration patterns |
| **Customizable** | Easy to customize with your own models and parameters |
| **Production-Tested** | Each template has been tested and optimized for real use |
| **Comprehensive Coverage** | Templates for research, content, code, finance, and more |

## Available Templates

### 1. Research-Analysis-Synthesis

**Use Case:** Academic research, market research, competitive analysis

**Workflow:** Research Specialist → Analysis Expert → Synthesis Specialist

**Best For:**
- Comprehensive topic investigation
- Market research reports
- Competitive analysis
- Academic literature reviews

```python
from swarms import SwarmTemplates

swarm = SwarmTemplates.research_analysis_synthesis(
    model_name="gpt-4o-mini"
)

result = swarm.run("Analyze the impact of AI on healthcare")
print(result)
```

---

### 2. Content Creation Pipeline

**Use Case:** Blog posts, articles, marketing content, documentation

**Workflow:** Content Strategist → Content Writer → Editor → SEO Specialist

**Best For:**
- Blog post creation
- Article writing
- Marketing content
- Technical documentation
- SEO-optimized content

```python
from swarms import SwarmTemplates

swarm = SwarmTemplates.content_creation_pipeline(
    model_name="gpt-4o-mini"
)

result = swarm.run("Create a blog post about AI agents in customer service")
print(result)
```

---

### 3. Code Development Team

**Use Case:** Software development, feature implementation, code review

**Workflow:** Requirements Analyst → Software Developer → QA Engineer → Code Reviewer

**Best For:**
- Feature development
- Code generation
- Technical implementation
- Code quality review
- Bug fixing

```python
from swarms import SwarmTemplates

swarm = SwarmTemplates.code_development_team(
    model_name="gpt-4o-mini"
)

result = swarm.run("Create a REST API for user authentication")
print(result)
```

---

### 4. Financial Analysis Team

**Use Case:** Investment analysis, financial reporting, risk assessment

**Workflow:** Data Collector → Financial Analyst → Risk Assessor → Report Writer

**Best For:**
- Investment analysis
- Financial performance evaluation
- Risk assessment
- Financial reporting
- Due diligence

```python
from swarms import SwarmTemplates

swarm = SwarmTemplates.financial_analysis_team(
    model_name="gpt-4o-mini"
)

result = swarm.run("Analyze Tesla's Q4 2024 financial performance")
print(result)
```

---

### 5. Marketing Campaign Team

**Use Case:** Marketing campaigns, product launches, brand awareness

**Workflow:** Marketing Strategist → Content Creator → Creative Director → Distribution Specialist

**Best For:**
- Campaign planning
- Product launches
- Marketing strategy
- Content marketing
- Multi-channel campaigns

```python
from swarms import SwarmTemplates

swarm = SwarmTemplates.marketing_campaign_team(
    model_name="gpt-4o-mini"
)

result = swarm.run("Create a product launch campaign for an AI fitness app")
print(result)
```

---

### 6. Customer Support Team

**Use Case:** Customer service, technical support, issue resolution

**Workflow:** Support Triage → Technical Support → Customer Success

**Best For:**
- Customer inquiries
- Technical troubleshooting
- Issue resolution
- Customer satisfaction
- Support ticket handling

```python
from swarms import SwarmTemplates

swarm = SwarmTemplates.customer_support_team(
    model_name="gpt-4o-mini"
)

result = swarm.run("Customer reports login issues after password reset")
print(result)
```

---

### 7. Legal Document Review

**Use Case:** Contract review, legal compliance, risk analysis

**Workflow:** Legal Analyst → Compliance Specialist → Risk Assessor → Summary Writer

**Best For:**
- Contract review
- Legal compliance checking
- Risk assessment
- Due diligence
- Legal document analysis

```python
from swarms import SwarmTemplates

swarm = SwarmTemplates.legal_document_review(
    model_name="gpt-4o-mini"
)

result = swarm.run("Review this SaaS service agreement")
print(result)
```

---

### 8. Data Science Pipeline

**Use Case:** Data analysis, ML pipelines, business intelligence

**Workflow:** Data Collector → Data Cleaner → Data Analyst → Visualization Specialist

**Best For:**
- Data analysis projects
- ML pipeline design
- Business intelligence
- Predictive analytics
- Data visualization

```python
from swarms import SwarmTemplates

swarm = SwarmTemplates.data_science_pipeline(
    model_name="gpt-4o-mini"
)

result = swarm.run("Analyze customer churn data and identify key factors")
print(result)
```

---

## Class Reference

### SwarmTemplates

Main class providing access to all swarm templates.

#### Class Methods

##### `list_templates()`

Get a list of all available swarm templates with descriptions.

**Returns:**
- `List[Dict[str, str]]`: List of templates with metadata

**Example:**

```python
from swarms import SwarmTemplates

templates = SwarmTemplates.list_templates()

for template in templates:
    print(f"{template['name']}: {template['description']}")
```

**Output:**

```
research_analysis_synthesis: Research → Analysis → Synthesis workflow
content_creation_pipeline: Planning → Writing → Editing → SEO workflow
code_development_team: Requirements → Development → Testing → Review workflow
...
```

---

##### `get_template_info(template_name: str)`

Get detailed information about a specific template.

**Parameters:**
- `template_name` (str): Name of the template

**Returns:**
- `Dict[str, Any]`: Detailed template information

**Raises:**
- `ValueError`: If template name is not found

**Example:**

```python
from swarms import SwarmTemplates

info = SwarmTemplates.get_template_info("research_analysis_synthesis")

print(f"Description: {info['description']}")
print(f"Use Case: {info['use_case']}")
print(f"Agents: {info['agents']}")
```

---

##### `create(template_name, model_name, max_loops, verbose, custom_params)`

Universal method to create any swarm template.

**Parameters:**
- `template_name` (str): Name of the template to create
- `model_name` (str): LLM model to use for all agents (default: "gpt-4o-mini")
- `max_loops` (int): Maximum loops for agent execution (default: 1)
- `verbose` (bool): Enable verbose logging (default: True)
- `custom_params` (Optional[Dict[str, Any]]): Additional custom parameters

**Returns:**
- Configured swarm workflow (type depends on template)

**Raises:**
- `ValueError`: If template name is not found

**Example:**

```python
from swarms import SwarmTemplates

# Create a template by name
swarm = SwarmTemplates.create(
    template_name="research_analysis_synthesis",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=True
)

result = swarm.run("Analyze quantum computing trends")
print(result)
```

---

##### Template-Specific Methods

Each template also has its own dedicated method:

**Method Signatures:**

```python
@classmethod
def research_analysis_synthesis(
    model_name: str = "gpt-4o-mini",
    max_loops: int = 1,
    verbose: bool = True,
) -> SequentialWorkflow:
    ...

@classmethod
def content_creation_pipeline(
    model_name: str = "gpt-4o-mini",
    max_loops: int = 1,
    verbose: bool = True,
) -> SequentialWorkflow:
    ...

# Similar signatures for all other templates
```

**Common Parameters:**
- `model_name` (str): LLM model to use
- `max_loops` (int): Maximum loops per agent
- `verbose` (bool): Enable verbose logging

**Returns:**
- Configured SequentialWorkflow instance

---

## Usage Patterns

### Pattern 1: Quick Start

Use the universal `create()` method for quick instantiation:

```python
from swarms import SwarmTemplates

swarm = SwarmTemplates.create("content_creation_pipeline")
result = swarm.run("Write a blog post about AI")
```

### Pattern 2: Direct Method Call

Use template-specific methods for type hints and IDE support:

```python
from swarms import SwarmTemplates

swarm = SwarmTemplates.research_analysis_synthesis(
    model_name="gpt-4o",
    verbose=True
)

result = swarm.run("Research renewable energy trends")
```

### Pattern 3: Discovery and Selection

List templates and select based on use case:

```python
from swarms import SwarmTemplates

# List all templates
templates = SwarmTemplates.list_templates()

# Find template for your use case
for template in templates:
    if "financial" in template['use_case'].lower():
        print(f"Found: {template['name']}")
        
        # Get detailed info
        info = SwarmTemplates.get_template_info(template['name'])
        
        # Create and use
        swarm = SwarmTemplates.create(template['name'])
```

### Pattern 4: Custom Model Configuration

Use different models for different needs:

```python
from swarms import SwarmTemplates

# High-quality model for critical analysis
financial_swarm = SwarmTemplates.financial_analysis_team(
    model_name="gpt-4o"  # Premium model
)

# Cost-effective model for routine tasks
content_swarm = SwarmTemplates.content_creation_pipeline(
    model_name="gpt-4o-mini"  # Economy model
)
```

### Pattern 5: Batch Processing

Process multiple tasks with the same template:

```python
from swarms import SwarmTemplates

# Create template once
swarm = SwarmTemplates.customer_support_team()

# Process multiple support tickets
tickets = [
    "Customer can't reset password",
    "Billing question about invoice",
    "Feature request for dark mode"
]

for ticket in tickets:
    result = swarm.run(ticket)
    print(f"Resolution: {result}\n")
```

---

## Customization

### Custom Models

All templates support any LLM model that works with Swarms:

```python
from swarms import SwarmTemplates

# OpenAI models
swarm = SwarmTemplates.create("research_analysis_synthesis", 
    model_name="gpt-4o"
)

# Anthropic models  
swarm = SwarmTemplates.create("code_development_team",
    model_name="claude-sonnet-4-5"
)

# Local models
swarm = SwarmTemplates.create("content_creation_pipeline",
    model_name="ollama/llama3"
)
```

### Max Loops Configuration

Control agent iteration depth:

```python
from swarms import SwarmTemplates

# Single-pass workflow (default)
swarm = SwarmTemplates.research_analysis_synthesis(
    max_loops=1
)

# Multi-iteration with refinement
swarm = SwarmTemplates.code_development_team(
    max_loops=3  # Allows for iteration and improvement
)
```

### Verbose Logging

Control logging verbosity:

```python
from swarms import SwarmTemplates

# Detailed logging (default)
swarm = SwarmTemplates.financial_analysis_team(
    verbose=True
)

# Silent operation
swarm = SwarmTemplates.content_creation_pipeline(
    verbose=False
)
```

---

## Best Practices

### 1. Choose the Right Template

Select templates based on your specific use case:

```python
# For research and analysis
research_swarm = SwarmTemplates.research_analysis_synthesis()

# For content generation
content_swarm = SwarmTemplates.content_creation_pipeline()

# For technical tasks
code_swarm = SwarmTemplates.code_development_team()
```

### 2. Provide Clear, Detailed Tasks

Give agents comprehensive context:

```python
# ❌ Too vague
result = swarm.run("Analyze the company")

# ✅ Clear and detailed
result = swarm.run("""
Analyze Company XYZ's financial performance including:
- Revenue growth trends (last 3 years)
- Profitability metrics
- Cash flow analysis
- Key risk factors
- Competitive positioning
""")
```

### 3. Use Appropriate Models

Match model capability to task complexity:

```python
# Complex analysis → Premium model
financial_swarm = SwarmTemplates.financial_analysis_team(
    model_name="gpt-4o"
)

# Routine tasks → Economy model
support_swarm = SwarmTemplates.customer_support_team(
    model_name="gpt-4o-mini"
)
```

### 4. Handle Outputs Appropriately

Process and validate results:

```python
from swarms import SwarmTemplates

swarm = SwarmTemplates.data_science_pipeline()

result = swarm.run("Analyze customer data")

# Validate output
if result and len(result) > 100:
    # Process valid result
    with open("analysis_report.txt", "w") as f:
        f.write(result)
else:
    # Handle insufficient output
    print("Warning: Analysis incomplete")
```

### 5. Reuse Templates for Similar Tasks

Create once, use multiple times:

```python
from swarms import SwarmTemplates

# Create template instance
content_swarm = SwarmTemplates.content_creation_pipeline()

# Reuse for multiple content pieces
topics = ["AI in Healthcare", "Future of Work", "Climate Tech"]

for topic in topics:
    article = content_swarm.run(f"Write article about {topic}")
    # Save or process article
```

---

## Production Deployment

### Error Handling

Implement robust error handling:

```python
from swarms import SwarmTemplates

def run_swarm_safely(template_name, task):
    """
    Safely execute swarm with error handling
    """
    try:
        swarm = SwarmTemplates.create(
            template_name=template_name,
            model_name="gpt-4o-mini",
            verbose=True
        )
        
        result = swarm.run(task)
        
        if not result:
            raise ValueError("Empty result from swarm")
            
        return {"success": True, "result": result}
        
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

# Usage
response = run_swarm_safely(
    "research_analysis_synthesis",
    "Analyze market trends"
)

if response["success"]:
    print(response["result"])
else:
    print(f"Error: {response['error']}")
```

### Performance Monitoring

Track template performance:

```python
import time
from swarms import SwarmTemplates

def run_with_monitoring(template_name, task):
    """
    Run swarm with performance tracking
    """
    start_time = time.time()
    
    swarm = SwarmTemplates.create(template_name)
    result = swarm.run(task)
    
    execution_time = time.time() - start_time
    
    return {
        "result": result,
        "execution_time": execution_time,
        "template": template_name
    }

# Track performance
metrics = run_with_monitoring(
    "financial_analysis_team",
    "Analyze Q4 results"
)

print(f"Completed in {metrics['execution_time']:.2f}s")
```

### Logging and Auditing

Implement comprehensive logging:

```python
import logging
from swarms import SwarmTemplates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_with_logging(template_name, task, user_id=None):
    """
    Run swarm with comprehensive logging
    """
    logger.info(f"Starting swarm: {template_name}")
    logger.info(f"User: {user_id}, Task: {task[:100]}")
    
    try:
        swarm = SwarmTemplates.create(template_name)
        result = swarm.run(task)
        
        logger.info(f"Swarm completed successfully")
        logger.info(f"Result length: {len(result)} characters")
        
        return result
        
    except Exception as e:
        logger.error(f"Swarm failed: {str(e)}")
        raise

# Usage with logging
result = run_with_logging(
    template_name="research_analysis_synthesis",
    task="Market analysis",
    user_id="user_123"
)
```

---

## FAQ

### Q: Can I modify the agent prompts in templates?

**A:** Templates use pre-optimized prompts. For custom prompts, create agents manually using the `Agent` class.

### Q: Which template should I use for my use case?

**A:** Use `SwarmTemplates.list_templates()` to see all options with use cases, or refer to the template descriptions above.

### Q: Can templates work with local models?

**A:** Yes! Templates support any model compatible with Swarms:

```python
swarm = SwarmTemplates.create(
    "content_creation_pipeline",
    model_name="ollama/llama3"
)
```

### Q: How do I handle long-running tasks?

**A:** Increase `max_loops` for iterative refinement:

```python
swarm = SwarmTemplates.create(
    "code_development_team",
    max_loops=3  # Allow multiple iterations
)
```

### Q: Are templates suitable for production?

**A:** Yes! Templates are production-tested and optimized. Implement proper error handling and monitoring for production deployments.

### Q: Can I mix agents from different templates?

**A:** Templates return workflow objects. For custom agent combinations, use `Agent` and workflow classes directly.

### Q: How much do template operations cost?

**A:** Cost depends on your chosen model and task complexity. Templates use sequential workflows, so expect 3-4 agent calls per task.

---

## Examples

See comprehensive examples in `/examples/swarm_templates_examples.py`

Run examples:

```bash
cd examples
python swarm_templates_examples.py
```

---

## Support

For issues, questions, or feature requests:
- GitHub Issues: https://github.com/kyegomez/swarms/issues
- Discord: https://discord.gg/EamjgSaEQf
- Documentation: https://docs.swarms.world

