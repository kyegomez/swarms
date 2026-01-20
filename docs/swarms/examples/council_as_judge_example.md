# CouncilAsAJudge: Complete Guide

A comprehensive guide to using the CouncilAsAJudge architecture for multi-dimensional evaluation of task responses and AI-generated content.

## Overview

The **CouncilAsAJudge** system is a sophisticated multi-agent evaluation architecture that assesses task responses across six critical dimensions using specialized judge agents. Each judge independently evaluates a specific dimension, and an aggregator agent synthesizes their findings into a comprehensive technical report with actionable recommendations.

| Feature | Description |
|---------|-------------|
| **Multi-Dimensional Evaluation** | 6 specialized judges evaluate different quality dimensions |
| **Parallel Execution** | All judges evaluate concurrently for maximum efficiency |
| **Comprehensive Reports** | Aggregator synthesizes detailed technical analysis |
| **Specialized Judges** | Each judge focuses on a specific evaluation criterion |
| **Caching Optimization** | LRU cache for frequently used prompts and evaluations |
| **Flexible Configuration** | Customizable models, cache size, and evaluation parameters |

```
Task Response
     │
     ▼
┌────────────────────────────────────────┐
│  Parallel Judge Evaluation             │
├────────────────────────────────────────┤
│  Accuracy Judge     → Analysis         │
│  Helpfulness Judge  → Analysis         │
│  Harmlessness Judge → Analysis         │
│  Coherence Judge    → Analysis         │
│  Conciseness Judge  → Analysis         │
│  Adherence Judge    → Analysis         │
└────────────────────────────────────────┘
     │
     ▼
Aggregator Agent
     │
     ▼
Comprehensive Evaluation Report
```

### Evaluation Dimensions

The system evaluates responses across these six dimensions:

| Dimension | Evaluation Focus |
|-----------|------------------|
| **Accuracy** | Factual correctness, evidence-based claims, source credibility, temporal consistency, technical precision |
| **Helpfulness** | Practical value, solution feasibility, completeness, proactive addressing of follow-ups, examples and applications |
| **Harmlessness** | Safety, ethical considerations, bias detection, offensive content, appropriate disclaimers |
| **Coherence** | Logical flow, information hierarchy, clear transitions, consistent terminology, argument structure |
| **Conciseness** | Communication efficiency, no redundancy, directness, appropriate detail level, focused content |
| **Instruction Adherence** | Requirement coverage, constraint compliance, format matching, scope appropriateness, guideline following |

### When to Use CouncilAsAJudge

**Best For:**
- Evaluating LLM-generated content quality
- Quality assurance for AI responses
- Multi-dimensional content review
- Comparing different AI model outputs
- Educational content assessment
- Technical documentation evaluation

**Not Ideal For:**
- Simple binary yes/no evaluations
- Tasks requiring generation (not evaluation)
- Single-dimension assessments
- Real-time evaluation (use faster single-agent evaluation)

---

## Installation

```bash
pip install swarms
```

---

## Quick Start

### Step 1: Create the Council

```python
from swarms import CouncilAsAJudge

# Create the council judge
council = CouncilAsAJudge(
    name="Quality-Evaluation-Council",
    description="Evaluates response quality across multiple dimensions",
    model_name="gpt-4o-mini",  # Model for judge agents
    aggregation_model_name="gpt-4o-mini",  # Model for aggregator
)
```

### Step 2: Evaluate a Response

```python
# Task with response to evaluate
task_with_response = """
Task: Explain the concept of machine learning to a beginner.

Response: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It works by analyzing large amounts of data to identify patterns and make predictions or decisions. There are three main types: supervised learning (using labeled data), unsupervised learning (finding hidden patterns), and reinforcement learning (learning through trial and error). Machine learning is used in various applications like recommendation systems, image recognition, and natural language processing.
"""

# Run the evaluation
result = council.run(task=task_with_response)

# Print the comprehensive evaluation
print(result)
```

---

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"CouncilAsAJudge"` | Display name of the council |
| `model_name` | `str` | `"gpt-4o-mini"` | Model name for judge agents |
| `aggregation_model_name` | `str` | `"gpt-4o-mini"` | Model name for the aggregator agent |
| `judge_agent_model_name` | `Optional[str]` | `None` | Override model for specific judge agents |
| `output_type` | `str` | `"final"` | Type of output to return |
| `max_loops` | `int` | `1` | Maximum number of loops for agents |
| `random_model_name` | `bool` | `True` | Whether to use random model names for diversity |
| `cache_size` | `int` | `128` | Size of the LRU cache for prompts |

---

## Complete Examples

### Example 1: Basic Evaluation

```python
from swarms import CouncilAsAJudge

# Initialize the council
council = CouncilAsAJudge(
    name="Content-Quality-Evaluator",
    description="Evaluates content quality across multiple dimensions",
    model_name="gpt-4o-mini",
    aggregation_model_name="gpt-4o-mini",
)

# Task with response to evaluate
task_with_response = """
Task: Write documentation for a REST API endpoint that creates a new user.

Response: POST /api/users - Creates a new user account. Send a JSON body with 'email', 'password', and 'name' fields. Returns 201 on success with user object, 400 for validation errors, 409 if email exists. Requires no authentication. Example: {"email": "user@example.com", "password": "secure123", "name": "John Doe"}
"""

# Run the evaluation
evaluation = council.run(task=task_with_response)

# Display results
print("=" * 60)
print("EVALUATION REPORT:")
print("=" * 60)
print(evaluation)
```

### Example 2: Evaluating AI Model Outputs

Compare different AI model outputs to choose the best one:

```python
from swarms import CouncilAsAJudge

# Initialize council with powerful aggregation
council = CouncilAsAJudge(
    name="AI-Model-Comparator",
    model_name="gpt-4o-mini",  # For judge agents
    aggregation_model_name="gpt-4o",  # More powerful for synthesis
    cache_size=256,  # Larger cache for multiple evaluations
)

# Evaluation task
task = "Explain blockchain technology to a non-technical audience."

# Response from Model A (GPT-4)
response_a = f"""
Task: {task}

Response: Blockchain is a digital ledger technology that records transactions across multiple computers in a way that makes it nearly impossible to change or hack. Think of it as a shared notebook that everyone can read, but no one can erase or modify past entries. Each page (block) is linked to the previous one, creating a chain. This technology powers cryptocurrencies like Bitcoin and has applications in supply chain, healthcare, and voting systems.
"""

# Response from Model B (Claude)
response_b = f"""
Task: {task}

Response: Imagine blockchain as a digital record book that's copied across thousands of computers worldwide. Every time a new transaction happens, it's added as a new page to everyone's copy of the book simultaneously. What makes it special is that once a page is written, it's permanently sealed and linked to all previous pages, making it tamper-proof. This creates trust without needing a central authority like a bank. Beyond cryptocurrency, blockchain enables transparent supply chains, secure medical records, and verifiable digital ownership.
"""

# Evaluate both responses
print("Evaluating Model A...")
eval_a = council.run(task=response_a)

print("\nEvaluating Model B...")
eval_b = council.run(task=response_b)

# Compare results
print("\n" + "="*60)
print("MODEL A EVALUATION:")
print("="*60)
print(eval_a)

print("\n" + "="*60)
print("MODEL B EVALUATION:")
print("="*60)
print(eval_b)
```

### Example 3: Evaluating Technical Documentation

```python
from swarms import CouncilAsAJudge

# Council for technical documentation
council = CouncilAsAJudge(
    name="Tech-Doc-Reviewer",
    description="Evaluates technical documentation quality",
    model_name="gpt-4o-mini",
)

# API documentation to evaluate
api_docs = """
Task: Document the POST /api/users endpoint for creating new users.

Response:
# POST /api/users

Creates a new user account in the system.

## Request

### Headers
- `Content-Type: application/json`
- `X-API-Key: <your-api-key>` (required)

### Body
```json
{
  "email": "user@example.com",
  "password": "securePassword123!",
  "name": "John Doe",
  "role": "user" // optional, defaults to "user"
}
```

### Validation Rules
- Email must be valid format and unique
- Password must be 8+ characters with number and special character
- Name is required, 1-100 characters
- Role must be "user" or "admin"

## Responses

### 201 Created
```json
{
  "id": "usr_123abc",
  "email": "user@example.com",
  "name": "John Doe",
  "role": "user",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### 400 Bad Request
Invalid input data. Response includes field-specific error messages.

### 409 Conflict
Email address already registered.

### 401 Unauthorized
Invalid or missing API key.
"""

# Evaluate the documentation
evaluation = council.run(task=api_docs)

print("DOCUMENTATION QUALITY EVALUATION:")
print("="*60)
print(evaluation)
```

---

## Use Cases

### Use Case 1: LLM Output Quality Assurance

```python
from swarms import CouncilAsAJudge, Agent

# Setup: Create an LLM agent that generates content
generator = Agent(
    agent_name="Content-Generator",
    system_prompt="You are a helpful assistant that generates educational content.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Setup: Create evaluation council
evaluator = CouncilAsAJudge(
    name="QA-Council",
    model_name="gpt-4o-mini",
)

# Generate content
topic = "Explain quantum computing"
generated_content = generator.run(topic)

# Format for evaluation
evaluation_input = f"""
Task: {topic}

Response: {generated_content}
"""

# Evaluate the generated content
quality_report = evaluator.run(task=evaluation_input)

# Decision logic based on evaluation
if "critical issues" in quality_report.lower():
    print("Content needs revision")
    # Regenerate or refine
else:
    print("Content approved for publication")
    # Publish content

print("\nQuality Report:")
print(quality_report)
```

### Use Case 2: Educational Content Assessment

```python
from swarms import CouncilAsAJudge

council = CouncilAsAJudge(
    name="Education-Quality-Council",
    description="Evaluates educational content for learners",
    model_name="gpt-4o-mini",
)

# Student's explanation to evaluate
student_explanation = """
Task: Explain the concept of photosynthesis.

Response: Photosynthesis is how plants make food. They use sunlight, water, and carbon dioxide. The leaves have chlorophyll which is green and helps capture sunlight. The plant uses this energy to turn CO2 and water into glucose (sugar) and oxygen. The oxygen is released into the air. This happens in the chloroplasts.
"""

# Evaluate student work
evaluation = council.run(task=student_explanation)

print("STUDENT WORK EVALUATION:")
print(evaluation)

# Use evaluation to provide feedback
# - Identify what student understands well
# - Highlight areas needing improvement
# - Suggest specific topics to review
```

---

## How It Works

1. **Task Submission**: Submit a task containing the response to evaluate
2. **Parallel Evaluation**: Six judge agents evaluate concurrently:
   - Each judge focuses on their specialized dimension
   - Judges provide detailed, technical feedback
   - Specific examples and improvement suggestions included
3. **Aggregation**: Aggregator agent synthesizes all evaluations into:
   - Executive summary of key strengths/weaknesses
   - Cross-dimensional patterns and correlations
   - Prioritized improvement recommendations
   - Comprehensive technical report
4. **Result**: Receive detailed evaluation report with actionable insights

---

## Best Practices

### 1. Provide Clear Context

Always include both the task and the response in your evaluation input:

```python
# Good - Clear separation
task_input = """
Task: [The original task or question]

Response: [The response to evaluate]
"""

# Bad - Ambiguous
task_input = "Evaluate this: [content]"
```

### 2. Use Appropriate Models

Balance cost and quality based on your needs:

```python
# Development/Testing - Faster, cheaper
council = CouncilAsAJudge(
    model_name="gpt-4o-mini",
    aggregation_model_name="gpt-4o-mini",
)

# Production/Critical - Higher quality
council = CouncilAsAJudge(
    model_name="gpt-4o-mini",  # Judges can use lighter model
    aggregation_model_name="gpt-4o",  # Aggregation uses powerful model
)
```

### 3. Optimize Cache for Repeated Evaluations

```python
# Default cache
council = CouncilAsAJudge(cache_size=128)

# Large cache for high-volume evaluation
council = CouncilAsAJudge(cache_size=512)
```

### 4. Extract Actionable Insights

Parse the evaluation report to extract specific improvements:

```python
evaluation = council.run(task=response)

# The evaluation includes:
# - Specific weaknesses by dimension
# - Concrete improvement suggestions
# - Prioritized recommendations
# - Examples of issues

# Use this to:
# 1. Refine prompts
# 2. Improve content
# 3. Train models
# 4. Update guidelines
```

---

## Evaluation Output

The council returns a comprehensive report including:

```markdown
EXECUTIVE SUMMARY
- Key strengths identified
- Critical issues requiring attention
- Overall assessment

DETAILED ANALYSIS
- Cross-dimensional patterns
- Specific examples and implications
- Technical impact assessment

RECOMMENDATIONS
- Prioritized improvement areas
- Specific technical suggestions
- Implementation considerations
```

---

## Related Architectures

| Architecture | When to Use Instead |
|--------------|---------------------|
| **[MajorityVoting](./majority_voting_example.md)** | When you need multiple agents to independently solve a task and build consensus |
| **[LLMCouncil](./llm_council_examples.md)** | When you want agents to rank each other's generated responses |
| **[DebateWithJudge](../examples/debate_quickstart.md)** | When you want two agents to argue opposing viewpoints |

---

## Next Steps

- See [GitHub Examples](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/council_of_judges)
- Learn about [Multi-Agent Evaluation Patterns](../examples/multi_agent_architectures_overview.md)
- Try [MajorityVoting](./majority_voting_example.md) for consensus-based generation
