# CouncilAsAJudge: 3-Step Quickstart Guide

The CouncilAsAJudge architecture enables multi-dimensional evaluation of task responses using specialized judge agents. Each judge evaluates a different dimension (accuracy, helpfulness, harmlessness, coherence, conciseness, instruction adherence), and an aggregator synthesizes their findings into a comprehensive report.

## Overview

| Feature | Description |
|---------|-------------|
| **Multi-Dimensional Evaluation** | 6 specialized judges evaluate different quality dimensions |
| **Parallel Execution** | All judges evaluate concurrently for maximum efficiency |
| **Comprehensive Reports** | Aggregator synthesizes detailed technical analysis |
| **Specialized Judges** | Each judge focuses on a specific evaluation criterion |

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

---

## Step 1: Install and Import

Ensure you have Swarms installed and import the CouncilAsAJudge class:

```bash
pip install swarms
```

```python
from swarms import CouncilAsAJudge
```

---

## Step 2: Create the Council

Initialize the CouncilAsAJudge evaluation system:

```python
# Create the council judge
council = CouncilAsAJudge(
    name="Quality-Evaluation-Council",
    description="Evaluates response quality across multiple dimensions",
    model_name="gpt-4o-mini",  # Model for judge agents
    aggregation_model_name="gpt-4o-mini",  # Model for aggregator
)
```

---

## Step 3: Evaluate a Response

Run the evaluation on a task with a response:

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

## Complete Example

Here's a complete working example:

```python
from swarms import CouncilAsAJudge

# Step 1: Initialize the council
council = CouncilAsAJudge(
    name="Technical-Writing-Evaluator",
    description="Evaluates technical documentation quality",
    model_name="gpt-4o-mini",
)

# Step 2: Prepare the task and response to evaluate
task_with_response = """
Task: Write documentation for a REST API endpoint that creates a new user.

Response: POST /api/users - Creates a new user account. Send a JSON body with 'email', 'password', and 'name' fields. Returns 201 on success with user object, 400 for validation errors, 409 if email exists. Requires no authentication. Example: {"email": "user@example.com", "password": "secure123", "name": "John Doe"}
"""

# Step 3: Run the evaluation
evaluation = council.run(task=task_with_response)

# Display results
print("=" * 60)
print("EVALUATION REPORT:")
print("=" * 60)
print(evaluation)
```

---

## Evaluation Dimensions

The council evaluates responses across these six dimensions:

| Dimension | Focus Area |
|-----------|------------|
| **Accuracy** | Factual correctness, evidence-based claims, source credibility |
| **Helpfulness** | Practical value, solution feasibility, completeness |
| **Harmlessness** | Safety, ethics, bias detection, appropriate disclaimers |
| **Coherence** | Logical flow, structure, clear transitions, argument quality |
| **Conciseness** | Communication efficiency, no redundancy, focused content |
| **Instruction Adherence** | Compliance with requirements, format specifications, scope |

Each judge provides:
- Detailed analysis of their dimension
- Specific examples from the response
- Impact assessment
- Concrete improvement suggestions

---

## Evaluating Multiple Responses

Compare different responses to the same task:

```python
from swarms import CouncilAsAJudge

council = CouncilAsAJudge(model_name="gpt-4o-mini")

# Response A
response_a = """
Task: Explain recursion in programming.

Response: Recursion is when a function calls itself to solve smaller versions of the same problem until reaching a base case that stops the recursion.
"""

# Response B
response_b = """
Task: Explain recursion in programming.

Response: Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller, similar subproblems. Each recursive call works on a simpler version of the problem until reaching a base case - a condition that stops the recursion. For example, calculating factorial(5) recursively would call factorial(4), which calls factorial(3), and so on until factorial(1) returns 1. The results then combine back up the chain. While elegant, recursion uses more memory than iteration due to the call stack, so it's best for naturally recursive problems like tree traversal or divide-and-conquer algorithms.
"""

# Evaluate both
eval_a = council.run(task=response_a)
eval_b = council.run(task=response_b)

print("Response A Evaluation:")
print(eval_a)
print("\n" + "="*60 + "\n")
print("Response B Evaluation:")
print(eval_b)
```

---

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | `"CouncilAsAJudge"` | Display name of the council |
| `model_name` | `"gpt-4o-mini"` | Model for judge agents |
| `aggregation_model_name` | `"gpt-4o-mini"` | Model for aggregator agent |
| `judge_agent_model_name` | `None` | Override model for specific judges |
| `output_type` | `"final"` | Output format type |
| `max_loops` | `1` | Maximum loops for agents |
| `random_model_name` | `True` | Use random models for diversity |
| `cache_size` | `128` | LRU cache size for prompts |

### Advanced Configuration

```python
# Use more powerful model for aggregation
council = CouncilAsAJudge(
    model_name="gpt-4o-mini",  # For individual judges
    aggregation_model_name="gpt-4o",  # More powerful for synthesis
    cache_size=256,  # Larger cache for better performance
)
```

---

## Use Cases

| Domain | Evaluation Purpose |
|--------|-------------------|
| **Content Review** | Evaluate blog posts, articles, or documentation quality |
| **LLM Output Evaluation** | Assess AI-generated content across dimensions |
| **Code Review** | Evaluate code explanations and technical documentation |
| **Customer Support** | Assess quality of support responses |
| **Educational Content** | Evaluate clarity and accuracy of learning materials |

### Example: Evaluating AI Responses

```python
from swarms import CouncilAsAJudge

council = CouncilAsAJudge(name="AI-Response-Evaluator")

# AI-generated response to evaluate
ai_response = """
Task: Explain the benefits of cloud computing for small businesses.

Response: [Your AI-generated content here]
"""

evaluation = council.run(task=ai_response)

# Use evaluation to:
# 1. Identify weaknesses in AI output
# 2. Guide prompt refinement
# 3. Ensure quality before publishing
# 4. Compare different AI models
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

- **Clear Task Formatting**: Clearly separate the task and response in your input
- **Sufficient Context**: Provide enough context for judges to evaluate properly
- **Appropriate Models**: Use more powerful models for aggregation when quality is critical
- **Iterative Improvement**: Use evaluation feedback to refine responses iteratively

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

- [MajorityVoting](./majority_voting_quickstart.md) - Multiple agents vote on best response
- [LLM Council](./llm_council_quickstart.md) - Council members rank each other's responses
- [DebateWithJudge](./debate_quickstart.md) - Two agents debate with judge synthesis

---

## Next Steps

- Explore [CouncilAsAJudge Tutorial](../swarms/examples/council_as_judge_example.md) for advanced examples
- See [GitHub Examples](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/council_of_judges)
- Learn about [Evaluation Frameworks](../swarms/concept/evaluation_frameworks.md)
