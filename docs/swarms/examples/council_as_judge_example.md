# CouncilAsAJudge: Practical Tutorial

A comprehensive guide to using the CouncilAsAJudge architecture for multi-dimensional evaluation of task responses and AI-generated content.

## Overview

The **CouncilAsAJudge** system is a sophisticated multi-agent evaluation architecture that assesses task responses across six critical dimensions using specialized judge agents. Each judge independently evaluates a specific dimension, and an aggregator agent synthesizes their findings into a comprehensive technical report with actionable recommendations.

| Feature | Description |
|---------|-------------|
| **Multi-Dimensional Analysis** | Six specialized judges evaluate different quality dimensions |
| **Parallel Execution** | All judges run concurrently using ThreadPoolExecutor for maximum efficiency |
| **Expert Judges** | Each judge specializes in one evaluation dimension with detailed rubrics |
| **Comprehensive Synthesis** | Aggregator creates executive summary, detailed analysis, and recommendations |
| **Caching Optimization** | LRU cache for frequently used prompts and evaluations |
| **Flexible Configuration** | Customizable models, cache size, and evaluation parameters |

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
pip install -U swarms
```

---

## Basic Example

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
Task: Explain the concept of machine learning to a beginner.

Response: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It works by analyzing large amounts of data to identify patterns and make predictions or decisions. There are three main types: supervised learning (using labeled data), unsupervised learning (finding hidden patterns), and reinforcement learning (learning through trial and error). Machine learning is used in various applications like recommendation systems, image recognition, and natural language processing.
"""

# Run the evaluation
evaluation_report = council.run(task=task_with_response)

# Display the report
print(evaluation_report)
```

### Output Structure

The output includes:
1. **Individual Judge Analyses**: Detailed feedback from each of the 6 judges
2. **Executive Summary**: Key strengths, critical issues, overall assessment
3. **Detailed Analysis**: Cross-dimensional patterns, specific examples, technical impact
4. **Recommendations**: Prioritized improvements, technical suggestions, implementation considerations

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

## Advanced Example 1: Evaluating AI Model Outputs

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

---

## Advanced Example 2: Evaluating Technical Documentation

Assess quality of technical documentation:

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

## Example Request
```bash
curl -X POST https://api.example.com/api/users \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: your_api_key_here" \\
  -d '{
    "email": "john@example.com",
    "password": "SecurePass123!",
    "name": "John Doe"
  }'
```

## Notes
- Passwords are hashed using bcrypt before storage
- Email verification is sent asynchronously
- Rate limit: 10 requests per minute per API key
"""

# Evaluate the documentation
evaluation = council.run(task=api_docs)

print("DOCUMENTATION QUALITY EVALUATION:")
print("="*60)
print(evaluation)
```

---

## Advanced Example 3: Batch Evaluation

Evaluate multiple responses efficiently:

```python
from swarms import CouncilAsAJudge

council = CouncilAsAJudge(
    name="Batch-Evaluator",
    model_name="gpt-4o-mini",
)

# Multiple responses to evaluate
responses = [
    """
    Task: What is recursion?
    Response: Recursion is when a function calls itself.
    """,
    """
    Task: What is recursion?
    Response: Recursion is a programming technique where a function calls itself to solve a problem by breaking it into smaller subproblems. It requires a base case to stop the recursion and prevent infinite loops.
    """,
    """
    Task: What is recursion?
    Response: Recursion is a powerful programming concept where a function calls itself to solve complex problems by dividing them into simpler instances. Key components include: 1) Base case - stopping condition, 2) Recursive case - self-referential call with modified parameters, 3) Progress toward base case. Example: factorial(n) = n * factorial(n-1) with base case factorial(0) = 1. Common in tree traversal, sorting algorithms, and mathematical computations.
    """,
]

# Evaluate each response
results = []
for i, response in enumerate(responses, 1):
    print(f"\nEvaluating Response {i}...")
    evaluation = council.run(task=response)
    results.append(evaluation)
    print(f"Response {i} evaluated.")

# Display comparative summary
print("\n" + "="*60)
print("COMPARATIVE EVALUATION RESULTS:")
print("="*60)
for i, result in enumerate(results, 1):
    print(f"\n### Response {i} ###")
    print(result)
    print("-"*60)
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

### Use Case 3: Customer Support Response Quality

```python
from swarms import CouncilAsAJudge

# Council for customer support evaluation
council = CouncilAsAJudge(
    name="Support-QA-Council",
    description="Evaluates customer support response quality",
    model_name="gpt-4o-mini",
)

# Support response to evaluate
support_interaction = """
Task: Customer asks - "My order #12345 hasn't arrived yet. It was supposed to be here 3 days ago. Can you help?"

Response: I apologize for the delay with your order #12345. I've checked your tracking information and see that your package is currently in transit at the regional distribution center. Due to unexpected weather conditions in your area, deliveries are experiencing delays of 2-3 business days. Your package should arrive within the next 2 days. I've applied a $10 credit to your account for the inconvenience. You can track your order in real-time at [tracking link]. Is there anything else I can help you with?
"""

# Evaluate response quality
evaluation = council.run(task=support_interaction)

# Use for:
# - Training support agents
# - Quality assurance
# - Identifying best practices
# - Improving response templates

print("SUPPORT RESPONSE EVALUATION:")
print(evaluation)
```

### Use Case 4: Code Documentation Review

```python
from swarms import CouncilAsAJudge

council = CouncilAsAJudge(
    name="Code-Doc-Reviewer",
    model_name="gpt-4o-mini",
)

# Code documentation to evaluate
code_docs = """
Task: Document the user authentication function.

Response:
```python
def authenticate_user(username: str, password: str) -> Optional[User]:
    '''
    Authenticates a user with username and password.

    Args:
        username (str): The user's username
        password (str): The user's password

    Returns:
        Optional[User]: User object if authentication succeeds, None otherwise

    Raises:
        ValueError: If username or password is empty
        DatabaseError: If database connection fails

    Example:
        >>> user = authenticate_user("john_doe", "secret123")
        >>> if user:
        ...     print(f"Welcome {user.name}!")
        ... else:
        ...     print("Invalid credentials")

    Note:
        Password is hashed using bcrypt before comparison.
        Failed attempts are logged for security monitoring.
    '''
    if not username or not password:
        raise ValueError("Username and password required")

    user = db.get_user(username)
    if user and verify_password(password, user.password_hash):
        return user
    return None
```
"""

evaluation = council.run(task=code_docs)
print("CODE DOCUMENTATION EVALUATION:")
print(evaluation)
```

### Use Case 5: Prompt Engineering Evaluation

```python
from swarms import CouncilAsAJudge, Agent

# Council for evaluating prompts
council = CouncilAsAJudge(
    name="Prompt-Evaluator",
    model_name="gpt-4o-mini",
)

# Test different prompts
prompt_v1 = "Write a summary of machine learning."

prompt_v2 = """
Write a comprehensive summary of machine learning for business executives.

Requirements:
- 200-250 words
- Focus on business value and ROI
- Include 2-3 real-world examples
- Avoid technical jargon
- Highlight key trends for 2024
"""

# Generate responses with both prompts
agent = Agent(model_name="gpt-4o-mini", max_loops=1)

response_v1 = agent.run(prompt_v1)
response_v2 = agent.run(prompt_v2)

# Evaluate both
eval_v1 = council.run(task=f"Task: {prompt_v1}\n\nResponse: {response_v1}")
eval_v2 = council.run(task=f"Task: {prompt_v2}\n\nResponse: {response_v2}")

# Compare to improve prompt engineering
print("PROMPT V1 EVALUATION:")
print(eval_v1)
print("\n" + "="*60 + "\n")
print("PROMPT V2 EVALUATION:")
print(eval_v2)
```

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

## Common Patterns

### Pattern 1: Iterative Improvement Loop

```python
from swarms import CouncilAsAJudge, Agent

generator = Agent(model_name="gpt-4o-mini", max_loops=1)
evaluator = CouncilAsAJudge(model_name="gpt-4o-mini")

task = "Explain neural networks"
max_loops = 3

for iteration in range(max_loops):
    # Generate content
    response = generator.run(task)

    # Evaluate
    eval_input = f"Task: {task}\n\nResponse: {response}"
    evaluation = evaluator.run(task=eval_input)

    # Check if quality is sufficient
    if "critical issues" not in evaluation.lower():
        print(f"Success after {iteration + 1} iterations!")
        break

    # Refine task based on evaluation feedback
    task = f"{task}\n\nPrevious attempt had issues. Improve on: {evaluation[:200]}"
```

### Pattern 2: A/B Testing Content

```python
from swarms import CouncilAsAJudge

council = CouncilAsAJudge(model_name="gpt-4o-mini")

# Evaluate different versions
versions = {
    "Version A": "Content A...",
    "Version B": "Content B...",
    "Version C": "Content C...",
}

evaluations = {}
for version, content in versions.items():
    eval_input = f"Task: Create engaging content\n\nResponse: {content}"
    evaluations[version] = council.run(task=eval_input)

# Compare and choose best version
for version, evaluation in evaluations.items():
    print(f"\n{version} Evaluation:")
    print(evaluation)
```

---

## Related Architectures

| Architecture | When to Use Instead |
|--------------|---------------------|
| **[MajorityVoting](./majority_voting_example.md)** | When you need multiple agents to independently solve a task and build consensus |
| **[LLMCouncil](../examples/llm_council_examples.md)** | When you want agents to rank each other's generated responses |
| **[DebateWithJudge](../examples/debate_quickstart.md)** | When you want two agents to argue opposing viewpoints |

---

## API Reference

### CouncilAsAJudge Class

```python
class CouncilAsAJudge:
    def __init__(
        self,
        id: str = swarm_id(),
        name: str = "CouncilAsAJudge",
        description: str = "Evaluates task responses across multiple dimensions",
        model_name: str = "gpt-4o-mini",
        output_type: str = "final",
        cache_size: int = 128,
        random_model_name: bool = True,
        max_loops: int = 1,
        aggregation_model_name: str = "gpt-4o-mini",
        judge_agent_model_name: Optional[str] = None,
    )

    def run(self, task: str) -> str:
        """
        Run the evaluation process across all dimensions.

        Args:
            task: Task containing the response to evaluate

        Returns:
            Comprehensive evaluation report
        """
```

### Evaluation Dimensions

```python
EVAL_DIMENSIONS = {
    "accuracy": "Factual accuracy assessment...",
    "helpfulness": "Practical value evaluation...",
    "harmlessness": "Safety and ethics assessment...",
    "coherence": "Structural integrity analysis...",
    "conciseness": "Communication efficiency evaluation...",
    "instruction_adherence": "Requirement compliance assessment...",
}
```

---

## Troubleshooting

### Issue: Evaluation too slow

**Solution**:
- Reduce `cache_size` if memory constrained
- Use `gpt-4o-mini` for all agents
- Ensure parallel execution is working (check max_workers)

### Issue: Inconsistent evaluations

**Solution**:
- Set `random_model_name=False` for deterministic results
- Use the same model for all judges
- Increase `cache_size` for better caching

### Issue: Evaluation too generic

**Solution**:
- Provide more context in the task input
- Include specific requirements in the task description
- Use a more powerful model for aggregation

---

## Next Steps

- Explore [CouncilAsAJudge Quickstart](../../examples/council_as_judge_quickstart.md)
- See [GitHub Examples](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/council_of_judges)
- Learn about [Multi-Agent Evaluation Patterns](../../examples/multi_agent_architectures_overview.md)
- Try [MajorityVoting](./majority_voting_example.md) for consensus-based generation
