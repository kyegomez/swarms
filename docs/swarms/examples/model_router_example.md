# ModelRouter Example

The ModelRouter is an intelligent routing system that automatically selects the best AI model for a given task. It analyzes task requirements and recommends the optimal model and provider combination, then executes the task using the selected model.

## How It Works

1. **Task Analysis**: Analyzes the task to understand requirements
2. **Model Selection**: Recommends the best model based on task characteristics
3. **Automatic Execution**: Runs the task using the selected model
4. **Optimization**: Considers factors like complexity, speed, cost, and capabilities

This architecture is perfect for:
- Optimizing model selection for different tasks
- Cost-effective AI operations
- Automatic model routing based on task requirements
- Multi-model orchestration

## Installation

Install the swarms package using pip:

```bash
pip install -U swarms
```

## Basic Setup

1. First, set up your environment variables:

```python
WORKSPACE_DIR="agent_workspace"
OPENAI_API_KEY="your-api-key"
# Optional: Set other provider API keys
ANTHROPIC_API_KEY="your-anthropic-key"
GOOGLE_API_KEY="your-google-key"
```

## Step-by-Step Example

### Step 1: Import Required Modules

```python
from swarms import ModelRouter
```

### Step 2: Create the ModelRouter

```python
router = ModelRouter(
    max_tokens=4000,
    temperature=0.5,
    max_workers=10,
)
```

### Step 3: Run a Task

The router will automatically:
1. Analyze the task
2. Select the best model
3. Execute the task
4. Return the result

```python
task = "Write a creative short story about a robot learning to paint"

result = router.run(task=task)

print(result)
```

## Understanding Model Selection

The ModelRouter considers various factors:

- **Task Complexity**: Simple tasks → faster/cheaper models
- **Reasoning Needs**: Complex reasoning → specialized models
- **Creativity**: Creative tasks → models with higher temperature
- **Speed Requirements**: Fast responses → optimized models
- **Cost Efficiency**: Balance between quality and cost

## Example Tasks

### Simple Query

```python
router = ModelRouter()

result = router.run("What is the capital of France?")
# Likely selects: gpt-4o-mini or gpt-3.5-turbo (fast, cost-effective)
```

### Complex Reasoning

```python
router = ModelRouter()

result = router.run("""
Solve this logic puzzle:
- Alice, Bob, and Charlie are sitting in a row
- Alice is not next to Bob
- Charlie is at one end
- Who is sitting where?
""")
# Likely selects: deepseek-reasoner or gpt-4-turbo (reasoning-focused)
```

### Creative Writing

```python
router = ModelRouter()

result = router.run("Write a poem about artificial intelligence")
# Likely selects: claude-3-5-sonnet or gpt-4 (creative capabilities)
```

### Technical Analysis

```python
router = ModelRouter()

result = router.run("""
Analyze the time complexity of this algorithm:
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""")
# Likely selects: claude-3-opus or gpt-4-turbo (technical analysis)
```

## Custom Configuration

### Adjust Temperature

```python
router = ModelRouter(
    temperature=0.7,  # Higher for more creative outputs
    max_tokens=2000,
)
```

### Set Max Tokens

```python
router = ModelRouter(
    max_tokens=8000,  # For longer outputs
    temperature=0.5,
)
```

### Custom System Prompt

```python
from swarms import ModelRouter

custom_prompt = """
You are an expert model router. Consider these factors:
1. Task complexity
2. Required reasoning depth
3. Creative needs
4. Speed requirements
5. Cost efficiency
"""

router = ModelRouter(
    system_prompt=custom_prompt,
    max_tokens=4000,
)
```

## Batch Processing

Process multiple tasks:

```python
router = ModelRouter()

tasks = [
    "Summarize this article: [article text]",
    "Translate to Spanish: Hello, how are you?",
    "Solve: 2x + 5 = 15",
]

results = router.batch_run(tasks)

print(results)
```

## Concurrent Execution

Run multiple tasks concurrently:

```python
router = ModelRouter(max_workers=5)

tasks = [
    "Task 1: Analyze market trends",
    "Task 2: Generate report summary",
    "Task 3: Create data visualization description",
]

results = router.concurrent_run(tasks)
```

## Single Step Execution

If you want to see the model selection process:

```python
router = ModelRouter()

# This will show the selected model, provider, and rationale
result = router.step("Your task here")
```

The output will display:
- Selected Model
- Provider
- Task (optimized for the model)
- Rationale (why this model was chosen)
- Max Tokens
- Temperature
- System Prompt

## Iterative Refinement

Run multiple iterations for improvement:

```python
router = ModelRouter(max_loops=3)

# Will run up to 3 iterations, stopping if output stabilizes
result = router.run("Improve this code: [code snippet]")
```

## Supported Models

The ModelRouter can route to various models including:
- GPT-4, GPT-3.5 (OpenAI)
- Claude 3.5 Sonnet, Claude 3 Opus (Anthropic)
- Gemini Pro (Google)
- DeepSeek Reasoner (DeepSeek)
- Mistral Large (Mistral)
- And more...

## Support and Community

If you're facing issues or want to learn more, check out the following resources:

| Platform | Link | Description |
|----------|------|-------------|
| 📚 Documentation | [docs.swarms.world](https://docs.swarms.world) | Official documentation and guides |
| 💬 Discord | [Join Discord](https://discord.gg/EamjgSaEQf) | Live chat and community support |
| 🐦 Twitter | [@swarms_corp](https://x.com/swarms_corp) | Latest news and announcements |

