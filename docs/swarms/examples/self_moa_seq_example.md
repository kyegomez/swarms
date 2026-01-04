# SelfMoASeq Example

The SelfMoASeq (Self-Mixture of Agents Sequential) is an ensemble method that generates multiple outputs from a single high-performing model and aggregates them sequentially using a sliding window approach. This addresses context length constraints while maintaining the effectiveness of in-model diversity.

## How It Works

1. **Sample Generation**: Generates multiple diverse samples from a proposer model
2. **Sliding Window Aggregation**: Processes samples in windows, aggregating progressively
3. **Sequential Synthesis**: Uses a sliding window to combine outputs while managing context
4. **Bias Toward Best**: Maintains the best output so far and uses it to guide aggregation

This architecture is ideal for:
- Tasks requiring high-quality outputs with context constraints
- Generating diverse perspectives and aggregating them
- Complex problem-solving with limited context windows
- Quality improvement through ensemble methods

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
```

## Step-by-Step Example

### Step 1: Import Required Modules

```python
from swarms import SelfMoASeq
```

### Step 2: Create SelfMoASeq

```python
self_moa = SelfMoASeq(
    model_name="gpt-4o-mini",
    temperature=0.7,
    window_size=6,
    reserved_slots=3,
    num_samples=30,
    max_loops=10,
    verbose=True,
)
```

### Step 3: Run a Task

```python
task = "Write a comprehensive analysis of the benefits and challenges of renewable energy"

result = self_moa.run(task=task)

print(result)
```

## Understanding the Parameters

- **model_name**: The model to use for generation
- **temperature**: Controls randomness (0.0-2.0)
- **window_size**: Number of samples to process in each aggregation window
- **reserved_slots**: Number of slots reserved for the best output in each window
- **num_samples**: Total number of samples to generate
- **max_loops**: Maximum number of aggregation iterations
- **verbose**: Whether to show detailed progress

## Output Structure

The result is a dictionary containing:

```python
{
    "final_output": "The synthesized best output",
    "all_samples": ["sample1", "sample2", ...],
    "aggregation_steps": 5,
    "metrics": {
        "total_samples_generated": 30,
        "total_aggregations": 5,
        "execution_time_seconds": 45.2
    },
    "task": "Your original task",
    "timestamp": "2024-01-01T12:00:00"
}
```

## Custom Configuration

### Adjust Sample Count

```python
self_moa = SelfMoASeq(
    num_samples=50,  # Generate more samples for better quality
    window_size=8,
    reserved_slots=3,
)
```

### Control Aggregation Window

```python
self_moa = SelfMoASeq(
    window_size=10,  # Larger window for more context
    reserved_slots=4,  # More slots for best output
    num_samples=40,
)
```

### Use Different Models

```python
self_moa = SelfMoASeq(
    proposer_model_name="gpt-4o",  # Better model for generation
    aggregator_model_name="gpt-4o-mini",  # Efficient model for aggregation
    num_samples=30,
)
```

## Example Use Cases

### Complex Analysis

```python
self_moa = SelfMoASeq(
    num_samples=30,
    window_size=6,
    max_loops=10,
)

result = self_moa.run("""
Analyze the impact of artificial intelligence on:
1. Employment and job markets
2. Education systems
3. Healthcare delivery
4. Economic growth

Provide a comprehensive analysis with multiple perspectives.
""")

print(result)
```

### Creative Writing

```python
self_moa = SelfMoASeq(
    temperature=0.8,  # Higher temperature for creativity
    num_samples=25,
    window_size=5,
)

result = self_moa.run("Write a short story about a future where AI and humans collaborate to solve climate change")

print(result)
```

### Technical Documentation

```python
self_moa = SelfMoASeq(
    temperature=0.3,  # Lower temperature for precision
    num_samples=20,
    window_size=6,
)

result = self_moa.run("""
Write comprehensive documentation for a REST API that includes:
- Authentication methods
- Endpoint descriptions
- Request/response examples
- Error handling
""")

print(result)
```

## Accessing Metrics

```python
result = self_moa.run("Your task here")

# Access execution metrics
metrics = result["metrics"]
print(f"Samples generated: {metrics['total_samples_generated']}")
print(f"Aggregations: {metrics['total_aggregations']}")
print(f"Execution time: {metrics['execution_time_seconds']:.2f}s")

# Access the final output
final_output = result["final_output"]
print(final_output)
```

## Retry Configuration

SelfMoASeq includes automatic retry logic:

```python
self_moa = SelfMoASeq(
    max_retries=5,  # Number of retry attempts
    retry_delay=2.0,  # Initial delay between retries
    retry_backoff_multiplier=2.0,  # Exponential backoff
    retry_max_delay=60.0,  # Maximum delay
)
```

## Support and Community

If you're facing issues or want to learn more, check out the following resources:

| Platform | Link | Description |
|----------|------|-------------|
| 📚 Documentation | [docs.swarms.world](https://docs.swarms.world) | Official documentation and guides |
| 💬 Discord | [Join Discord](https://discord.gg/EamjgSaEQf) | Live chat and community support |
| 🐦 Twitter | [@swarms_corp](https://x.com/swarms_corp) | Latest news and announcements |

