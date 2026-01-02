# MALT (Multi-Agent Learning Task) Example

MALT is a sophisticated multi-agent architecture designed for complex problem-solving tasks, particularly mathematical proofs and rigorous analysis. It uses a three-agent system: a creator, verifier, and refiner, working together to produce high-quality outputs through iterative refinement.

## How It Works

1. **Creator Agent**: Generates the initial solution or proof
2. **Verifier Agents**: Multiple verifiers run concurrently to check the solution
3. **Majority Voting**: Consensus is reached on verification results
4. **Refiner Agents**: Multiple refiners improve the solution based on feedback
5. **Iterative Process**: Can run multiple loops for continuous improvement

This architecture is ideal for:
- Mathematical proofs and theorem generation
- Complex problem-solving requiring verification
- Tasks needing rigorous validation
- Academic and research applications

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
from swarms import Agent, MALT
```

### Step 2: Create MALT with Preset Agents (Recommended)

The easiest way is to use MALT's preset agents, which are optimized for mathematical proofs:

```python
malt = MALT(
    preset_agents=True,  # Uses optimized proof creator, verifier, and refiner
    max_loops=1,
    return_dict=False,  # Return as string
)
```

### Step 3: Run MALT on a Task

```python
task = "Prove that the sum of two even numbers is always even"

result = malt.run(task=task)

print(result)
```

## Custom Agents Example

### Step 1: Create Custom Agents

```python
# Creator Agent: Generates solutions
creator = Agent(
    agent_name="Solution-Creator",
    system_prompt="""You are an expert problem solver. Generate comprehensive 
    solutions with clear reasoning and step-by-step explanations.""",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Verifier Agent: Validates solutions
verifier = Agent(
    agent_name="Solution-Verifier",
    system_prompt="""You are a rigorous validator. Check solutions for correctness, 
    logical consistency, and completeness. Identify any errors or gaps.""",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Refiner Agent: Improves solutions
refiner = Agent(
    agent_name="Solution-Refiner",
    system_prompt="""You are a solution refiner. Take verified feedback and improve 
    the solution by addressing identified issues and enhancing clarity.""",
    model_name="gpt-4o-mini",
    max_loops=1,
)
```

### Step 2: Create MALT with Custom Agents

```python
malt = MALT(
    main_agent=creator,
    verifier_agent=verifier,
    refiner_agent=refiner,
    max_loops=1,
    return_dict=False,
)
```

### Step 3: Run the Task

```python
task = "Solve: If a train travels 120 km in 2 hours, what is its average speed? Show your work."

result = malt.run(task=task)
print(result)
```

## Understanding the Process

1. **Creation Phase**: The creator agent generates an initial solution
2. **Verification Phase**: Three verifier agents run concurrently to check the solution
3. **Voting Phase**: A majority voting agent synthesizes the verification results
4. **Refinement Phase**: Three refiner agents improve the solution based on feedback
5. **Output**: The refined solution is returned

## Output Formats

### String Format (Default)

```python
malt = MALT(
    preset_agents=True,
    return_list=False,
    return_dict=False,
)

result = malt.run(task="Your task here")
# Returns: str
```

### List Format

```python
malt = MALT(
    preset_agents=True,
    return_list=True,
    return_dict=False,
)

result = malt.run(task="Your task here")
# Returns: list of messages
```

### Dictionary Format

```python
malt = MALT(
    preset_agents=True,
    return_list=False,
    return_dict=True,
)

result = malt.run(task="Your task here")
# Returns: dict of messages
```

## Multiple Iterations

You can run multiple loops for iterative improvement:

```python
malt = MALT(
    preset_agents=True,
    max_loops=3,  # Run 3 iterations
    return_dict=False,
)

result = malt.run(task="Prove the Pythagorean theorem")
```

## Batch Processing

Process multiple tasks:

```python
tasks = [
    "Prove that sqrt(2) is irrational",
    "Prove that the sum of angles in a triangle is 180 degrees",
    "Prove that 0.999... = 1",
]

results = malt.run_batched(tasks)

print(results)
```

## Use Cases

### Mathematical Proofs

```python
malt = MALT(preset_agents=True, max_loops=1)
result = malt.run("Prove that there are infinitely many prime numbers")
```

### Problem Solving

```python
malt = MALT(preset_agents=True, max_loops=2)
result = malt.run("""
Solve this optimization problem:
Maximize f(x,y) = 2x + 3y subject to:
- x + y <= 10
- 2x + y <= 16
- x >= 0, y >= 0
""")
```

### Algorithm Verification

```python
malt = MALT(preset_agents=True, max_loops=1)
result = malt.run("""
Verify the correctness of this algorithm:
1. Sort the array
2. Use binary search to find the target
3. Return the index if found, -1 otherwise

Provide a proof of correctness.
""")
```

## Support and Community

If you're facing issues or want to learn more, check out the following resources:

| Platform | Link | Description |
|----------|------|-------------|
| 📚 Documentation | [docs.swarms.world](https://docs.swarms.world) | Official documentation and guides |
| 💬 Discord | [Join Discord](https://discord.gg/EamjgSaEQf) | Live chat and community support |
| 🐦 Twitter | [@swarms_corp](https://x.com/swarms_corp) | Latest news and announcements |

