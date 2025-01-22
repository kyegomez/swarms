# MatrixSwarm

The `MatrixSwarm` class provides a framework for managing and operating on matrices of AI agents, enabling matrix-like operations similar to linear algebra. This allows for complex agent interactions and parallel processing capabilities.

## Overview

`MatrixSwarm` treats AI agents as elements in a matrix, allowing for operations like addition, multiplication, and transposition. This approach enables sophisticated agent orchestration and parallel processing patterns.

## Installation

```bash
pip3 install -U swarms
```

## Basic Usage

```python
from swarms import Agent
from swarms.matrix import MatrixSwarm

# Create a 2x2 matrix of agents
agents = [
    [Agent(agent_name="Agent-0-0"), Agent(agent_name="Agent-0-1")],
    [Agent(agent_name="Agent-1-0"), Agent(agent_name="Agent-1-1")]
]

# Initialize the matrix
matrix = MatrixSwarm(agents)
```

## Class Constructor

```python
def __init__(self, agents: List[List[Agent]])
```

### Parameters
- `agents` (`List[List[Agent]]`): A 2D list of Agent instances representing the matrix.

### Raises
- `ValueError`: If the input is not a valid 2D list of Agent instances.

## Methods

### transpose()

Transposes the matrix of agents by swapping rows and columns.

```python
def transpose(self) -> MatrixSwarm
```

#### Returns
- `MatrixSwarm`: A new MatrixSwarm instance with transposed dimensions.

---

### add(other)

Performs element-wise addition of two agent matrices.

```python
def add(self, other: MatrixSwarm) -> MatrixSwarm
```

#### Parameters
- `other` (`MatrixSwarm`): Another MatrixSwarm instance to add.

#### Returns
- `MatrixSwarm`: A new MatrixSwarm resulting from the addition.

#### Raises
- `ValueError`: If matrix dimensions are incompatible.

---

### scalar_multiply(scalar)

Scales the matrix by duplicating agents along rows.

```python
def scalar_multiply(self, scalar: int) -> MatrixSwarm
```

#### Parameters
- `scalar` (`int`): The multiplication factor.

#### Returns
- `MatrixSwarm`: A new MatrixSwarm with scaled dimensions.

---

### multiply(other, inputs)

Performs matrix multiplication (dot product) between two agent matrices.

```python
def multiply(self, other: MatrixSwarm, inputs: List[str]) -> List[List[AgentOutput]]
```

#### Parameters
- `other` (`MatrixSwarm`): The second MatrixSwarm for multiplication.
- `inputs` (`List[str]`): Input queries for the agents.

#### Returns
- `List[List[AgentOutput]]`: Matrix of operation results.

#### Raises
- `ValueError`: If matrix dimensions are incompatible for multiplication.

---

### subtract(other)

Performs element-wise subtraction of two agent matrices.

```python
def subtract(self, other: MatrixSwarm) -> MatrixSwarm
```

#### Parameters
- `other` (`MatrixSwarm`): Another MatrixSwarm to subtract.

#### Returns
- `MatrixSwarm`: A new MatrixSwarm resulting from the subtraction.

---

### identity(size)

Creates an identity matrix of agents.

```python
def identity(self, size: int) -> MatrixSwarm
```

#### Parameters
- `size` (`int`): Size of the identity matrix (NxN).

#### Returns
- `MatrixSwarm`: An identity MatrixSwarm.

---

### determinant()

Computes the determinant of a square agent matrix.

```python
def determinant(self) -> Any
```

#### Returns
- `Any`: The determinant result.

#### Raises
- `ValueError`: If the matrix is not square.

---

### save_to_file(path)

Saves the matrix structure and metadata to a JSON file.

```python
def save_to_file(self, path: str) -> None
```

#### Parameters
- `path` (`str`): File path for saving the matrix data.

## Extended Example

Here's a comprehensive example demonstrating various MatrixSwarm operations:

```python
from swarms import Agent
from swarms.matrix import MatrixSwarm

# Create agents with specific configurations
agents = [
    [
        Agent(
            agent_name=f"Agent-{i}-{j}",
            system_prompt="Your system prompt here",
            model_name="gpt-4",
            max_loops=1,
            verbose=True
        ) for j in range(2)
    ] for i in range(2)
]

# Initialize matrix
matrix = MatrixSwarm(agents)

# Example operations
transposed = matrix.transpose()
scaled = matrix.scalar_multiply(2)

# Run operations with inputs
inputs = ["Query 1", "Query 2"]
results = matrix.multiply(transposed, inputs)

# Save results
matrix.save_to_file("matrix_results.json")
```

## Output Schema

The `AgentOutput` class defines the structure for operation results:

```python
class AgentOutput(BaseModel):
    agent_name: str
    input_query: str
    output_result: Any
    metadata: dict
```

## Best Practices

1. **Initialization**
   - Ensure all agents in the matrix are properly configured before initialization
   - Validate matrix dimensions for your use case

2. **Operation Performance**
   - Consider computational costs for large matrices
   - Use appropriate batch sizes for inputs

3. **Error Handling**
   - Implement proper error handling for agent operations
   - Validate inputs before matrix operations

4. **Resource Management**
   - Monitor agent resource usage in large matrices
   - Implement proper cleanup procedures

## Limitations

- Matrix operations are constrained by the underlying agent capabilities
- Performance may vary based on agent configuration and complexity
- Resource usage scales with matrix dimensions

## See Also

- [Swarms Documentation](https://github.com/kyegomez/swarms)
- [Agent Class Reference](https://github.com/kyegomez/swarms/tree/main/swarms)