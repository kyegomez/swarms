# RoundRobin: Round-Robin Task Execution in a Swarm

## Introduction

The `RoundRobinSwarm` class is designed to manage and execute tasks among multiple agents in a round-robin fashion. This approach ensures that each agent in a swarm receives an equal opportunity to execute tasks, which promotes fairness and efficiency in distributed systems. It is particularly useful in environments where collaborative, sequential task execution is needed among various agents.

## Conceptual Overview

### What is Round-Robin?

Round-robin is a scheduling technique commonly used in computing for managing processes in shared systems. It involves assigning a fixed time slot to each process and cycling through all processes in a circular order without prioritization. In the context of swarms of agents, this method ensures equitable distribution of tasks and resource usage among all agents.

### Application in Swarms

In swarms, `RoundRobinSwarm` utilizes the round-robin scheduling to manage tasks among agents like software components, autonomous robots, or virtual entities. This strategy is beneficial where tasks are interdependent or require sequential processing.

## Class Attributes

- `agents (List[Agent])`: List of agents participating in the swarm.
- `verbose (bool)`: Enables or disables detailed logging of swarm operations.
- `max_loops (int)`: Limits the number of times the swarm cycles through all agents.
- `index (int)`: Maintains the current position in the agent list to ensure round-robin execution.

## Methods

### `__init__`

Initializes the swarm with the provided list of agents, verbosity setting, and operational parameters.

**Parameters:**
- `agents`: Optional list of agents in the swarm.
- `verbose`: Boolean flag for detailed logging.
- `max_loops`: Maximum number of execution cycles.
- `callback`: Optional function called after each loop.

### `run`

Executes a specified task across all agents in a round-robin manner, cycling through each agent repeatedly for the number of specified loops.

**Conceptual Behavior:**
- Distribute the task sequentially among all agents starting from the current index.
- Each agent processes the task and potentially modifies it or produces new output.
- After an agent completes its part of the task, the index moves to the next agent.
- This cycle continues until the specified maximum number of loops is completed.
- Optionally, a callback function can be invoked after each loop to handle intermediate results or perform additional actions.

## Examples
### Example 1: Load Balancing Among Servers

In this example, `RoundRobinSwarm` is used to distribute network requests evenly among a group of servers. This is common in scenarios where load balancing is crucial for maintaining system responsiveness and scalability.

```python
from swarms.structs.round_robin import RoundRobinSwarm
from swarms import Agent

# Define server agents
server1 = Agent(agent_name="Server1", system_prompt="Handle network requests")
server2 = Agent(agent_name="Server2", system_prompt="Handle network requests")
server3 = Agent(agent_name="Server3", system_prompt="Handle network requests")

# Initialize the swarm with server agents
network_load_balancer = RoundRobinSwarm(agents=[server1, server2, server3], verbose=True)

# Define a network request task
task = "Process incoming network request"

# Simulate processing of multiple requests
for _ in range(10):  # Assume 10 incoming requests
    results = network_load_balancer.run(task)
    print("Request processed:", results)
```

### Example 2: Document Review Process

This example demonstrates how `RoundRobinSwarm` can be used to distribute parts of a document among different reviewers in a sequential manner, ensuring that each part of the document is reviewed by different agents.

```python
from swarms.structs.round_robin import RoundRobinSwarm
from swarms import Agent

# Define reviewer agents
reviewer1 = Agent(agent_name="Reviewer1", system_prompt="Review document section")
reviewer2 = Agent(agent_name="Reviewer2", system_prompt="Review document section")
reviewer3 = Agent(agent_name="Reviewer3", system_prompt="Review document section")

# Initialize the swarm with reviewer agents
document_review_swarm = RoundRobinSwarm(agents=[reviewer1, reviewer2, reviewer3], verbose=True)

# Define a document review task
task = "Review section of the document"

# Distribute sections of the document to different reviewers
for section in range(5):  # Assume the document has 5 sections
    results = document_review_swarm.run(task)
    print(f"Section {section + 1} reviewed:", results)
```

### Example 3: Multi-Stage Data Processing

In this scenario, `RoundRobinSwarm` facilitates a multi-stage data processing pipeline where data passes through multiple agents, each performing a specific type of data processing in sequence.

```python
from swarms.structs.round_robin import RoundRobinSwarm
from swarms import Agent

# Define data processing agents
preprocessor = Agent(agent_name="Preprocessor", system_prompt="Preprocess data")
analyzer = Agent(agent_name="Analyzer", system_prompt="Analyze data")
summarizer = Agent(agent_name="Summarizer", system_prompt="Summarize analysis results")

# Initialize the swarm with data processing agents
data_processing_swarm = RoundRobinSwarm(agents=[preprocessor, analyzer, summarizer], verbose=True)

# Define a data processing task
task = "Initial raw data"

# Run the data through the processing pipeline
results = data_processing_swarm.run(task)
print("Final results from data processing:", results)
```

These examples provide a glimpse into how the `RoundRobinSwarm` class can be adapted to various domains and applications, showcasing its versatility in managing tasks and resources in a distributed environment.
```

## Conclusion

The RoundRobinSwarm class provides a robust and flexible framework for managing tasks among multiple agents in a fair and efficient manner. This class is especially useful in environments where tasks need to be distributed evenly among a group of agents, ensuring that all tasks are handled timely and effectively. Through the round-robin algorithm, each agent in the swarm is guaranteed an equal opportunity to contribute to the overall task, promoting efficiency and collaboration.
