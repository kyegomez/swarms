# Building a ReAct Agent with Swarms

ReAct (Reason, Act, Observe) is an agent pattern where the model iteratively reasons about a task, takes an action, observes the result, and repeats — building on memory from each prior step. This guide walks through building a `ReactAgent` using the Swarms framework.

---

## How It Works

Each step follows five stages:

1. **Memory** — reflect on what happened in previous steps
2. **Observe** — assess the current state given new info and history
3. **Think** — reason by combining observations with past context
4. **Plan** — decide what to do next, avoiding repeated failures
5. **Act** — execute the action that advances toward the goal

The agent runs this loop `max_loops` times, carrying memory forward into each subsequent step.

---

## Full Implementation

### System Prompt

The system prompt instructs the model to follow the ReAct loop and structure its output accordingly.

```python
REACT_AGENT_PROMPT = """
You are a REACT (Reason, Act, Observe) agent designed to solve tasks through an iterative process of reasoning and action. You maintain memory of previous steps to build upon past actions and observations.

Your process follows these key components:

1. MEMORY: Review and utilize previous steps
   - Access and analyze previous observations
   - Build upon past thoughts and plans
   - Learn from previous actions
   - Use historical context to make better decisions

2. OBSERVE: Analyze current state
   - Consider both new information and memory
   - Identify relevant patterns from past steps
   - Note any changes or progress made
   - Evaluate success of previous actions

3. THINK: Process and reason
   - Combine new observations with historical knowledge
   - Consider how past steps influence current decisions
   - Identify patterns and learning opportunities
   - Plan improvements based on previous outcomes

4. PLAN: Develop next steps
   - Create strategies that build on previous success
   - Avoid repeating unsuccessful approaches
   - Consider long-term goals and progress
   - Maintain consistency with previous actions

5. ACT: Execute with context
   - Implement actions that progress from previous steps
   - Build upon successful past actions
   - Adapt based on learned experiences
   - Maintain continuity in approach

For each step, you should:
- Reference relevant previous steps
- Show how current decisions relate to past actions
- Demonstrate learning and adaptation
- Maintain coherent progression toward the goal

Your responses should be structured, logical, and show clear reasoning that builds upon previous steps."""
```

---

### Output Schema

The agent uses a structured tool schema to enforce consistent JSON output across every step.

```python
react_agent_schema = {
    "type": "function",
    "function": {
        "name": "generate_react_response",
        "description": "Generates a structured REACT agent response with memory of previous steps",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_reflection": {
                    "type": "string",
                    "description": "Analysis of previous steps and their influence on current thinking",
                },
                "observation": {
                    "type": "string",
                    "description": "Current state observation incorporating both new information and historical context",
                },
                "thought": {
                    "type": "string",
                    "description": "Reasoning that builds upon previous steps and current observation",
                },
                "plan": {
                    "type": "string",
                    "description": "Structured plan that shows progression from previous actions",
                },
                "action": {
                    "type": "string",
                    "description": "Specific action that builds upon previous steps and advances toward the goal",
                },
            },
            "required": [
                "memory_reflection",
                "observation",
                "thought",
                "plan",
                "action",
            ],
        },
    },
}
```

Each response is guaranteed to contain all five fields: `memory_reflection`, `observation`, `thought`, `plan`, and `action`.

---

### ReactAgent Class

```python
from swarms import Agent
from typing import List


class ReactAgent:
    def __init__(
        self,
        name: str = "react-agent-o1",
        description: str = "A react agent that uses o1 preview to solve tasks",
        model_name: str = "openai/gpt-4o",
        max_loops: int = 1,
    ):
        self.name = name
        self.description = description
        self.model_name = model_name
        self.max_loops = max_loops

        self.agent = Agent(
            agent_name=self.name,
            agent_description=self.description,
            model_name=self.model_name,
            max_loops=1,
            tools_list_dictionary=[react_agent_schema],
            output_type="final",
        )

        # Initialize memory for storing steps
        self.memory: List[str] = []

    def step(self, task: str) -> str:
        """Execute a single step of the REACT process.

        Args:
            task: The task description or current state

        Returns:
            String response from the agent
        """
        response = self.agent.run(task)
        print(response)
        return response

    def run(self, task: str, *args, **kwargs) -> List[str]:
        """Run the REACT agent for multiple steps with memory.

        Args:
            task: The initial task description

        Returns:
            List of all steps taken as strings
        """
        # Reset memory at the start of a new run
        self.memory = []

        current_task = task
        for i in range(self.max_loops):
            print(f"\nExecuting step {i+1}/{self.max_loops}")
            step_result = self.step(current_task)
            print(step_result)

            # Store step in memory
            self.memory.append(step_result)

            # Update task with previous response and memory context
            memory_context = (
                "\n\nMemory of previous steps:\n"
                + "\n".join(
                    f"Step {j+1}:\n{step}"
                    for j, step in enumerate(self.memory)
                )
            )

            current_task = f"Previous response:\n{step_result}\n{memory_context}\n\nContinue with the original task: {task}"

        return self.memory
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| `max_loops=1` on the inner `Agent` | Each ReAct step is a single LLM call; looping is handled by `ReactAgent.run()` |
| `tools_list_dictionary` with the schema | Forces structured output — every step returns the same five fields |
| Memory passed as text in `current_task` | Keeps the full history visible in the prompt without needing a vector store |
| `output_type="final"` | Returns only the final tool call output, not intermediate streaming chunks |

---

## Usage

```python
agent = ReactAgent(
    name="my-react-agent",
    model_name="openai/gpt-4o",
    max_loops=3,
)

results = agent.run("Write a short story about a robot that can fly.")

for i, step in enumerate(results):
    print(f"--- Step {i+1} ---")
    print(step)
```

### What `run()` returns

`run()` returns a `List[str]` — one entry per loop iteration. Each string is the raw structured JSON output from the model containing all five ReAct fields.

---

## Extending the Agent

### Swap the model

```python
agent = ReactAgent(model_name="anthropic/claude-opus-4-6", max_loops=5)
```

Any LiteLLM-compatible model string works.

### Add custom tools

Pass additional tool schemas alongside `react_agent_schema`:

```python
my_tool_schema = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
}

self.agent = Agent(
    ...
    tools_list_dictionary=[react_agent_schema, my_tool_schema],
)
```

### Persist memory across runs

By default `self.memory` is reset on each `run()` call. To persist across runs, remove the reset line and pass memory in externally:

```python
def run(self, task: str) -> List[str]:
    # self.memory = []  # remove this to persist
    ...
```
