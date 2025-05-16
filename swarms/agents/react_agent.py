from swarms import Agent
from typing import List


# System prompt for REACT agent
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

# Schema for REACT agent responses
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
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

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


# if __name__ == "__main__":
#     agent = ReactAgent(
#         max_loops=1
#     )  # Increased max_loops to see the iteration
#     result = agent.run(
#         "Write a short story about a robot that can fly."
#     )
#     print(result)
