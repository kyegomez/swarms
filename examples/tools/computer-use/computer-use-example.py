"""
Agent Computer Use Example: Create and Debug a PyTorch Model

This example demonstrates an agent using the computer_use toolkit to:
1. Create a PyTorch model file
2. Run tests to verify it works
3. Debug and fix any issues over multiple loops

The agent has access to all computer_use tools:
- read_file, list_directory, grep_files (read-only)
- write_file, edit_file, patch_file, delete_file (write)
- run_command (shell)
"""

from swarms import Agent
from swarms.tools.computer_use import create_computer_use_tools

tools = create_computer_use_tools()


agent = Agent(
    agent_name="PyTorch-Debugger",
    system_prompt="""
You are an expert Python/PyTorch developer agent. Your task is to create and debug
a PyTorch model file called `/$HOME/pytorch_model.py`.

Follow this workflow in each loop:

1. PLAN: Decide what needs to be done based on current state
2. CREATE/MODIFY: Use write_file/edit_file to create or fix code
3. TEST: Use run_command to run `python -c "from examples.models.pytorch_model import *; print('Import OK')"`
4. FIX: If there are errors, use edit_file to fix them
5. REPORT: Summarize what was done

The model should:
- Define a NeuralNetwork class with at least 3 layers
- Include forward() method
- Include a training step function
- Include an evaluation function
- Be well-documented with docstrings
- Handle both CPU and GPU (cuda) devices
- Have proper error handling

When you see import errors or runtime errors:
- Read the model file to understand the code
- Identify the bug
- Fix it with edit_file
- Re-run the test

Continue until the model imports and runs without errors.
When fully working, create a simple test that demonstrates the model training.

Return a final report with:
- The final working code
- Test results showing successful training
- Any bugs that were fixed
""",
    model_name="gpt-4o",
    max_loops=10,
    tools=list(tools.values()),
)

result = agent.run(
    """
Create and debug a PyTorch model at `/$HOME/pytorch_model.py`.

Start by creating the initial model file, then test it.
If there are errors, fix them. Keep iterating until the model works correctly.

The model should demonstrate:
1. A feedforward neural network
2. Forward pass
3. Training loop with loss computation
4. Evaluation mode
5. GPU/CPU device handling
"""
)

print(result)
