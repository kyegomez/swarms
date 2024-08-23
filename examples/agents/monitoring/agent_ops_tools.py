"""
* WORKING

What this script does:
Simple agent run to test AgentOps to record tool actions (https://www.agentops.ai/)

Requirements:
1. Create an account on https://www.agentops.ai/ and run pip install agentops
2. Add the folowing API key(s) in your .env file:
   - OPENAI_API_KEY
   - AGENTOPS_API_KEY
3. Go to your agentops dashboard to observe your activity

"""

################ Adding project root to PYTHONPATH ################################
# If you are running examples examples in the project files directly, use this:

import sys
import os

sys.path.insert(0, os.getcwd())

################ Adding project root to PYTHONPATH ################################


from swarms import Agent, OpenAIChat
from agentops import record_function


# Add agentops decorator on your tools
@record_function("length_checker")
def length_checker(string: str) -> int:
    """
    For a given string it returns the length of the string.

    Args:
        string (str): string to check the length of

    Returns:
        int: length of the string
    """
    return len(string)


agent1 = Agent(
    agent_name="lengther",
    system_prompt="return the length of the string",
    agent_description=(
        "For a given string it calls the function length_checker to return the length of the string."
    ),
    llm=OpenAIChat(),
    max_loops=1,
    agent_ops_on=True,
    tools=[length_checker],
    execute_tool=True,
)


agent1.run("hello")
