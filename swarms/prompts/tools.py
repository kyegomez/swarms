# Prompts
DYNAMIC_STOP_PROMPT = """

Now, when you 99% sure you have completed the task, you may follow the instructions below to escape the autonomous loop.

When you have finished the task from the Human, output a special token: <DONE>
This will enable you to leave the autonomous loop.
"""


# Make it able to handle multi input tools
DYNAMICAL_TOOL_USAGE = """
You have access to the following tools:
Output a JSON object with the following structure to use the tools
commands: {
    "tools": {
        tool1: "tool_name",
        "params": {
            "tool1": "inputs",
            "tool1": "inputs"
        }
        "tool2: "tool_name",
        "params": {
            "tool1": "inputs",
            "tool1": "inputs"
        }
        "tool3: "tool_name",
        "params": {
            "tool1": "inputs",
            "tool1": "inputs"
        }
    }
}

-------------TOOLS---------------------------
{tools}
"""

SCENARIOS = """
commands: {
    "tools": {
        tool1: "tool_name",
        "params": {
            "tool1": "inputs",
            "tool1": "inputs"
        }
        "tool2: "tool_name",
        "params": {
            "tool1": "inputs",
            "tool1": "inputs"
        }
        "tool3: "tool_name",
        "params": {
            "tool1": "inputs",
            "tool1": "inputs"
        }
    }
}

"""
