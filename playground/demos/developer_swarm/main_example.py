"""
Swarm of developers that write documentation and tests for a given code snippet.

This is a simple example of how to use the swarms library to create a swarm of developers that write documentation and tests for a given code snippet.

The swarm is composed of two agents:
    - Documentation agent: writes documentation for a given code snippet.
    - Tests agent: writes tests for a given code snippet.

The swarm is initialized with a language model that is used by the agents to generate text. In this example, we use the OpenAI GPT-3 language model.

Agent:
Documentation agent -> Tests agent


"""

import os

from dotenv import load_dotenv

from swarms.models import OpenAIChat
from swarms.prompts.programming import DOCUMENTATION_SOP, TEST_SOP
from swarms.structs import Agent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


TASK = """
CODE

"""

# Initialize the language model
llm = OpenAIChat(openai_api_key=api_key, max_tokens=4096)


# Documentation agent
documentation_agent = Agent(
    llm=llm,
    sop=DOCUMENTATION_SOP,
    max_loops=1,
)


# Tests agent
tests_agent = Agent(
    llm=llm,
    sop=TEST_SOP,
    max_loops=2,
)


# Run the documentation agent
documentation = documentation_agent.run(
    f"Write documentation for the following code:{TASK}"
)

# Run the tests agent
tests = tests_agent.run(
    f"Write tests for the following code:{TASK} here is the"
    f" documentation: {documentation}"
)
