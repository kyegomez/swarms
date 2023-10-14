import os

import interpreter

from swarms.agents.hf_agents import HFAgent
from swarms.agents.omni_modal_agent import OmniModalAgent
from swarms.models import OpenAIChat
from swarms.tools.autogpt import tool
from swarms.workers import Worker

# Initialize API Key
api_key = ""


# Initialize the language model,
# This model can be swapped out with Anthropic, ETC, Huggingface Models like Mistral, ETC
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
)


# wrap a function with the tool decorator to make it a tool, then add docstrings for tool documentation
@tool
def hf_agent(task: str = None):
    """
    An tool that uses an openai model to call and respond to a task by search for a model on huggingface
    It first downloads the model then uses it.

    Rules: Don't call this model for simple tasks like generating a summary, only call this tool for multi modal tasks like generating images, videos, speech, etc

    """
    agent = HFAgent(model="text-davinci-003", api_key=api_key)
    response = agent.run(task, text="¡Este es un API muy agradable!")
    return response


# wrap a function with the tool decorator to make it a tool
@tool
def omni_agent(task: str = None):
    """
    An tool that uses an openai Model to utilize and call huggingface models and guide them to perform a task.

    Rules: Don't call this model for simple tasks like generating a summary, only call this tool for multi modal tasks like generating images, videos, speech
    The following tasks are what this tool should be used for:

    Tasks omni agent is good for:
    --------------
    document-question-answering
    image-captioning
    image-question-answering
    image-segmentation
    speech-to-text
    summarization
    text-classification
    text-question-answering
    translation
    huggingface-tools/text-to-image
    huggingface-tools/text-to-video
    text-to-speech
    huggingface-tools/text-download
    huggingface-tools/image-transformation
    """
    agent = OmniModalAgent(llm)
    response = agent.run(task)
    return response


# Code Interpreter
@tool
def compile(task: str):
    """
    Open Interpreter lets LLMs run code (Python, Javascript, Shell, and more) locally.
    You can chat with Open Interpreter through a ChatGPT-like interface in your terminal
    by running $ interpreter after installing.

    This provides a natural-language interface to your computer's general-purpose capabilities:

    Create and edit photos, videos, PDFs, etc.
    Control a Chrome browser to perform research
    Plot, clean, and analyze large datasets
    ...etc.
    ⚠️ Note: You'll be asked to approve code before it's run.

    Rules: Only use when given to generate code or an application of some kind
    """
    task = interpreter.chat(task, return_messages=True)
    interpreter.chat()
    interpreter.reset(task)

    os.environ["INTERPRETER_CLI_AUTO_RUN"] = True
    os.environ["INTERPRETER_CLI_FAST_MODE"] = True
    os.environ["INTERPRETER_CLI_DEBUG"] = True


# Append tools to an list
tools = [hf_agent, omni_agent, compile]


# Initialize a single Worker node with previously defined tools in addition to it's
# predefined tools
node = Worker(
    llm=llm,
    ai_name="Optimus Prime",
    openai_api_key=api_key,
    ai_role="Worker in a swarm",
    external_tools=tools,
    human_in_the_loop=False,
    temperature=0.5,
)

# Specify task
task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."

# Run the node on the task
response = node.run(task)

# Print the response
print(response)
