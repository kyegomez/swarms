from swarms.models import OpenAIChat
from swarms.workers import Worker
from swarms.tools.autogpt import tool
from swarms.agents.hf_agents import HFAgent
from swarms.agents.omni_modal_agent import OmniModalAgent

#Initialize API Key
api_key = ""


# Initialize the language model,
# This model can be swapped out with Anthropic, ETC, Huggingface Models like Mistral, ETC
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
)

#wrap a function with the tool decorator to make it a tool
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


#wrap a function with the tool decorator to make it a tool
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

# Append tools to an list
tools = [
    hf_agent,
    omni_agent,
    
]


#Initialize a single Worker node with previously defined tools in addition to it's
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

#Specify task
task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."

# Run the node on the task
response = node.run(task)

# Print the response
print(response)
