import os
from dotenv import load_dotenv
from swarms.models import OpenAIChat
from swarms.structs import Agent
import swarms.prompts.autoswarm as sdsp

# Load environment variables and initialize the OpenAI Chat model
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAIChat(model_name="gpt-4", openai_api_key=api_key)

user_idea = "screenplay writing team"

role_identification_agent = Agent(
    llm=llm,
    sop=sdsp.AGENT_ROLE_IDENTIFICATION_AGENT_PROMPT,
    max_loops=1,
)
agent_configuration_agent = Agent(
    llm=llm, sop=sdsp.AGENT_CONFIGURATION_AGENT_PROMPT, max_loops=1
)
swarm_assembly_agent = Agent(
    llm=llm, sop=sdsp.SWARM_ASSEMBLY_AGENT_PROMPT, max_loops=1
)
testing_optimization_agent = Agent(
    llm=llm, sop=sdsp.TESTING_OPTIMIZATION_AGENT_PROMPT, max_loops=1
)

# Process the user idea through each agent
role_identification_output = role_identification_agent.run(user_idea)
agent_configuration_output = agent_configuration_agent.run(
    role_identification_output
)
swarm_assembly_output = swarm_assembly_agent.run(
    agent_configuration_output
)
testing_optimization_output = testing_optimization_agent.run(
    swarm_assembly_output
)
