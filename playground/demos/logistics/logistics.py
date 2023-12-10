from swarms.structs import Agent
import os
from dotenv import load_dotenv
from swarms.models import GPT4VisionAPI
from swarms.prompts.logistics import (
    Health_Security_Agent_Prompt,
    Quality_Control_Agent_Prompt,
    Productivity_Agent_Prompt,
    Safety_Agent_Prompt,
    Security_Agent_Prompt,
    Sustainability_Agent_Prompt,
    Efficiency_Agent_Prompt,
)
# from swarms.utils.phoenix_handler import phoenix_trace_decorator
# from swarms.utils.banana_wrapper import banana

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# GPT4VisionAPI or llama
# @banana #- deploy to banana
llm = GPT4VisionAPI(openai_api_key=api_key)

# Image for analysis
factory_image = "factory_image1.jpg"

# Initialize agents with respective prompts
# @phoenix_trace_decorator("This function is an agent and is traced by Phoenix.")
health_security_agent = Agent(
    llm=llm,
    sop=Health_Security_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

#@phoenix_trace_decorator("This function is an agent and is traced by Phoenix.")
quality_control_agent = Agent(
    llm=llm,
    sop=Quality_Control_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

#@phoenix_trace_decorator("This function is an agent and is traced by Phoenix.")
productivity_agent = Agent(
    llm=llm,
    sop=Productivity_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

#@phoenix_trace_decorator("This function is an agent and is traced by Phoenix.")
safety_agent = Agent(
    llm=llm, sop=Safety_Agent_Prompt, max_loops=1, multi_modal=True
)

#@phoenix_trace_decorator("This function is an agent and is traced by Phoenix.")
security_agent = Agent(
    llm=llm, sop=Security_Agent_Prompt, max_loops=1, multi_modal=True
)

#@phoenix_trace_decorator("This function is an agent and is traced by Phoenix.")
sustainability_agent = Agent(
    llm=llm,
    sop=Sustainability_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

#@phoenix_trace_decorator("This function is an agent and is traced by Phoenix.")
efficiency_agent = Agent(
    llm=llm,
    sop=Efficiency_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

# Run agents with respective tasks on the same image
health_analysis = health_security_agent.run(
    "Analyze the safety of this factory", factory_image
)
quality_analysis = quality_control_agent.run(
    "Examine product quality in the factory", factory_image
)
productivity_analysis = productivity_agent.run(
    "Evaluate factory productivity", factory_image
)
safety_analysis = safety_agent.run(
    "Inspect the factory's adherence to safety standards",
    factory_image,
)
security_analysis = security_agent.run(
    "Assess the factory's security measures and systems",
    factory_image,
)
sustainability_analysis = sustainability_agent.run(
    "Examine the factory's sustainability practices", factory_image
)
efficiency_analysis = efficiency_agent.run(
    "Analyze the efficiency of the factory's manufacturing process",
    factory_image,
)
