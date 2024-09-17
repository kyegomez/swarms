import os
from swarms import Gemini, Agent
from swarms.structs.multi_process_workflow import MultiProcessWorkflow
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Initialize LLM
llm = Gemini(
    model_name="gemini-pro",
    api_key=api_key,
)

# Initialize the agents
finance_agent = Agent(
    agent_name="Finance Agent",
    llm=llm,
    max_loops=1,
    system_prompt="Finance",
)

marketing_agent = Agent(
    agent_name="Marketing Agent",
    llm=llm,
    max_loops=1,
    system_prompt="Marketing",
)

product_agent = Agent(
    agent_name="Product Agent",
    llm=llm,
    max_loops=1,
    system_prompt="Product",
)

other_agent = Agent(
    agent_name="Other Agent",
    llm=llm,
    max_loops=1,
    system_prompt="Other",
)

# Swarm
workflow = MultiProcessWorkflow(
    agents=[
        finance_agent,
        marketing_agent,
        product_agent,
        other_agent,
    ],
    max_workers=5,
    autosave=True,
)


# Run the workflow
results = workflow.run(
    "What is the best way to market a new product?"
)
