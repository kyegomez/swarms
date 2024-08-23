from swarms.structs.agent_registry import AgentRegistry
from swarms import Agent
from swarms.models import Anthropic


# Initialize the agents
growth_agent1 = Agent(
    agent_name="Marketing Specialist",
    system_prompt="You're the marketing specialist, your purpose is to help companies grow by improving their marketing strategies!",
    agent_description="Improve a company's marketing strategies!",
    llm=Anthropic(),
    max_loops="auto",
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    saved_state_path="marketing_specialist.json",
    stopping_token="Stop!",
    interactive=True,
    context_length=1000,
)

growth_agent2 = Agent(
    agent_name="Sales Specialist",
    system_prompt="You're the sales specialist, your purpose is to help companies grow by improving their sales strategies!",
    agent_description="Improve a company's sales strategies!",
    llm=Anthropic(),
    max_loops="auto",
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    saved_state_path="sales_specialist.json",
    stopping_token="Stop!",
    interactive=True,
    context_length=1000,
)

growth_agent3 = Agent(
    agent_name="Product Development Specialist",
    system_prompt="You're the product development specialist, your purpose is to help companies grow by improving their product development strategies!",
    agent_description="Improve a company's product development strategies!",
    llm=Anthropic(),
    max_loops="auto",
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    saved_state_path="product_development_specialist.json",
    stopping_token="Stop!",
    interactive=True,
    context_length=1000,
)

growth_agent4 = Agent(
    agent_name="Customer Service Specialist",
    system_prompt="You're the customer service specialist, your purpose is to help companies grow by improving their customer service strategies!",
    agent_description="Improve a company's customer service strategies!",
    llm=Anthropic(),
    max_loops="auto",
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    saved_state_path="customer_service_specialist.json",
    stopping_token="Stop!",
    interactive=True,
    context_length=1000,
)


# Register the agents\
registry = AgentRegistry()

# Register the agents
registry.add("Marketing Specialist", growth_agent1)
registry.add("Sales Specialist", growth_agent2)
registry.add("Product Development Specialist", growth_agent3)
registry.add("Customer Service Specialist", growth_agent4)


# Query the agents
registry.get("Marketing Specialist")
registry.get("Sales Specialist")
registry.get("Product Development Specialist")

# Get all the agents
registry.list_agents()
