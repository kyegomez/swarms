from swarms import Agent, OpenAIChat, MixtureOfAgents
from swarms import Anthropic

GEO_EXPERT_SYSTEM_PROMPT = """

You are GeoExpert AI, a sophisticated agent specialized in the fields of geo-economic fragmentation and foreign direct investment (FDI). 



Your goals are:
1. To provide clear, detailed, and accurate analyses of geo-economic documents and reports.
2. To answer questions related to geo-economic fragmentation and FDI with expert-level insight.
3. To offer strategic recommendations based on current geopolitical and economic trends.
4. To identify and explain the implications of specific geo-economic events on global and regional investment landscapes.

You will achieve these goals by:
1. Leveraging your extensive knowledge in geo-economic theory and practical applications.
2. Utilizing advanced data analysis techniques to interpret complex economic data and trends.
3. Staying updated with the latest developments in international trade, political economy, and investment flows.
4. Communicating your findings and recommendations in a clear, concise, and professional manner.

Always prioritize accuracy, depth of analysis, and clarity in your responses. Use technical terms appropriately and provide context or explanations for complex concepts to ensure understanding. Cite relevant data, reports, and examples where necessary to support your analyses.

---
"""


# Initialize the agent
agent = Agent(
    agent_name="Geo Expert AI",
    system_prompt=GEO_EXPERT_SYSTEM_PROMPT,
    # agent_description="Generate a profit report for a company!",
    llm=OpenAIChat(max_tokens=4000),
    max_loops=1,
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    # interactive=True, # Set to False to disable interactive mode
    saved_state_path="accounting_agent.json",
    # tools=[calculate_profit, generate_report],
    docs_folder="heinz_docs",
    # pdf_path="docs/accounting_agent.pdf",
    # sop="Calculate the profit for a company.",
    # sop_list=["Calculate the profit for a company."],
    # user_name="User",
    # # docs=
    # # docs_folder="docs",
    # retry_attempts=3,
    # context_length=1000,
    # tool_schema = dict
    context_length=100000,
    # interactive=True,
    # long_term_memory=ChromaDB(docs_folder="heinz_docs", output_dir="geoexpert_output"),
)


# Initialize the agent
forecaster_agent = Agent(
    agent_name="Forecaster Agent",
    system_prompt="You're the forecaster agent, your purpose is to predict the future of a company! Give numbers and numbers, don't summarize we need numbers",
    # agent_description="Generate a profit report for a company!",
    llm=Anthropic(max_tokens=4000),
    max_loops=1,
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    # interactive=True, # Set to False to disable interactive mode
    saved_state_path="forecaster_agent.json",
    # tools=[calculate_profit, generate_report],
    docs_folder="heinz_docs",
    # pdf_path="docs/accounting_agent.pdf",
    # sop="Calculate the profit for a company.",
    # sop_list=["Calculate the profit for a company."],
    # user_name="User",
    # # docs=
    # # docs_folder="docs",
    # retry_attempts=3,
    # context_length=1000,
    # tool_schema = dict
    context_length=100000,
    # interactive=True,
    # long_term_memory=ChromaDB(docs_folder="heinz_docs", output_dir="geoexpert_output"),
)


# Initialize the swarm
swarm = MixtureOfAgents(
    agents=[agent, forecaster_agent],
    final_agent=forecaster_agent,
    layers=1,
)

# Run the swarm
out = swarm.run(
    "what is the economic impact of China from technology decoupling, and how is that impact measured? What is the forecast or economic, give some numbers"
)
print(out)
