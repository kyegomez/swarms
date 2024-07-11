from swarms import Agent, OpenAIChat
from swarms_memory import ChromaDB
from swarms.models.tiktoken_wrapper import TikTokenizer

# Initialize the agent
agent = Agent(
    agent_name="Accounting Assistant",
    system_prompt="You're the accounting agent, your purpose is to generate a profit report for a company!",
    agent_description="Generate a profit report for a company!",
    llm=OpenAIChat(),
    max_loops=1,
    autosave=True,
    # dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    # interactive=True, # Set to False to disable interactive mode
    # docs_folder="docs",
    # pdf_path="docs/accounting_agent.pdf",
    # sop="Calculate the profit for a company.",
    # sop_list=["Calculate the profit for a company."],
    # user_name="User",
    # # docs=
    # # docs_folder="docs",
    # retry_attempts=3,
    # context_length=1000,
    # tool_schema = dict
    context_length=1000,
    # long_term_memory=ChromaDB(docs_folder="artifacts"),
    long_term_memory=ChromaDB(
        docs_folder="artifacts", output_dir="test", n_results=1
    ),
    tokenizer=TikTokenizer(),
)

agent.run("Whats the best agent available for accounting")
