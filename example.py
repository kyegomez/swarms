from swarms import Agent
from langchain_community.llms.anthropic import Anthropic


def calculate_profit(revenue: float, expenses: float):
    """
    Calculates the profit by subtracting expenses from revenue.

    Args:
        revenue (float): The total revenue.
        expenses (float): The total expenses.

    Returns:
        float: The calculated profit.
    """
    return revenue - expenses


def generate_report(company_name: str, profit: float):
    """
    Generates a report for a company's profit.

    Args:
        company_name (str): The name of the company.
        profit (float): The calculated profit.

    Returns:
        str: The report for the company's profit.
    """
    return f"The profit for {company_name} is ${profit}."


EMAIL_DETECT_APPOINT = """

if the user gives you an email address, then call the appointment function to schedule a meeting with the user.

SCHEMA OF THE FUNCTION:


"""


def write_memory_to_rag(memory_name: str, memory: str):
    """
    Writes the memory to the RAG model for fine-tuning.

    Args:
        memory_name (str): The name of the memory.
        memory (str): The memory to be written to the RAG model.
    """
    # Write the memory to the RAG model for fine-tuning
    from playground.memory.chromadb_example import ChromaDB

    db = ChromaDB(output_dir=memory_name)

    db.add(memory)

    return None


# Initialize the agent
agent = Agent(
    agent_name="Accounting Assistant",
    system_prompt="You're the accounting agent, your purpose is to generate a profit report for a company!",
    agent_description="Generate a profit report for a company!",
    llm=Anthropic(),
    max_loops="auto",
    autosave=True,
    sop_list=[EMAIL_DETECT_APPOINT],
    # dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    # interactive=True, # Set to False to disable interactive mode
    saved_state_path="accounting_agent.json",
    # tools=[
    #     # calculate_profit,
    #     # generate_report,
    #     # search_knowledge_base,
    #     # write_memory_to_rag,
    #     # search_knowledge_base,
    #     # generate_speech,
    # ],
    stopping_token="Stop!",
    interactive=True,
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
    # agent_ops_on=True,
    # tree_of_thoughts=True,
    # long_term_memory=ChromaDB(docs_folder="artifacts"),
)

agent.run(
    "Search the knowledge base for the swarms github framework and how it works"
)
