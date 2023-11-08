# !pip install --upgrade swarms==2.0.6

from swarms.models import BioGPT
from swarms.models.nougat import Nougat
from swarms.structs import Flow

# # URL of the image of the financial document
IMAGE_OF_FINANCIAL_DOC_URL = "bank_statement_2.jpg"

# Example usage
api_key = ""  # Your actual API key here

# Initialize the OCR model


# Initialize the language flow
llm = BioGPT()


# Create a prompt for the language model
def summary_agent_prompt(analyzed_doc: str):
    model = Nougat(
        max_new_tokens=5000,
    )

    out = model(analyzed_doc)

    return f"""
    Generate an actionable summary of this financial document, provide bulletpoints:

    Here is the Analyzed Document:
    ---
    {out}
    """


# Initialize the Flow with the language flow
flow1 = Flow(llm=llm, max_loops=1, dashboard=False)

# Create another Flow for a different task
flow2 = Flow(llm=llm, max_loops=1, dashboard=False)


# Add tasks to the workflow
summary_agent = flow1.run(summary_agent_prompt(IMAGE_OF_FINANCIAL_DOC_URL))

# Suppose the next task takes the output of the first task as input
out = flow2.run(
    f"Provide an actionable step by step plan on how to cut costs from the analyzed financial document. {summary_agent}"
)
