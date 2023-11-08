# !pip install --upgrade swarms==2.0.6



from swarms.models import OpenAIChat
from swarms.models.nougat import Nougat
from swarms.structs import Flow
from swarms.structs.sequential_workflow import SequentialWorkflow

# # URL of the image of the financial document
IMAGE_OF_FINANCIAL_DOC_URL = "bank_statement_2.jpg"

# Example usage
api_key = ""  # Your actual API key here

# Initialize the OCR model
def ocr_model(img: str):
    ocr = Nougat()
    analyze_finance_docs = ocr(img)
    return str(analyze_finance_docs)

# Initialize the language flow
llm = OpenAIChat(
    model_name="gpt-4-turbo",
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=3000,
)

# Create a prompt for the language model
def summary_agent_prompt(analyzed_doc: str):
    analyzed_doc = ocr_model(img=analyzed_doc)
    return f"""
    Generate an actionable summary of this financial document, provide bulletpoints:

    Here is the Analyzed Document:
    ---
    {analyzed_doc}
    """

# Initialize the Flow with the language flow
flow1 = Flow(llm=llm, max_loops=1, dashboard=False)

# Create another Flow for a different task
flow2 = Flow(llm=llm, max_loops=1, dashboard=False)

# Create the workflow
workflow = SequentialWorkflow(max_loops=1)

# Add tasks to the workflow
workflow.add(summary_agent_prompt(IMAGE_OF_FINANCIAL_DOC_URL), flow1)

# Suppose the next task takes the output of the first task as input
workflow.add("Provide an actionable step by step plan on how to cut costs from the analyzed financial document.", flow2)

# Run the workflow
workflow.run()

# Output the results
for task in workflow.tasks:
    print(f"Task: {task.description}, Result: {task.result}")