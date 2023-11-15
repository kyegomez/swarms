import re
from swarms.models.nougat import Nougat
from swarms.structs import Flow
from swarms.models import OpenAIChat
from swarms.models import LayoutLMDocumentQA

# # URL of the image of the financial document
IMAGE_OF_FINANCIAL_DOC_URL = "bank_statement_2.jpg"

# Example usage
api_key = ""

# Initialize the language flow
llm = OpenAIChat(
    openai_api_key=api_key,
)

# LayoutLM Document QA
pdf_analyzer = LayoutLMDocumentQA()

question = "What is the total amount of expenses?"
answer = pdf_analyzer(
    question,
    IMAGE_OF_FINANCIAL_DOC_URL,
)

# Initialize the Flow with the language flow
agent = Flow(llm=llm)
SUMMARY_AGENT_PROMPT = f"""
Generate an actionable summary of this financial document be very specific and precise, provide bulletpoints be very specific provide methods of lowering expenses: {answer}"
"""

# Add tasks to the workflow
summary_agent = agent.run(SUMMARY_AGENT_PROMPT)
print(summary_agent)
