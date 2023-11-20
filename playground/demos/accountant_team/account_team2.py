import os
from dotenv import load_dotenv
from swarms.models import Anthropic, OpenAIChat
from swarms.prompts.accountant_swarm_prompts import (
    DECISION_MAKING_PROMPT,
    DOC_ANALYZER_AGENT_PROMPT,
    FRAUD_DETECTION_AGENT_PROMPT,
    SUMMARY_GENERATOR_AGENT_PROMPT,
)
from swarms.structs import Flow
from swarms.utils.pdf_to_text import pdf_to_text

# Environment variables
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


# Base llms
llm1 = OpenAIChat(
    openai_api_key=openai_api_key,
)

llm2 = Anthropic(
    anthropic_api_key=anthropic_api_key,
)


# Agents
doc_analyzer_agent = Flow(
    llm=llm2,
    sop=DOC_ANALYZER_AGENT_PROMPT,
    max_loops="auto",
)
summary_generator_agent = Flow(
    llm=llm2,
    sop=SUMMARY_GENERATOR_AGENT_PROMPT,
    max_loops="auto",
)
decision_making_support_agent = Flow(
    llm=llm2,
    sop=DECISION_MAKING_PROMPT,
    max_loops="auto",
)


pdf_path="swarmdeck_a1.pdf"
fraud_detection_instructions="Detect fraud in the document"
summary_agent_instructions="Generate an actionable summary of the document"
decision_making_support_agent_instructions="Provide decision making support to the business owner:"


# Transform the pdf to text
pdf_text = pdf_to_text(pdf_path)
print(pdf_text)


# Detect fraud in the document
fraud_detection_agent_output = doc_analyzer_agent.run(
    f"{fraud_detection_instructions}: {pdf_text}"
)
print(fraud_detection_agent_output)

# Generate an actionable summary of the document
summary_agent_output = summary_generator_agent.run(
    f"{summary_agent_instructions}: {fraud_detection_agent_output}"
)
print(summary_agent_output)

# Provide decision making support to the accountant
decision_making_support_agent_output = decision_making_support_agent.run(
    f"{decision_making_support_agent_instructions}: {summary_agent_output}"
)
print(decision_making_support_agent_output)