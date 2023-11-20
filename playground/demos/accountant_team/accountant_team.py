import os
from typing import List

from dotenv import load_dotenv

from swarms.models import Anthropic, OpenAIChat
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


# Prompts for each agent
SUMMARY_AGENT_PROMPT = """
    Generate an actionable summary of this financial document be very specific and precise, provide bulletpoints be very specific provide methods of lowering expenses: {answer}"
"""


# Agents
user_consultant_agent = Flow(
    llm=llm1,
)
doc_analyzer_agent = Flow(
    llm=llm1,
)
summary_generator_agent = Flow(
    llm=llm2,
)
fraud_detection_agent = Flow(
    llm=llm2,
)
decision_making_support_agent = Flow(
    llm=llm2,
)


class AccountantSwarms:
    """
    Accountant Swarms is a collection of agents that work together to help
    accountants with their work.

    Flow: analyze doc -> detect fraud -> generate summary -> decision making support

    The agents are:
    - User Consultant: Asks the user many questions
    - Document Analyzer: Extracts text from the image of the financial document
    - Fraud Detection: Detects fraud in the document
    - Summary Agent: Generates an actionable summary of the document
    - Decision Making Support: Provides decision making support to the accountant

    The agents are connected together in a workflow that is defined in the
    run method.

    The workflow is as follows:
    1. The Document Analyzer agent extracts text from the image of the
    financial document.
    2. The Fraud Detection agent detects fraud in the document.
    3. The Summary Agent generates an actionable summary of the document.
    4. The Decision Making Support agent provides decision making support
    to the accountant.

    Example:
    >>> accountant_swarms = AccountantSwarms(


    """

    def __init__(
        self,
        pdf_path: str,
        list_pdfs: List[str] = None,
        fraud_detection_instructions: str = None,
        summary_agent_instructions: str = None,
        decision_making_support_agent_instructions: str = None,
    ):
        super().__init__()
        self.pdf_path = pdf_path
        self.list_pdfs = list_pdfs
        self.fraud_detection_instructions = fraud_detection_instructions
        self.summary_agent_instructions = summary_agent_instructions
        self.decision_making_support_agent_instructions = (
            decision_making_support_agent_instructions
        )

    def run(self):
        # Transform the pdf to text
        pdf_text = pdf_to_text(self.pdf_path)

        # Detect fraud in the document
        fraud_detection_agent_output = self.fraud_detection_agent(pdf_text)

        # Generate an actionable summary of the document
        summary_agent_output = self.summary_agent(fraud_detection_agent_output)

        # Provide decision making support to the accountant
        decision_making_support_agent_output = self.decision_making_support_agent(
            summary_agent_output
        )

        return decision_making_support_agent_output
