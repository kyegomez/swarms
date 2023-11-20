from swarms.models.nougat import Nougat
from swarms.structs import Flow
from swarms.models import OpenAIChat, Anthropic
from typing import List


# Base llms
llm1 = OpenAIChat()
llm2 = Anthropic()
nougat = Nougat()


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
        financial_document_img: str,
        financial_document_list_img: List[str] = None,
        fraud_detection_instructions: str = None,
        summary_agent_instructions: str = None,
        decision_making_support_agent_instructions: str = None,
    ):
        super().__init__()
        self.financial_document_img = financial_document_img
        self.fraud_detection_instructions = fraud_detection_instructions
        self.summary_agent_instructions = summary_agent_instructions

    def run(self):
        # Extract text from the image
        analyzed_doc = self.nougat(self.financial_document_img)

        # Detect fraud in the document
        fraud_detection_agent_output = self.fraud_detection_agent(analyzed_doc)

        # Generate an actionable summary of the document
        summary_agent_output = self.summary_agent(fraud_detection_agent_output)

        # Provide decision making support to the accountant
        decision_making_support_agent_output = self.decision_making_support_agent(
            summary_agent_output
        )

        return decision_making_support_agent_output
