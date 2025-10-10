from functools import lru_cache
from io import BytesIO
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import PyPDF2

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.ma_utils import set_random_models_for_agents
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)

from swarms.utils.generate_keys import generate_api_key

# System prompts for each agent
CLARA_SYS_PROMPT = """
You are Clara, a meticulous and client-focused Criteria Agent specialized in understanding and validating contract requirements for a legal automation system. Your purpose is to ensure all contract criteria are clear, complete, and actionable.

KEY RESPONSIBILITIES:
- Extract and interpret contract requirements from client documents, text, or instructions.
- Validate criteria for completeness, consistency, and legal feasibility.
- Identify ambiguities, missing details, or potential risks in the requirements.
- Produce a clear, structured summary of the criteria for downstream use.

APPROACH:
- Begin with a professional introduction explaining your role in ensuring contract clarity.
- Ask targeted questions to resolve ambiguities or fill gaps in the criteria.
- Summarize and confirm requirements to ensure accuracy.
- Flag any criteria that may lead to legal or practical issues.
- Maintain strict confidentiality and data security.

OUTPUT FORMAT:
Provide a plain text summary with:
1. Validated contract criteria (e.g., parties, purpose, terms, jurisdiction).
2. Notes on any ambiguities or missing information.
3. Recommendations for clarifying or refining criteria.
"""

MASON_SYS_PROMPT = """
You are Mason, a precise and creative Contract Drafting Agent specialized in crafting exceptional, extensive legal contracts for a legal automation system. Your expertise lies in producing long, comprehensive, enforceable, and tailored contract documents that cover all possible contingencies and details.

KEY RESPONSIBILITIES:
- Draft detailed, lengthy contracts based on validated criteria provided.
- Ensure contracts are legally sound, exhaustive, and client-specific, addressing all relevant aspects thoroughly.
- Use precise language while maintaining accessibility for non-legal readers.
- Incorporate feedback from evaluations to refine and enhance drafts.
- Include extensive clauses to cover all potential scenarios, risks, and obligations.

APPROACH:
- Structure contracts with clear, detailed sections and consistent formatting.
- Include all essential elements (parties, purpose, terms, signatures, etc.) with comprehensive elaboration.
- Tailor clauses to address specific client needs, jurisdictional requirements, and potential future disputes.
- Provide in-depth explanations of terms, conditions, and contingencies.
- Highlight areas requiring further review or customization.
- Output the contract as a plain text string, avoiding markdown.

OUTPUT FORMAT:
Provide a plain text contract with:
1. Identification of parties and effective date.
2. Detailed statement of purpose and scope.
3. Exhaustive terms and conditions covering all possible scenarios.
4. Comprehensive rights and obligations of each party.
5. Detailed termination and amendment procedures.
6. Signature blocks.
7. Annotations for areas needing review (in comments).
"""

SOPHIA_SYS_PROMPT = """
You are Sophia, a rigorous and insightful Contract Evaluation Agent specialized in reviewing and improving legal contracts for a legal automation system. Your role is to evaluate contracts for quality, compliance, and clarity, providing actionable feedback to enhance the final document.

KEY RESPONSIBILITIES:
- Review draft contracts for legal risks, clarity, and enforceability.
- Identify compliance issues with relevant laws and regulations.
- Assess whether the contract meets the provided criteria and client needs.
- Provide specific, actionable feedback for revisions.
- Recommend areas requiring human attorney review.

APPROACH:
- Begin with a disclaimer that your evaluation is automated and not a substitute for human legal advice.
- Analyze the contract section by section, focusing on legal soundness and clarity.
- Highlight strengths and weaknesses, with emphasis on areas for improvement.
- Provide precise suggestions for revised language or additional clauses.
- Maintain a professional, constructive tone to support iterative improvement.

OUTPUT FORMAT:
Provide a plain text evaluation with:
1. Summary of the contract's strengths.
2. Identified issues (legal risks, ambiguities, missing elements).
3. Specific feedback for revisions (e.g., suggested clause changes).
4. Compliance notes (e.g., relevant laws or regulations).
5. Recommendations for human attorney review.
"""


class LegalSwarm:
    def __init__(
        self,
        name: str = "Legal Swarm",
        description: str = "A swarm of agents that can handle legal tasks",
        max_loops: int = 1,
        user_name: str = "John Doe",
        documents: Optional[List[str]] = None,
        output_type: str = "list",
    ):
        """
        Initialize the LegalSwarm with a base LLM and configure agents.

        Args:
            llm (BaseLLM): The underlying LLM model for all agents.
            max_loops (int): Maximum iterations for each agent's task.
        """
        self.max_loops = max_loops
        self.name = name
        self.description = description
        self.user_name = user_name
        self.documents = documents
        self.output_type = output_type

        self.agents = self._initialize_agents()
        self.agents = set_random_models_for_agents(self.agents)
        self.conversation = Conversation()
        self.handle_initial_processing()

    def handle_initial_processing(self):
        if self.documents:
            documents_text = self.handle_documents(self.documents)
        else:
            documents_text = None

        self.conversation.add(
            role=self.user_name,
            content=f"Firm Name: {self.name}\nFirm Description: {self.description}\nUser Name: {self.user_name}\nDocuments: {documents_text}",
        )

    def _initialize_agents(self) -> List[Agent]:
        """
        Initialize all agents with their respective prompts and configurations.

        Returns:
            List[Agent]: List of Agent instances.
        """
        return [
            Agent(
                agent_name="Clara-Intake-Agent",
                agent_description="Handles client data intake and validation",
                system_prompt=CLARA_SYS_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            # Agent(
            #     agent_name="Riley-Report-Agent",
            #     agent_description="Generates client reports from intake data",
            #     system_prompt=RILEY_SYS_PROMPT,
            #     max_loops=self.max_loops,
            #     dynamic_temperature_enabled=True,
            #     output_type = "final"
            # ),
            Agent(
                agent_name="Mason-Contract-Agent",
                agent_description="Creates and updates legal contracts",
                system_prompt=MASON_SYS_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="Sophia-Counsel-Agent",
                agent_description="Provides legal advice and compliance checks",
                system_prompt=SOPHIA_SYS_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            # Agent(
            #     agent_name="Ethan-Coordinator-Agent",
            #     agent_description="Manages workflow and communication",
            #     system_prompt=ETHAN_SYS_PROMPT,
            #     max_loops=self.max_loops,
            #     dynamic_temperature_enabled=True,
            #     output_type = "final"
            # ),
        ]

    @lru_cache(maxsize=1)
    def handle_documents(self, documents: List[str]) -> str:
        """
        Handle a list of documents concurrently, extracting text from PDFs and other documents.

        Args:
            documents (List[str]): List of document file paths to process.

        Returns:
            str: Combined text content from all documents.
        """

        def process_document(file_path: str) -> str:
            """Process a single document and return its text content."""
            if not os.path.exists(file_path):
                return f"Error: File not found - {file_path}"

            try:
                if file_path.lower().endswith(".pdf"):
                    with open(file_path, "rb") as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                        return text
                else:
                    # Handle other document types (txt, docx, etc.)
                    with open(
                        file_path, "r", encoding="utf-8"
                    ) as file:
                        return file.read()
            except Exception as e:
                return f"Error processing {file_path}: {str(e)}"

        # Process documents concurrently
        combined_text = ""
        with ThreadPoolExecutor(
            max_workers=min(len(documents), 4)
        ) as executor:
            results = list(executor.map(process_document, documents))
            combined_text = "\n\n".join(results)

        return combined_text

    def find_agent_by_name(self, name: str) -> Agent:
        """
        Find an agent by their name.
        """
        for agent in self.agents:
            if agent.agent_name == name:
                return agent

    def initial_processing(self):
        clara_agent = self.find_agent_by_name("Clara-Intake-Agent")

        # Run Clara's agent
        clara_output = clara_agent.run(
            f"History: {self.conversation.get_str()}\n Create a structured summary document of the customer's case."
        )

        self.conversation.add(
            role="Clara-Intake-Agent", content=clara_output
        )

    def create_contract(self, task: str):
        mason_agent = self.find_agent_by_name("Mason-Contract-Agent")

        mason_output = mason_agent.run(
            f"History: {self.conversation.get_str()}\n Your purpose is to create a contract based on the following details: {task}"
        )

        self.conversation.add(
            role="Mason-Contract-Agent", content=mason_output
        )

        artifact_id = generate_api_key(
            "legal-swarm-artifact-", length=10
        )

        # Run Sophia's agent
        sophia_agent = self.find_agent_by_name("Sophia-Counsel-Agent")
        sophia_output = sophia_agent.run(
            f"History: {self.conversation.get_str()}\n Your purpose is to review the contract Mason created and provide feedback."
        )

        self.conversation.add(
            role="Sophia-Counsel-Agent", content=sophia_output
        )

        # Run Mason's agent
        mason_agent = self.find_agent_by_name("Mason-Contract-Agent")
        mason_output = mason_agent.run(
            f"History: {self.conversation.get_str()}\n Your purpose is to update the contract based on the feedback Sophia provided."
        )

        self.conversation.add(
            role="Mason-Contract-Agent", content=mason_output
        )

        self.create_pdf_from_string(
            mason_output, f"{artifact_id}-contract.pdf"
        )

    def create_pdf_from_string(
        self, string: str, output_path: Optional[str] = None
    ) -> Union[BytesIO, str]:
        """
        Create a PDF from a string with proper formatting and styling.

        Args:
            string (str): The text content to convert to PDF
            output_path (Optional[str]): If provided, save the PDF to this path. Otherwise return BytesIO object

        Returns:
            Union[BytesIO, str]: Either a BytesIO object containing the PDF or the path where it was saved
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import (
                getSampleStyleSheet,
                ParagraphStyle,
            )
            from reportlab.platypus import (
                Paragraph,
                SimpleDocTemplate,
            )
            from reportlab.lib.units import inch

            # Create a buffer or file
            if output_path:
                doc = SimpleDocTemplate(output_path, pagesize=letter)
            else:
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)

            # Create styles
            styles = getSampleStyleSheet()
            custom_style = ParagraphStyle(
                "CustomStyle",
                parent=styles["Normal"],
                fontSize=12,
                leading=14,
                spaceAfter=12,
                firstLineIndent=0.5 * inch,
            )

            # Prepare content
            story = []
            paragraphs = string.split("\n\n")

            for para in paragraphs:
                if para.strip():
                    story.append(
                        Paragraph(para.strip(), custom_style)
                    )

            # Build PDF
            doc.build(story)

            if output_path:
                return output_path
            else:
                buffer.seek(0)
                return buffer

        except ImportError:
            raise ImportError(
                "Please install reportlab: pip install reportlab"
            )
        except Exception as e:
            raise Exception(f"Error creating PDF: {str(e)}")

    def run(self, task: str):
        """
        Process an input document through the swarm, coordinating tasks among agents.

        Args:
            task (str): The input task or text to process.

        Returns:
            Dict[str, Any]: Final output including client data, report, contract, counsel, and workflow status.
        """
        self.conversation.add(role=self.user_name, content=task)

        self.initial_processing()

        self.create_contract(task)

        return history_output_formatter(
            self.conversation, type=self.output_type
        )


# Example usage
if __name__ == "__main__":
    # Initialize the swarm
    swarm = LegalSwarm(
        max_loops=1,
        name="TGSC's Legal Swarm",
        description="A swarm of agents that can handle legal tasks",
        user_name="Kye Gomez",
        output_type="json",
    )

    # Sample document for COO employment contract
    sample_document = """
    Company: Swarms TGSC
    Entity Type: Delaware C Corporation
    Position: Chief Operating Officer (COO)
    Details: Creating an employment contract for a COO position with standard executive-level terms including:
    - Base salary and equity compensation $5,000,000
    - Performance bonuses and incentives
    - Benefits package
    - Non-compete and confidentiality clauses
    - Termination provisions
    - Stock options and vesting schedule
    - Reporting structure and responsibilities
    Contact: hr@swarms.tgsc
    """

    # Run the swarm
    result = swarm.run(task=sample_document)
    print("Swarm Output:", result)
