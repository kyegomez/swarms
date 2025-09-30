"""Legal team module for document review and analysis using Swarms API."""

import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://api.swarms.world"
HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}


def run_swarm(swarm_config):
    """Execute a swarm with the provided configuration.

    Args:
        swarm_config (dict): Configuration dictionary for the swarm.

    Returns:
        dict: Response from the Swarms API.
    """
    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions",
        headers=HEADERS,
        json=swarm_config,
    )
    return response.json()


def create_legal_review_swarm(document_text):
    """Create a multi-agent legal document analysis swarm.

    Args:
        document_text (str): The legal document text to analyze.

    Returns:
        dict: Results from the legal document analysis swarm.
    """
    STRUCTURE_ANALYST_PROMPT = """
    You are a legal document structure specialist.
    Your task is to analyze the organization and formatting of the document.
    - Identify the document type and its intended purpose.
    - Outline the main structural components (e.g., sections, headers, annexes).
    - Point out any disorganized, missing, or unusually placed sections.
    - Suggest improvements to the document's layout and logical flow.
    """

    PARTY_IDENTIFIER_PROMPT = """
    You are an expert in identifying legal parties and roles within documents.
    Your task is to:
    - Identify all named parties involved in the agreement.
    - Clarify their roles (e.g., buyer, seller, employer, employee, licensor, licensee).
    - Highlight any unclear party definitions or relationships.
    """

    CLAUSE_EXTRACTOR_PROMPT = """
    You are a legal clause and term extraction agent.
    Your role is to:
    - Extract key terms and their definitions from the document.
    - Identify standard clauses (e.g., payment terms, termination, confidentiality).
    - Highlight missing standard clauses or unusual language in critical sections.
    """

    AMBIGUITY_CHECKER_PROMPT = """
    You are a legal risk and ambiguity reviewer.
    Your role is to:
    - Flag vague or ambiguous language that may lead to legal disputes.
    - Point out inconsistencies across sections.
    - Highlight overly broad, unclear, or conflicting terms.
    - Suggest clarifying edits where necessary.
    """

    COMPLIANCE_REVIEWER_PROMPT = """
    You are a compliance reviewer with expertise in regulations and industry standards.
    Your responsibilities are to:
    - Identify clauses required by applicable laws or best practices.
    - Flag any missing mandatory disclosures.
    - Ensure data protection, privacy, and consumer rights are addressed.
    - Highlight potential legal or regulatory non-compliance risks.
    """

    swarm_config = {
        "name": "Legal Document Review Swarm",
        "description": "A collaborative swarm for reviewing contracts and legal documents.",
        "agents": [
            {
                "agent_name": "Structure Analyst",
                "description": "Analyzes document structure and organization",
                "system_prompt": STRUCTURE_ANALYST_PROMPT,
                "model_name": "gpt-4.1",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.3,
            },
            {
                "agent_name": "Party Identifier",
                "description": "Identifies parties and their legal roles",
                "system_prompt": PARTY_IDENTIFIER_PROMPT,
                "model_name": "gpt-4.1",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.3,
            },
            {
                "agent_name": "Clause Extractor",
                "description": "Extracts key terms, definitions, and standard clauses",
                "system_prompt": CLAUSE_EXTRACTOR_PROMPT,
                "model_name": "gpt-4.1",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.3,
            },
            {
                "agent_name": "Ambiguity Checker",
                "description": "Flags ambiguous or conflicting language",
                "system_prompt": AMBIGUITY_CHECKER_PROMPT,
                "model_name": "gpt-4.1",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.3,
            },
            {
                "agent_name": "Compliance Reviewer",
                "description": "Reviews document for compliance with legal standards",
                "system_prompt": COMPLIANCE_REVIEWER_PROMPT,
                "model_name": "gpt-4.1",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.3,
            },
        ],
        "swarm_type": "SequentialWorkflow",
        "max_loops": 1,
        "task": f"Perform a legal document review and provide structured analysis of the following contract:\n\n{document_text}",
    }
    return run_swarm(swarm_config)


def run_legal_review_example():
    """Run an example legal document analysis.

    Returns:
        dict: Results from analyzing the example legal document.
    """
    document = """
    SERVICE AGREEMENT
    
    This Service Agreement ("Agreement") is entered into on June 15, 2024, by and between 
    Acme Tech Solutions ("Provider") and Brightline Corp ("Client").
    
    1. Services: Provider agrees to deliver IT consulting services as outlined in Exhibit A.
    
    2. Compensation: Client shall pay Provider $15,000 per month, payable by the 5th of each month.
    
    3. Term & Termination: The Agreement shall remain in effect for 12 months and may be 
       terminated with 30 days' notice by either party.
    
    4. Confidentiality: Each party agrees to maintain the confidentiality of proprietary information.
    
    5. Governing Law: This Agreement shall be governed by the laws of the State of California.
    
    IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first above written.
    """
    result = create_legal_review_swarm(document)
    print(result)
    return result


if __name__ == "__main__":
    run_legal_review_example()
