import os

from dotenv import load_dotenv

from swarms.models import Anthropic, OpenAIChat
from swarms.prompts.ai_research_team import (
    PAPER_IMPLEMENTOR_AGENT_PROMPT,
    PAPER_SUMMARY_ANALYZER,
)
from swarms.structs import Flow
from swarms.utils.pdf_to_text import pdf_to_text

# Base llms
# Environment variables
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

PDF_PATH = "shallowfeedforward.pdf"


# Base llms
llm1 = OpenAIChat(
    openai_api_key=openai_api_key,
)

llm2 = Anthropic(
    anthropic_api_key=anthropic_api_key,
)

# Agents
paper_summarizer_agent = Flow(
    llm=llm2,
    sop=PAPER_SUMMARY_ANALYZER,
    max_loops=1,
    autosave=True,
    saved_state_path='paper_summarizer.json'
)

paper_implementor_agent = Flow(
    llm=llm1,
    sop=PAPER_IMPLEMENTOR_AGENT_PROMPT,
    max_loops=1,
    autosave=True,
    saved_state_path='paper_implementor.json'
)

paper = pdf_to_text(PDF_PATH)
algorithmic_psuedocode_agent = paper_summarizer_agent.run(paper)
pytorch_code = paper_implementor_agent.run(algorithmic_psuedocode_agent)