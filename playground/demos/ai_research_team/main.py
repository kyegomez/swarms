import os

from dotenv import load_dotenv

from swarms.models import Anthropic, OpenAIChat
from swarms.prompts.ai_research_team import (
    PAPER_IMPLEMENTOR_AGENT_PROMPT,
    PAPER_SUMMARY_ANALYZER,
)
from swarms.structs import Agent
from swarms.utils.pdf_to_text import pdf_to_text

# Base llms
# Environment variables
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

PDF_PATH = "fasterffn.pdf"


# Base llms
llm1 = OpenAIChat(
    openai_api_key=openai_api_key,
)

llm2 = Anthropic(
    anthropic_api_key=anthropic_api_key,
)

# Agents
paper_summarizer_agent = Agent(
    llm=llm2,
    sop=PAPER_SUMMARY_ANALYZER,
    max_loops=1,
    autosave=True,
    saved_state_path="paper_summarizer.json",
)

paper_implementor_agent = Agent(
    llm=llm1,
    sop=PAPER_IMPLEMENTOR_AGENT_PROMPT,
    max_loops=1,
    autosave=True,
    saved_state_path="paper_implementor.json",
    code_interpreter=False,
)

paper = pdf_to_text(PDF_PATH)
algorithmic_psuedocode_agent = paper_summarizer_agent.run(
    "Focus on creating the algorithmic pseudocode for the novel"
    f" method in this paper: {paper}"
)
pytorch_code = paper_implementor_agent.run(
    algorithmic_psuedocode_agent
)
