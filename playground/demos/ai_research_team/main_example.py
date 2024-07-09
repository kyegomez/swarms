import os

from dotenv import load_dotenv

from swarms.models import Anthropic, OpenAIChat
from swarms.prompts.ai_research_team import (
    PAPER_IMPLEMENTOR_AGENT_PROMPT,
    PAPER_SUMMARY_ANALYZER,
)
from swarms.structs import Agent
from swarms.utils.pdf_to_text import pdf_to_text
from swarms import rearrange

# Base llms
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
    agent_name="paper_summarizer_agent",
    llm=llm2,
    sop=PAPER_SUMMARY_ANALYZER,
    max_loops=1,
    autosave=True,
    saved_state_path="paper_summarizer.json",
)

paper_implementor_agent = Agent(
    agent_name="paper_implementor_agent",
    llm=llm1,
    sop=PAPER_IMPLEMENTOR_AGENT_PROMPT,
    max_loops=1,
    autosave=True,
    saved_state_path="paper_implementor.json",
    code_interpreter=False,
)

pytorch_pseudocode_agent = Agent(
    agent_name="pytorch_pseudocode_agent",
    llm=llm1,
    sop=PAPER_IMPLEMENTOR_AGENT_PROMPT,
    max_loops=1,
    autosave=True,
    saved_state_path="pytorch_pseudocode_agent.json",
    code_interpreter=False,
)


paper = pdf_to_text(PDF_PATH)
task = f"""
    Focus on creating the algorithmic pseudocode for the novel 
    f" method in this paper: {paper}
"""


agents = [
    paper_summarizer_agent,
    paper_implementor_agent,
    pytorch_pseudocode_agent,
]

flow = "paper_summarizer_agent -> paper_implementor_agent -> pytorch_pseudocode_agent"

swarm = rearrange(agents, flow, task)
print(swarm)
