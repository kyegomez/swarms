import os

from dotenv import load_dotenv

from swarms.models import Anthropic, OpenAIChat
from swarms.prompts.ai_research_team import (
    PAPER_IMPLEMENTOR_AGENT_PROMPT,
    PAPER_SUMMARY_ANALYZER,
)
<<<<<<< HEAD
from swarms.structs import Agent
=======
from swarms.structs import Flow
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4
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
<<<<<<< HEAD
paper_summarizer_agent = Agent(
=======
paper_summarizer_agent = Flow(
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4
    llm=llm2,
    sop=PAPER_SUMMARY_ANALYZER,
    max_loops=1,
    autosave=True,
    saved_state_path="paper_summarizer.json",
)

<<<<<<< HEAD
paper_implementor_agent = Agent(
=======
paper_implementor_agent = Flow(
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4
    llm=llm1,
    sop=PAPER_IMPLEMENTOR_AGENT_PROMPT,
    max_loops=1,
    autosave=True,
    saved_state_path="paper_implementor.json",
    code_interpreter=False,
)

paper = pdf_to_text(PDF_PATH)
algorithmic_psuedocode_agent = paper_summarizer_agent.run(
    "Focus on creating the algorithmic pseudocode for the novel method in this"
    f" paper: {paper}"
)
pytorch_code = paper_implementor_agent.run(algorithmic_psuedocode_agent)
