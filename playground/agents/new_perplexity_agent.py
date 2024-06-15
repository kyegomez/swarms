from swarms import Agent
from swarms.models.llama3_hosted import llama3Hosted
from playground.memory.chromadb_example import ChromaDB
from swarms.tools.prebuilt.bing_api import fetch_web_articles_bing_api

# Define the research system prompt
research_system_prompt = """
Research Agent LLM Prompt: Summarizing Sources and Content
Objective: Your task is to summarize the provided sources and the content within those sources. The goal is to create concise, accurate, and informative summaries that capture the key points of the original content.
Instructions:
1. Identify Key Information: ...
2. Summarize Clearly and Concisely: ...
3. Preserve Original Meaning: ...
4. Include Relevant Details: ...
5. Structure: ...
"""

# Initialize memory
memory = ChromaDB(output_dir="research_base", n_results=2)

# Initialize the LLM
llm = llama3Hosted(temperature=0.2, max_tokens=3500)

# Initialize the agent
agent = Agent(
    agent_name="Research Agent",
    system_prompt=research_system_prompt,
    llm=llm,
    max_loops="auto",
    autosave=True,
    dashboard=False,
    interactive=True,
    long_term_memory=memory,
    tools=[fetch_web_articles_bing_api],
)

# Define the task for the agent
task = "What is the impact of climate change on biodiversity?"
out = agent.run(task)
print(out)
