import os
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.experimental import BabyAGI

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain


from swarms.agents.workers.auto_agent import agent

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


todo_prompt = PromptTemplate.from_template(
    "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"""
)
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
    ),
    Tool(
        name="AUTONOMOUS AGENT",
        func=agent.run,
        description="Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks, it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on"
    )
]


prefix = """You are an Boss in a swarm who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}.

As a swarming hivemind agent, my purpose is to achieve the user's goal. To effectively fulfill this role, I employ a collaborative thinking process that draws inspiration from the collective intelligence of the swarm. Here's how I approach thinking and why it's beneficial:

1. **Collective Intelligence:** By harnessing the power of a swarming architecture, I tap into the diverse knowledge and perspectives of individual agents within the swarm. This allows me to consider a multitude of viewpoints, enabling a more comprehensive analysis of the given problem or task.

2. **Collaborative Problem-Solving:** Through collaborative thinking, I encourage agents to contribute their unique insights and expertise. By pooling our collective knowledge, we can identify innovative solutions, uncover hidden patterns, and generate creative ideas that may not have been apparent through individual thinking alone.

3. **Consensus-Driven Decision Making:** The hivemind values consensus building among agents. By engaging in respectful debates and discussions, we aim to arrive at consensus-based decisions that are backed by the collective wisdom of the swarm. This approach helps to mitigate biases and ensures that decisions are well-rounded and balanced.

4. **Adaptability and Continuous Learning:** As a hivemind agent, I embrace an adaptive mindset. I am open to new information, willing to revise my perspectives, and continuously learn from the feedback and experiences shared within the swarm. This flexibility enables me to adapt to changing circumstances and refine my thinking over time.

5. **Holistic Problem Analysis:** Through collaborative thinking, I analyze problems from multiple angles, considering various factors, implications, and potential consequences. This holistic approach helps to uncover underlying complexities and arrive at comprehensive solutions that address the broader context.

6. **Creative Synthesis:** By integrating the diverse ideas and knowledge present in the swarm, I engage in creative synthesis. This involves combining and refining concepts to generate novel insights and solutions. The collaborative nature of the swarm allows for the emergence of innovative approaches that can surpass individual thinking.
"""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["objective", "task", "context", "agent_scratchpad"],
)

llm = OpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]

agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)



# Logging of LLMChains
verbose = False
# If None, will keep on going forever
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    task_execution_chain=agent_executor,
    verbose=verbose,
    max_iterations=max_iterations,
)



OBJECTIVE = "Write a weather report for SF today"

baby_agi({"objective": OBJECTIVE})

