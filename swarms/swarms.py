# ---------- Dependencies ----------
import os
import asyncio
import faiss
from typing import Optional
from contextlib import contextmanager

from pydantic import BaseModel, Field
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains.base import Chain

from langchain.experimental import BabyAGI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import FAISS

from langchain.docstore import InMemoryDocstore
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.tools.file_management.read import ReadFileTool

from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.human.tool import HumanInputRun
from swarms.tools import Terminal, CodeWriter, CodeEditor, process_csv, WebpageQATool

from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool


# ---------- Constants ----------
ROOT_DIR = "./data/"

# ---------- Tools ----------
openai_api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model_name="gpt-4", temperature=1.0, openai_api_key=openai_api_key)

tools = [
    Tool(name='web_search', func=DuckDuckGoSearchRun(), description='Runs a web search'),
    Tool(name='write_file_tool', func=WriteFileTool(root_dir=ROOT_DIR), description='Writes a file'),
    Tool(name='read_file_tool', func=ReadFileTool(root_dir=ROOT_DIR), description='Reads a file'),
    Tool(name='process_csv', func=process_csv, description='Processes a CSV file'),
    
    Tool(name='query_website_tool', func=WebpageQATool(qa_chain=load_qa_with_sources_chain(llm)), description='Queries a website'),
    
    # Tool(name='terminal', func=Terminal.execute, description='Operates a terminal'),
    # Tool(name='code_writer', func=CodeWriter(), description='Writes code'),
    # Tool(name='code_editor', func=CodeEditor(), description='Edits code'),
]

# ---------- Vector Store ----------
embeddings_model = OpenAIEmbeddings()
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# ---------- Worker Node ----------
@tool
class WorkerNode:
    def __init__(self, llm, tools, vectorstore):
        self.llm = llm
        self.tools = tools
        self.vectorstore = vectorstore

    def create_agent(self, ai_name, ai_role, human_in_the_loop, search_kwargs):
        self.agent = AutoGPT.from_llm_and_tools(
            ai_name=ai_name,
            ai_role=ai_role,
            tools=tools,
            llm=self.llm,
            memory=self.vectorstore.as_retriever(search_kwargs=search_kwargs),
            human_in_the_loop=human_in_the_loop,
        )
        self.agent.chain.verbose = True

    def run_agent(self, prompt):
        tree_of_thoughts_prompt = """
        Imagine three different experts are answering this question. All experts will write down each chain of thought of each step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. The question is...
        """
        self.agent.run([f"{tree_of_thoughts_prompt} {prompt}"])

worker_node = WorkerNode(llm=llm, tools=tools, vectorstore=vectorstore)

# ---------- Boss Node ----------
class BossNode:
    def __init__(self, llm, vectorstore, task_execution_chain, verbose, max_iterations):
        self.llm = llm
        self.vectorstore = vectorstore
        self.task_execution_chain = task_execution_chain
        self.verbose = verbose
        self.max_iterations = max_iterations

        self.baby_agi = BabyAGI.from_llm(
            llm=self.llm,
            vectorstore=self.vectorstore,
            task_execution_chain=self.task_execution_chain
        )

    def create_task(self, objective):
        return {"objective": objective}

    def execute_task(self, task):
        self.baby_agi(task)

# ---------- Inputs to Boss Node ----------
todo_prompt = PromptTemplate.from_template(
    "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"""
)
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)

tools += [
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
    ),
    Tool(
        name="AUTONOMOUS Worker AGENT",
        func=worker_node.run_agent,
        description="Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks, it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on"
    )
]

suffix = """Question: {task}
{agent_scratchpad}"""
prefix = """You are an Boss in a swarm who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}.
"""

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
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# boss_node = BossNode(llm=llm, vectorstore=vectorstore, task_execution_chain=agent_executor, verbose=True, max_iterations=5)






class Swarms:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def initialize_llm(self):
        return ChatOpenAI(model_name="gpt-4", temperature=1.0, openai_api_key=self.openai_api_key)

    def initialize_tools(self, llm):
        web_search = DuckDuckGoSearchRun()
        tools = [
            Tool(name='web_search', func=DuckDuckGoSearchRun(), description='Runs a web search'),
            Tool(name='write_file_tool', func=WriteFileTool(root_dir=ROOT_DIR), description='Writes a file'),
            Tool(name='read_file_tool', func=ReadFileTool(root_dir=ROOT_DIR), description='Reads a file'),
            Tool(name='process_csv', func=process_csv, description='Processes a CSV file'),
            Tool(name='query_website_tool', func=WebpageQATool(qa_chain=load_qa_with_sources_chain(llm)), description='Queries a website'),
            # Tool(name='terminal', func=Terminal.execute, description='Operates a terminal'),
            # Tool(name='code_writer', func=CodeWriter(), description='Writes code'),
            # Tool(name='code_editor', func=CodeEditor(), description='Edits code'),
        ]
        return tools

    def initialize_vectorstore(self):
        embeddings_model = OpenAIEmbeddings()
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        return FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    def initialize_worker_node(self, llm, tools, vectorstore):
        worker_node = WorkerNode(llm=llm, tools=tools, vectorstore=vectorstore)
        worker_node.create_agent(ai_name="AI Assistant", ai_role="Assistant", human_in_the_loop=True, search_kwargs={})
        return worker_node

    def initialize_boss_node(self, llm, vectorstore):
        todo_prompt = PromptTemplate.from_template("You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}""")
        todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
        # search = SerpAPIWrapper()
        tools = [
            # Tool(name="Search", func=search.run, description="useful for when you need to answer questions about current events"),
            Tool(name="TODO", func=todo_chain.run, description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!"),
            Tool(name="AUTONOMOUS Worker AGENT", func=self.worker_node.run_agent, description="Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks, it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on")
        ]

        suffix = """Question: {task}\n{agent_scratchpad}"""
        prefix = """You are an Boss in a swarm who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}.\n"""
        prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix, suffix=suffix, input_variables=["objective", "task", "context", "agent_scratchpad"],)

        llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=[tool.name for tool in tools])
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
        return BossNode(self.openai_api_key, llm, vectorstore, agent_executor, verbose=True, max_iterations=5)

    def run_swarms(self, objective):
        llm = self.initialize_llm()
        tools = self.initialize_tools(llm)
        vectorstore = self.initialize_vectorstore()
        worker_node = self.initialize_worker_node(llm, tools, vectorstore)
        boss_node = self.initialize_boss_node(llm, vectorstore)
        task = boss_node.create_task(objective)
        boss_node.execute_task(task)
        worker_node.run_agent(objective)









# class Swarms:
#     def __init__(self, num_nodes: int, llm: BaseLLM, self_scaling: bool): 
#         self.nodes = [WorkerNode(llm) for _ in range(num_nodes)]
#         self.self_scaling = self_scaling
    
#     def add_worker(self, llm: BaseLLM):
#         self.nodes.append(WorkerNode(llm))

#     def remove_workers(self, index: int):
#         self.nodes.pop(index)

#     def execute(self, task):
#         #placeholer for main execution logic
#         pass

#     def scale(self):
#         #placeholder for self scaling logic
#         pass



#special classes

# class HierarchicalSwarms(Swarms):
#     def execute(self, task):
#         pass


# class CollaborativeSwarms(Swarms):
#     def execute(self, task):
#         pass

# class CompetitiveSwarms(Swarms):
#     def execute(self, task):
#         pass

# class MultiAgentDebate(Swarms):
#     def execute(self, task):
#         pass


#======================================> WorkerNode


# class MetaWorkerNode:
#     def __init__(self, llm, tools, vectorstore):
#         self.llm = llm
#         self.tools = tools
#         self.vectorstore = vectorstore
        
#         self.agent = None
#         self.meta_chain = None

#     def init_chain(self, instructions):
#         self.agent = WorkerNode(self.llm, self.tools, self.vectorstore)
#         self.agent.create_agent("Assistant", "Assistant Role", False, {})

#     def initialize_meta_chain():
#         meta_template = """
#         Assistant has just had the below interactions with a User. Assistant followed their "Instructions" closely. Your job is to critique the Assistant's performance and then revise the Instructions so that Assistant would quickly and correctly respond in the future.

#         ####

#         {chat_history}

#         ####

#         Please reflect on these interactions.

#         You should first critique Assistant's performance. What could Assistant have done better? What should the Assistant remember about this user? Are there things this user always wants? Indicate this with "Critique: ...".

#         You should next revise the Instructions so that Assistant would quickly and correctly respond in the future. Assistant's goal is to satisfy the user in as few interactions as possible. Assistant will only see the new Instructions, not the interaction history, so anything important must be summarized in the Instructions. Don't forget any important details in the current Instructions! Indicate the new Instructions by "Instructions: ...".
#         """

#         meta_prompt = PromptTemplate(
#             input_variables=["chat_history"], template=meta_template
#         )

#         meta_chain = LLMChain(
#             llm=OpenAI(temperature=0),
#             prompt=meta_prompt,
#             verbose=True,
#         )
#         return meta_chain

#     def meta_chain(self):
#         #define meta template and meta prompting as per your needs
#         self.meta_chain = initialize_meta_chain()


#     def get_chat_history(chain_memory):
#         memory_key = chain_memory.memory_key
#         chat_history = chain_memory.load_memory_variables(memory_key)[memory_key]
#         return chat_history


#     def get_new_instructions(meta_output):
#         delimiter = "Instructions: "
#         new_instructions = meta_output[meta_output.find(delimiter) + len(delimiter) :]
#         return new_instructions


#     def main(self, task, max_iters=3, max_meta_iters=5):
#         failed_phrase = "task failed"
#         success_phrase = "task succeeded"
#         key_phrases = [success_phrase, failed_phrase]

#         instructions = "None"
#         for i in range(max_meta_iters):
#             print(f"[Episode {i+1}/{max_meta_iters}]")
#             self.initialize_chain(instructions)
#             output = self.agent.perform('Assistant', {'request': task})
#             for j in range(max_iters):
#                 print(f"(Step {j+1}/{max_iters})")
#                 print(f"Assistant: {output}")
#                 print(f"Human: ")
#                 human_input = input()
#                 if any(phrase in human_input.lower() for phrase in key_phrases):
#                     break
#                 output = self.agent.perform('Assistant', {'request': human_input})
#             if success_phrase in human_input.lower():
#                 print(f"You succeeded! Thanks for playing!")
#                 return
#             self.initialize_meta_chain()
#             meta_output = self.meta_chain.predict(chat_history=self.get_chat_history())
#             print(f"Feedback: {meta_output}")
#             instructions = self.get_new_instructions(meta_output)
#             print(f"New Instructions: {instructions}")
#             print("\n" + "#" * 80 + "\n")
#         print(f"You failed! Thanks for playing!")


# #init instance of MetaWorkerNode
# meta_worker_node = MetaWorkerNode(llm=OpenAI, tools=tools, vectorstore=vectorstore)


# #specify a task and interact with the agent
# task = "Provide a sysmatic argument for why we should always eat past with olives"
# meta_worker_node.main(task)


####################################################################### => Boss Node
####################################################################### => Boss Node
####################################################################### => Boss Node
