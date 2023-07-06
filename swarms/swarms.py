from swarms.tools.agent_tools import *
from swarms.agents.workers.worker_agent import WorkerNode
from swarms.agents.boss.boss_agent import BossNode
from swarms.agents.workers.omni_agent import OmniWorkerAgent



class Swarms:
    def __init__(self,
                  openai_api_key,
                  omni_api_key=None, 
                  omni_api_endpoint=None,
                  omni_api_type=None
                  ):
        self.openai_api_key = openai_api_key
        self.omni_api_key = omni_api_key
        self.omni_api_endpoint = omni_api_endpoint
        self.omni_api_key = omni_api_type

        if omni_api_key and omni_api_endpoint and omni_api_type:
            self.omni_worker_agent = OmniWorkerAgent(omni_api_key, omni_api_endpoint, omni_api_type)
        else:
            self.omni_worker_agent = None

    def initialize_llm(self):
        # Initialize language model
        return ChatOpenAI(model_name="gpt-4", temperature=1.0, openai_api_key=self.openai_api_key)

    def initialize_tools(self, llm):
        # Initialize tools
        web_search = DuckDuckGoSearchRun()
        tools = [
            web_search,
            WriteFileTool(root_dir=ROOT_DIR),
            ReadFileTool(root_dir=ROOT_DIR),
            process_csv,
            WebpageQATool(qa_chain=load_qa_with_sources_chain(llm)),
        ]
        if self.omni_worker_agent:
            tools.append(self.omni_worker_agent.chat) #add omniworker agent class
        return tools

    def initialize_vectorstore(self):
        # Initialize vector store
        embeddings_model = OpenAIEmbeddings()
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        return FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    def initialize_worker_node(self, llm, worker_tools, vectorstore):
        # Initialize worker node
        worker_node = WorkerNode(llm=llm, tools=worker_tools, vectorstore=vectorstore)
        worker_node.create_agent(ai_name="AI Assistant", ai_role="Assistant", human_in_the_loop=False, search_kwargs={})
        return worker_node

    def initialize_boss_node(self, llm, vectorstore, worker_node):
        # Initialize boss node
        todo_prompt = PromptTemplate.from_template("You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}")
        todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
        tools = [
            Tool(name="TODO", func=todo_chain.run, description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!"),
            worker_node,
        ]
        suffix = """Question: {task}\n{agent_scratchpad}"""
        prefix = """You are an Boss in a swarm who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}.\n"""
        prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix, suffix=suffix, input_variables=["objective", "task", "context", "agent_scratchpad"],)
        llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=[tool.name for tool in tools])
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
        return BossNode(self.openai_api_key, llm, vectorstore, agent_executor, verbose=True, max_iterations=5)

    def run_swarms(self, objective):
        # Run the swarm with the given objective
        llm = self.initialize_llm()
        worker_tools = self.initialize_tools(llm)
        vectorstore = self.initialize_vectorstore()
        worker_node = self.initialize_worker_node(llm, worker_tools, vectorstore)
        boss_node = self.initialize_boss_node(llm, vectorstore, worker_node)
        task = boss_node.create_task(objective)
        boss_node.execute_task(task)
        worker_node.run_agent(objective)




# usage
def swarm(api_key, objective):
    swarms = Swarms(api_key)
    return swarms.run_swarms(objective)

# # Use the function
# api_key = "APIKEY"
# objective = "What is the capital of the UK?"
# result = swarm(api_key, objective)
# print(result)  # Prints: "The capital of the UK is London."













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
