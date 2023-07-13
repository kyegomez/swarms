from swarms.tools.agent_tools import *
from swarms.agents.workers.WorkerNode import WorkerNode
from swarms.agents.boss.BossNode import BossNode

# from swarms.agents.workers.WorkerNode import worker_tool
from swarms.agents.workers.WorkerNode import worker_node
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Swarms:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def initialize_llm(self, llm_class, temperature=0.5):
        # Initialize language model
        return llm_class(openai_api_key=self.openai_api_key, temperature=temperature)

    def initialize_tools(self, llm_class):
        llm = self.initialize_llm(llm_class)
        # Initialize tools
        web_search = DuckDuckGoSearchRun()
        tools = [
            web_search,
            WriteFileTool(root_dir=ROOT_DIR),
            ReadFileTool(root_dir=ROOT_DIR),

            process_csv,
            WebpageQATool(qa_chain=load_qa_with_sources_chain(llm)),

        ]
        assert tools is not None, "tools is not initialized"
        return tools

    def initialize_vectorstore(self):
        # Initialize vector store
        embeddings_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        return FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    def initialize_worker_node(self, worker_tools, vectorstore):
        # Initialize worker node
        llm = self.initialize_llm(ChatOpenAI)
        worker_node = WorkerNode(llm=llm, tools=worker_tools, vectorstore=vectorstore)
        worker_node.create_agent(ai_name="Swarm Worker AI Assistant", ai_role="Assistant", human_in_the_loop=False, search_kwargs={})
        worker_node_tool = Tool(name="WorkerNode AI Agent", func=worker_node.run, description="Input: an objective with a todo list for that objective. Output: your task completed: Please be very clear what the objective and task instructions are. The Swarm worker agent is Useful for when you need to spawn an autonomous agent instance as a worker to accomplish any complex tasks, it can search the internet or write code or spawn child multi-modality models to process and generate images and text or audio and so on")
        return worker_node_tool

    def initialize_boss_node(self, vectorstore, worker_node):
        # Initialize boss node
        llm = self.initialize_llm(OpenAI)
        todo_prompt = PromptTemplate.from_template("You are a boss planer in a swarm who is an expert at coming up with a todo list for a given objective and then creating an worker to help you accomplish your task. Come up with a todo list for this objective: {objective} and then spawn a worker agent to complete the task for you. Always spawn an worker agent after creating a plan and pass the objective and plan to the worker agent.")
        todo_chain = LLMChain(llm=llm, prompt=todo_prompt)
        tools = [
            Tool(name="TODO", func=todo_chain.run, description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!"),
            worker_node
        ]
        suffix = """Question: {task}\n{agent_scratchpad}"""
        prefix = """You are an Boss in a swarm who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}.\n """
        prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix, suffix=suffix, input_variables=["objective", "task", "context", "agent_scratchpad"],)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=[tool.name for tool in tools])
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
        # return BossNode(return BossNode(llm, vectorstore, agent_executor, max_iterations=5)
        return BossNode(llm, vectorstore, agent_executor, max_iterations=5)


    def run_swarms(self, objective):
        try:
            # Run the swarm with the given objective
            worker_tools = self.initialize_tools(OpenAI)
            assert worker_tools is not None, "worker_tools is not initialized"

            vectorstore = self.initialize_vectorstore()
            worker_node = self.initialize_worker_node(worker_tools, vectorstore)

            boss_node = self.initialize_boss_node(vectorstore, worker_node)

            task = boss_node.create_task(objective)
            return boss_node.execute_task(task)
        except Exception as e:
            logging.error(f"An error occurred in run_swarms: {e}")
            raise


# usage
def swarm(api_key, objective):
    """
    import swarm
    api_key = "APIKEY"
    objective = "What is the capital of the UK?"
    result = swarm(api_key, objective)
    print(result)  # Prints: "The capital of the UK is London."
    """
    swarms = Swarms(api_key)
    return swarms.run_swarms(objective)
