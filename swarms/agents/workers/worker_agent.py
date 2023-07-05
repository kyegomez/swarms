

# ---------- Worker Node ----------
# Define the input schema for the WorkerNode
class WorkerNodeInput(BaseModel):
    llm: Any = Field(description="Language model")
    tools: List[Tool] = Field(description="List of tools")
    vectorstore: VectorStore = Field(description="Vector store")

@tool("WorkerNode", args_schema=WorkerNodeInput)
class WorkerNode:
    """Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks, it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on """
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
