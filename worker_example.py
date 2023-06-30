from swarms import WorkerNode, tools, vectorstore, llm
#define tools (assuming process csv)



#inti worker node with llm
worker_node = WorkerNode(llm=llm, tools=tools, vectorstore=vectorstore)

#create an agent within the worker node
worker_node.create_agent(ai_name="AI Assistant", ai_role="Assistant", human_in_the_loop=True, search_kwargs={})

#use the agent to perform a task
worker_node.run_agent("Find 20 potential customers for a Swarms based AI Agent automation infrastructure")

