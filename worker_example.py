from swarms import worker_node

#cretae an agent 
worker_node.create_agent(ai_name="Workerx", ai_role="Assistant", human_in_the_loop=True, search_kwargs={"k": 8})

#use the agent to perform a task
worker_node.run_agent("Find 20 potential customers for a Swarms based AI Agent automation infrastructure")
