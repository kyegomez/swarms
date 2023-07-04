from swarms import Swarms

# Retrieve your API key from the environment or replace with your actual key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize Swarms with your API key
swarm = Swarms(api_key)

# Initialize lower level models and tools
llm = swarm.initialize_llm()
tools = swarm.initialize_tools(llm)

# Initialize vector store
vectorstore = swarm.initialize_vectorstore()

# Initialize the worker node
worker_node = swarm.initialize_worker_node(llm, tools, vectorstore)
worker_node.create_agent("AI Assistant", "Assistant", True, {})

# Define an objective
objective = "Find 20 potential customers for a Swarms based AI Agent automation infrastructure"

# Initialize the boss node
boss_node = swarm.initialize_boss_node(llm, vectorstore, agent_executor)

# Create and execute a task
task = boss_node.create_task(objective)
boss_node.execute_task(task)

# Use the worker agent to perform a task
worker_node.run_agent(objective)
