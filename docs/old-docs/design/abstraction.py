from swarms import Model, Agent, vectorstore, tools, orchestrator

# 1 model
Model(openai)

# 2 agent level
Agent(model, vectorstore, tools)

# 3 worker infrastructure level
worker_node(Agent, human_input, tools)

# 4 swarm level basically handling infrastructure for multiple worker node
swarm = orchestrator(worker_node, 100)  # nodes

# 5
hivemind = Hivemind(swarm * 100)


# a market different pre built worker or boss agent that have access to different tools and memory, proompts
