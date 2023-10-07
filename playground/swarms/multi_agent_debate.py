from swarms import Worker, MultiAgentDebate, select_speaker

# Initialize agents
worker1 = Worker(openai_api_key="", ai_name="Optimus Prime")
worker2 = Worker(openai_api_key="", ai_name="Bumblebee")
worker3 = Worker(openai_api_key="", ai_name="Megatron")

agents = [worker1, worker2, worker3]

# Initialize multi-agent debate with the selection function
debate = MultiAgentDebate(agents, select_speaker)

# Run task
task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
results = debate.run(task, max_iters=4)

# Print results
for result in results:
    print(f"Agent {result['agent']} responded: {result['response']}")
