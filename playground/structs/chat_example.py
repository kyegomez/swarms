from swarms import Orchestrator, Worker

# Instantiate the Orchestrator with 10 agents
orchestrator = Orchestrator(
    Worker, agent_list=[Worker] * 10, task_queue=[]
)

# Agent 1 sends a message to Agent 2
orchestrator.chat(
    sender_id=1, receiver_id=2, message="Hello, Agent 2!"
)
