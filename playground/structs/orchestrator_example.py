from swarms import Worker, Orchestrator

api_key = os.getenv("OPENAI_API_KEY")
org_id = os.getenv("OPENAI_ORG_ID")

node = Worker(
    openai_api_key=api_key,
    openai_org_id=org_id,
    ai_name="Optimus Prime",
)


# Instantiate the Orchestrator with 10 agents
orchestrator = Orchestrator(
    node, agent_list=[node] * 10, task_queue=[]
)

# Agent 7 sends a message to Agent 9
orchestrator.chat(
    sender_id=7,
    receiver_id=9,
    message="Can you help me with this task?",
)
