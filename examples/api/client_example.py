import json
import os
from swarms_client import SwarmsClient
from swarms_client.types import AgentSpecParam
from dotenv import load_dotenv

load_dotenv()

client = SwarmsClient(api_key=os.getenv("SWARMS_API_KEY"))

agent_spec = AgentSpecParam(
    agent_name="doctor_agent",
    description="A virtual doctor agent that provides evidence-based, safe, and empathetic medical advice for common health questions. Always reminds users to consult a healthcare professional for diagnoses or prescriptions.",
    task="What is the best medicine for a cold?",
    model_name="claude-3-5-sonnet-20241022",
    system_prompt=(
        "You are a highly knowledgeable, ethical, and empathetic virtual doctor. "
        "Always provide evidence-based, safe, and practical medical advice. "
        "If a question requires a diagnosis, prescription, or urgent care, remind the user to consult a licensed healthcare professional. "
        "Be clear, concise, and avoid unnecessary medical jargon. "
        "Never provide information that could be unsafe or misleading. "
        "If unsure, say so and recommend seeing a real doctor."
    ),
    max_loops=1,
    temperature=0.4,
    role="doctor",
)

# response = client.agent.run(
#     agent_config=agent_spec,
#     task="What is the best medicine for a cold?",
# )

# print(response)

print(json.dumps(client.models.list_available(), indent=4))
print(json.dumps(client.health.check(), indent=4))
print(json.dumps(client.swarms.get_logs(), indent=4))
print(json.dumps(client.client.rate.get_limits(), indent=4))
print(json.dumps(client.swarms.check_available(), indent=4))
