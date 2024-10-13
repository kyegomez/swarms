import os
from swarm_models import Anthropic
from dotenv import load_dotenv


load_dotenv()


model = Anthropic(
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.1,
)

model.run(
    "Where is the best state to open up a c corp with the lowest taxes"
)
