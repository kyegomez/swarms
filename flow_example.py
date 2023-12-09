import os
from swarms.models import OpenAIChat
from swarms.structs.flow import Flow, stop_when_repeats
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAIChat model
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAIChat(openai_api_key=openai_api_key)

# Initialize the Flow
flow = Flow(llm=llm, max_loops=3, stopping_condition=stop_when_repeats)

# Run the Flow with a task
response = flow.run("")
print(response)
