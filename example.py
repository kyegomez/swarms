from swarms.models import OpenAIChat
from swarms.structs import Flow

# Initialize the language model
llm = OpenAIChat(
    temperature=0.5,
)


## Initialize the workflow
flow = Flow(llm=llm, max_loops=1, dashboard=True)

# Run the workflow on a task
out = flow.run("Generate a 10,000 word blog on health and wellness.")
