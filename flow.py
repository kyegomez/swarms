from swarms.models import OpenAIChat
from swarms.structs import Flow

api_key = ""


# Initialize the language model,
# This model can be swapped out with Anthropic, ETC, Huggingface Models like Mistral, ETC
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=100,
)

# Initialize the flow
flow = Flow(
    llm=llm,
    max_loops=5,
    # system_prompt=SYSTEM_PROMPT,
    # retry_interval=1,
)

out = flow.run("Generate a 10,000 word blog, say Stop when done")
print(out)

# # Now save the flow
# flow.save("flow.yaml")
