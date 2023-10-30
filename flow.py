from swarms.models import OpenAIChat
from swarms.structs import Flow

api_key = "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"


# Initialize the language model,
# This model can be swapped out with Anthropic, ETC, Huggingface Models like Mistral, ETC
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
)

# Initialize the flow
flow = Flow(
    llm=llm,
    max_loops=5,
)

out = flow.run("Generate a 10,000 word blog on health and wellness.")
print(out)

