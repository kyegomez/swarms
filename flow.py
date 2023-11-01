from swarms.models import OpenAIChat
from swarms.structs import Flow

api_key = "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"


# Initialize the language model,
# This model can be swapped out with Anthropic, ETC, Huggingface Models like Mistral, ETC
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=3000,
)

# Initialize the flow
flow = Flow(llm=llm, max_loops=5, dashboard=True)

out = flow.run("Generate a 10,000 word blog on health and wellness.")
print(out)
