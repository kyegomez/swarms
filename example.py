from swarms.models import OpenAIChat
from swarms.structs import Flow

api_key = ""

# Initialize the language model, this model can be swapped out with Anthropic, ETC, Huggingface Models like Mistral, ETC
llm = OpenAIChat(
    # model_name="gpt-4"
    openai_api_key=api_key,
    temperature=0.5,
    # max_tokens=100,
)

## Initialize the workflow
flow = Flow(
    llm=llm,
    max_loops=2,
    dashboard=True,
    # stopping_condition=None,  # You can define a stopping condition as needed.
    # loop_interval=1,
    # retry_attempts=3,
    # retry_interval=1,
    # interactive=False,  # Set to 'True' for interactive mode.
    # dynamic_temperature=False,  # Set to 'True' for dynamic temperature handling.
)

# out = flow.load_state("flow_state.json")
# temp = flow.dynamic_temperature()
# filter = flow.add_response_filter("Trump")
out = flow.run("Generate a 10,000 word blog on health and wellness.")
# out = flow.validate_response(out)
# out = flow.analyze_feedback(out)
# out = flow.print_history_and_memory()
# # out = flow.save_state("flow_state.json")
# print(out)
