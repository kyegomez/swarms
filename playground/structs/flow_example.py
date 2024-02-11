from swarms.models import OpenAIChat
from swarms.structs import Agent

api_key = ""

# Initialize the language model, this model can be swapped out with Anthropic, ETC, Huggingface Models like Mistral, ETC
llm = OpenAIChat(
    # model_name="gpt-4"
    openai_api_key=api_key,
    temperature=0.5,
    # max_tokens=100,
)

## Initialize the workflow
agent = Agent(
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

# out = agent.load_state("flow_state.json")
# temp = agent.dynamic_temperature()
# filter = agent.add_response_filter("Trump")
out = agent.run("Generate a 10,000 word blog on health and wellness.")
# out = agent.validate_response(out)
# out = agent.analyze_feedback(out)
# out = agent.print_history_and_memory()
# # out = agent.save_state("flow_state.json")
# print(out)
