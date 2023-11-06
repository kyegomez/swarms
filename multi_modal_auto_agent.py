from swarms.structs import Flow
from swarms.models import Idefics

# Multi Modality Auto Agent
llm = Idefics(max_length=2000)

task = "User: What is in this image? https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG"

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
out = flow.run(task)
# out = flow.validate_response(out)
# out = flow.analyze_feedback(out)
# out = flow.print_history_and_memory()
# # out = flow.save_state("flow_state.json")
# print(out)
