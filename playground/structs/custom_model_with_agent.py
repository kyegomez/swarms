from swarms import Agent
from swarms.models.base_llm import AbstractLLM


class ExampleLLM(AbstractLLM):
    def __init__():
        pass

    def run(self, task: str, *args, **kwargs):
        pass


# Initialize the workflow
agent = Agent(
    llm=ExampleLLM(),
    max_loops="auto",
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    interactive=True,
)

# Run the workflow on a task
agent(
    "Generate a transcript for a youtube video on what swarms are!"
    " Output a <DONE> token when done."
)
