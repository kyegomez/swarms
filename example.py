from swarms import Agent, OpenAIChat

import logging
import contextlib
from http.client import HTTPConnection

HTTPConnection.debuglevel = 1

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

for x in (
        "httpcore",
        "requests.packages.urllib3"):
    alog = logging.getLogger(x)
    alog.setLevel(logging.DEBUG)
    alog.propagate = True

    

# Initialize the agent
agent = Agent(
    agent_name="Transcript Generator",
    agent_description=(
        "Generate a transcript for a youtube video on what swarms" " are!"
    ),
    llm=OpenAIChat(),
    max_loops="auto",
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    interactive=True,
    state_save_file_type="json",
    saved_state_path="transcript_generator.json",
)

# Run the Agent on a task
out = agent.run(
    "Generate a transcript for a youtube video on what swarms are!"
)
print(out)
