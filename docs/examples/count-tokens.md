To count tokens you can use Swarms events and the `TokenCounter` util:

```python
from swarms import utils
from swarms.events import (
    StartPromptEvent, FinishPromptEvent,
)
from swarms.structures import Agent


token_counter = utils.TokenCounter()

agent = Agent(
    event_listeners={
        StartPromptEvent: [
            lambda e: token_counter.add_tokens(e.token_count)
        ],
        FinishPromptEvent: [
            lambda e: token_counter.add_tokens(e.token_count)
        ],
    }
)

agent.run("tell me about large language models")
agent.run("tell me about GPT")

print(f"total tokens: {token_counter.tokens}")

```