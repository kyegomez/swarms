# One-to-One Examples

Two agents exchange messages on a shared conversation. The sender answers the task, the receiver replies to the sender, and the exchange repeats `max_loops` times. Each agent reads the conversation as typed turns, so it can tell its own earlier messages from the other agent's.

- [one_to_one_class_example.py](one_to_one_class_example.py) - `OneToOne`, a reusable pair that can run many tasks (OpenAI `gpt-5.4`)
- [one_to_one_function_example.py](one_to_one_function_example.py) - `one_to_one()`, the functional form for a single exchange (Anthropic `claude-sonnet-4-6`)

```python
from swarms import Agent, OneToOne, one_to_one

pair = OneToOne(sender=writer, receiver=editor, output_type="dict")
history = pair.run("Write a tagline.", max_loops=2)

# or, for one exchange
history = one_to_one(writer, editor, "Write a tagline.")
```

Both return the conversation history in the format named by `output_type`. See `swarms/structs/one_to_one.py`.

Each example names its model with a plain LiteLLM string; swap it for any provider you have a key for. OpenRouter models need `OPENROUTER_API_KEY`.
