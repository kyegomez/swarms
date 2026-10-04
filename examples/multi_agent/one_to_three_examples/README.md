# One-to-Three Examples

A fixed-shape broadcast: one sender, exactly three receivers. The sender answers the task, then each receiver reads the shared conversation and replies in turn. Any receiver count other than three raises `ValueError`, at construction for the class and at call time for the function.

- [one_to_three_class_example.py](one_to_three_class_example.py) - `OneToThree`, a reusable sender-plus-trio (Qwen 3.8 Flash via OpenRouter)
- [one_to_three_function_example.py](one_to_three_function_example.py) - `one_to_three()`, the functional form (DeepSeek V4 Flash via OpenRouter)

```python
from swarms import Agent, OneToThree, one_to_three

panel = OneToThree(sender=founder, receivers=[a, b, c], output_type="dict")
history = panel.run("Pitch the idea.")

# or, for one run
history = one_to_three(founder, [a, b, c], "Pitch the idea.")
```

Both return the conversation history in the format named by `output_type`. See `swarms/structs/one_to_three.py`.

Each example names its model with a plain LiteLLM string; swap it for any provider you have a key for. OpenRouter models need `OPENROUTER_API_KEY`.
