# Agent with GPT-4o-Mini

- Add `OPENAI_API_KEY="your_key"` to your `.env` file
- Select your model like `gpt-4o-mini` or `gpt-4o`

```python
from swarms import Agent

Agent(
    agent_name="Stock-Analysis-Agent",
    model_name="gpt-4o-mini",
    max_loops="auto",
    interactive=True,
    streaming_on=True,
).run("What are 5 hft algorithms")
```