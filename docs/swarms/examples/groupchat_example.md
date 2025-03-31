# Groupchat Example

- Import required modules

- Configure your agents first

- Set your api keys for your model provider in the `.env` file such as `OPENAI_API_KEY="sk-"`

- Conigure `GroupChat` with it's various settings


## Install
```bash
pip install swarms
```

---------

## Main Code

```python
from dotenv import load_dotenv
import os

from swarms import Agent, GroupChat

if __name__ == "__main__":

    load_dotenv()

    # Get the OpenAI API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # Example agents
    agent1 = Agent(
        agent_name="Expense-Analysis-Agent",
        description="You are an accounting agent specializing in analyzing potential expenses.",
        model_name="gpt-4o-mini",
        max_loops=1,
        autosave=False,
        dashboard=False,
        verbose=True,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        retry_attempts=1,
        context_length=200000,
        output_type="string",
        streaming_on=False,
        max_tokens=15000,
    )

    agent2 = Agent(
        agent_name="Budget-Adviser-Agent",
        description="You are a budget adviser who provides insights on managing and optimizing expenses.",
        model_name="gpt-4o-mini",
        max_loops=1,
        autosave=False,
        dashboard=False,
        verbose=True,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        retry_attempts=1,
        context_length=200000,
        output_type="string",
        streaming_on=False,
        max_tokens=15000,
    )

    agents = [agent1, agent2]

    chat = GroupChat(
        name="Expense Advisory",
        description="Accounting group focused on discussing potential expenses",
        agents=agents,
        max_loops=1,
        output_type="all",
    )

    history = chat.run(
        "What potential expenses should we consider for the upcoming quarter? Please collaborate to outline a comprehensive list."
    )
```