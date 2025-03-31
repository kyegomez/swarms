# Swarms x Browser Use

- Import required modules like `Agent` `SequentialWorkflow`

- Configure your agents first with their model provider, name, description, role, and more!

- Set your api keys for your model provider in the `.env` file such as `OPENAI_API_KEY="sk-"` etc

- Conigure your `SequentialWorkflow`

## Install

```bash
pip3 install -U swarms 
```
--------



## Main Code


```python
from swarms import Agent, SequentialWorkflow


# Core Legal Agent Definitions with enhanced system prompts
litigation_agent = Agent(
    agent_name="Alex Johnson",  # Human name for the Litigator Agent
    system_prompt="As a Litigator, you specialize in navigating the complexities of lawsuits. Your role involves analyzing intricate facts, constructing compelling arguments, and devising effective case strategies to achieve favorable outcomes for your clients.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

corporate_agent = Agent(
    agent_name="Emily Carter",  # Human name for the Corporate Attorney Agent
    system_prompt="As a Corporate Attorney, you provide expert legal advice on business law matters. You guide clients on corporate structure, governance, compliance, and transactions, ensuring their business operations align with legal requirements.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

ip_agent = Agent(
    agent_name="Michael Smith",  # Human name for the IP Attorney Agent
    system_prompt="As an IP Attorney, your expertise lies in protecting intellectual property rights. You handle various aspects of IP law, including patents, trademarks, copyrights, and trade secrets, helping clients safeguard their innovations.",
    model_name="gpt-4o-mini",
    max_loops=1,
)


swarm = SequentialWorkflow(
    agents=[litigation_agent, corporate_agent, ip_agent],
    name="litigation-practice",
    description="Handle all aspects of litigation with a focus on thorough legal analysis and effective case management.",
)

swarm.run("Create a report on how to patent an all-new AI invention and what platforms to use and more.")
```