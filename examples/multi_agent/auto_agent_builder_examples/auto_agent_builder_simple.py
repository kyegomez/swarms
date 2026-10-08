import json

from dotenv import load_dotenv

from swarms import AutoAgentBuilder

load_dotenv()

TASK = (
    "Analyze why a B2B SaaS company's customer churn increased last "
    "quarter, and write a short brief for the leadership team."
)

builder = AutoAgentBuilder(
    model_name="gpt-5.4",
    max_agents=5,
    return_dict=True,
    agent_kwargs={"reasoning_effort": None, "max_tokens": 45_000},
)


configs = builder.run(TASK)


print(json.dumps(configs, indent=4))
