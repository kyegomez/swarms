import json

from swarms import Agent, SequentialWorkflow

MODEL = "gpt-5.4"

workflow = SequentialWorkflow(
    agents=[
        Agent(
            agent_name="Researcher",
            system_prompt="Give one fact. One sentence.",
            model_name=MODEL,
            max_loops=1,
        ),
        Agent(
            agent_name="Analyst",
            system_prompt="Analyse the fact you were given. One sentence.",
            model_name=MODEL,
            max_loops=1,
        ),
        Agent(
            agent_name="Writer",
            system_prompt="Summarise the discussion so far. One sentence.",
            model_name=MODEL,
            max_loops=1,
        ),
    ],
    max_loops=1,
)

workflow.run("Why did central banks raise rates in 2022?")

messages = (
    workflow.agent_rearrange.conversation.return_messages_as_list()
)

print(json.dumps(messages, indent=4))
