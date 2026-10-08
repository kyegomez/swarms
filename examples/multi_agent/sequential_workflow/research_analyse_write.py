"""
Three agents, one after another.

Each agent sees only what is new since its last turn, so context grows
linearly across loops instead of exponentially.

    python3 sequential_workflow_example.py

Needs OPENAI_API_KEY (read from .env).
"""

from dotenv import load_dotenv

from swarms import Agent, SequentialWorkflow

load_dotenv()


def agent(name, role):
    return Agent(
        agent_name=name,
        agent_description=role,
        system_prompt=f"{role} Be brief: three sentences at most.",
        model_name="gpt-5.4",
        max_loops=1,
        print_on=True,
        verbose=False,
    )


workflow = SequentialWorkflow(
    agents=[
        agent("Researcher", "You gather the key facts on a topic."),
        agent(
            "Analyst",
            "You draw out the implications of the facts you are given.",
        ),
        agent(
            "Writer",
            "You turn the analysis into a short, clear summary.",
        ),
    ],
    max_loops=1,
    multi_agent_collab_prompt=True,
)

result = workflow.run(
    "What has driven the growth of small language models in the last year?"
)

# `result` is the whole conversation: one entry per agent, in order.
for message in result:
    print(f"\n--- {message['role']} ---")
    print(message["content"])
