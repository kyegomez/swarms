import json

from swarms import Agent, GraphWorkflow

MODEL = "gpt-5.4"


def agent(name: str, prompt: str) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=prompt,
        model_name=MODEL,
        max_loops=1,
    )


workflow = GraphWorkflow()
workflow.add_node(agent("Alpha", "Name one colour. Just the word."))
workflow.add_node(agent("Beta", "Name one animal. Just the word."))
workflow.add_node(
    agent("Merge", "Combine the two inputs into one short phrase.")
)

workflow.add_edge("Alpha", "Merge")
workflow.add_edge("Beta", "Merge")

results = workflow.run(task="Begin.")

print("\n=== PER-NODE RESULTS ===")
for node, output in results.items():
    print(f"\n[{node}]\n{output}")

messages = workflow.conversation.return_messages_as_list()

print(json.dumps(messages, indent=4))
