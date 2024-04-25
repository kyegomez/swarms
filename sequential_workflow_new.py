from swarms import Anthropic, Agent, SequentialWorkflow


# Initialize the language model agent (e.g., GPT-3)

llm = Anthropic()


# Initialize agents for individual tasks

agent1 = Agent(
    agent_name="Blog generator", llm=llm, max_loops=1, dashboard=False
)

agent2 = Agent(
    agent_name="summarizer", llm=llm, max_loops=1, dashboard=False
)


# Create the Sequential workflow

workflow = SequentialWorkflow(
    max_loops=1, objective="Create a full blog and then summarize it"
)


# Add tasks to the workflow

workflow.add(
    "Generate a 10,000 word blog on health and wellness.", agent1
)  # this task will be executed task,

workflow.add(
    "Summarize the generated blog", agent2
)  # then the next agent will accomplish this task


# Run the workflow

out = workflow.run()
print(f"{out}")
