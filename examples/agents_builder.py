from swarms.structs.agent_builder import AgentsBuilder

example_task = "Write a blog post about the benefits of using Swarms for AI agents."

agents_builder = AgentsBuilder()

agents = agents_builder.run(example_task)

print(agents)
