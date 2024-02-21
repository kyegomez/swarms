from swarms import HierarchicalSwarm

swarm = HierarchicalSwarm(
    openai_api_key="key",
    model_type="openai",
    model_id="gpt-4",
    use_vectorstore=False,
    use_async=False,
    human_in_the_loop=False,
    logging_enabled=False,
)

# run the swarm with an objective
result = swarm.run("Design a new car")

# or huggingface
swarm = HierarchicalSwarm(
    model_type="huggingface",
    model_id="tiaueu/falcon",
    use_vectorstore=True,
    embedding_size=768,
    use_async=False,
    human_in_the_loop=True,
    logging_enabled=False,
)

# Run the swarm with a particular objective
result = swarm.run("Write a sci-fi short story")
