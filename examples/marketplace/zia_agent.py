from swarms import Agent

agent = Agent(
    model_name="gpt-4.1",
    marketplace_prompt_id="75fc0d28-b0d0-4372-bc04-824aa388b7d2",
    max_loops="auto",
    interactive=True,
    streaming_on=True,
)

response = agent.run("Hello, what can you help me with?")
print(response)
