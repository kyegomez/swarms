from swarms import Agent, GroupChat

# Create agents with organized configuration
afu = Agent(
    agent_name="afu", system_prompt="...", model_name="gpt-4.1"
)

nunu = Agent(
    agent_name="nunu", system_prompt="...", model_name="gpt-4.1"
)

# Set up the group chat
chat = GroupChat(
    agents=[afu, nunu],
    speaker_function="random-dynamic-speaker",
    interactive=False,
)

# Run a prompt through the group chat
response = chat.run(
    "@afu Write a Xiaohongshu lipstick advertisement copy"
)
print(response)
