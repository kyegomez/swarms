from swarms import Flow, OpenAI
from swarms.swarms.groupchat import GroupChat, GroupChatManager

llm = OpenAI(
    openai_api_key="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",
    temperature=0.4,
    max_tokens=3000,
)

# Initialize the flow
flow1 = Flow(
    llm=llm,
    max_loops=1,
    system_prompt="Makes silly jokes",
    name='silly',
    dashboard=True,
)
flow2 = Flow(
    llm=llm,
    max_loops=1,
    system_prompt="CAN ANSWER RIDDLES",
    name='detective',
    dashboard=True,
)
flow3 = Flow(
    llm=llm,
    max_loops=1,
    system_prompt="YOU MAKE RIDDLES but DOES NOT GIVE AN answer",
    name='riddler',
    dashboard=True,
)
manager = Flow(
    llm=llm,
    max_loops=1,
    system_prompt="YOU ARE A GROUP CHAT MANAGER",
    name='manager',
    dashboard=True,
)


# Example usage:
agents = [flow1, flow2, flow3]

group_chat = GroupChat(agents=agents, messages=[], max_round=10)
chat_manager = GroupChatManager(groupchat=group_chat, selector = manager)
chat_history = chat_manager("Write me a riddle")