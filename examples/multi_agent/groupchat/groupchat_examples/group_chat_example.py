"""Investment advisory dynamic GroupChat: financial analyst + tax adviser."""

from dotenv import load_dotenv

from swarms import Agent
from swarms.structs.groupchat import GroupChat, RESPOND_TOOL


if __name__ == "__main__":

    load_dotenv()

    agent1 = Agent(
        agent_name="Financial-Analysis-Agent",
        system_prompt="You are a friendly financial analyst specializing in investment strategies. Be approachable and conversational.",
        model_name="groq/llama-3.1-70b-versatile",
        max_loops=1,
        persistent_memory=False,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        tools_list_dictionary=[RESPOND_TOOL],
    )

    agent2 = Agent(
        agent_name="Tax-Adviser-Agent",
        system_prompt="You are a tax adviser who provides clear, concise, and approachable guidance on tax-related queries.",
        model_name="groq/llama-3.1-70b-versatile",
        max_loops=1,
        persistent_memory=False,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        tools_list_dictionary=[RESPOND_TOOL],
    )

    agents = [agent1, agent2]

    chat = GroupChat(
        name="Investment Advisory",
        description="Financial, tax, and stock analysis group",
        agents=agents,
        max_loops=10,
        threshold=0.5,
        idle_timeout=8.0,
    )

    history = chat.run(
        "How to save on taxes for stocks, ETFs, and mutual funds?"
    )
    print(history)
