"""Crypto tax debate using the dynamic GroupChat."""

from dotenv import load_dotenv

from swarms import Agent
from swarms.structs.groupchat import GroupChat, RESPOND_TOOL

if __name__ == "__main__":

    load_dotenv()

    agent1 = Agent(
        agent_name="Crypto-Tax-Optimization-Agent",
        system_prompt="You are a friendly tax expert specializing in cryptocurrency investments. Provide approachable insights on optimizing tax savings for crypto transactions.",
        model_name="groq/llama-3.1-70b-versatile",
        max_loops=1,
        persistent_memory=False,
        dynamic_temperature_enabled=True,
        user_name="User",
        tools_list_dictionary=[RESPOND_TOOL],
    )

    agent2 = Agent(
        agent_name="Crypto-Investment-Strategies-Agent",
        system_prompt="You are a conversational financial analyst focused on cryptocurrency investments. Offer debatable advice on investment strategies that minimize tax liabilities.",
        model_name="groq/llama-3.1-70b-versatile",
        max_loops=1,
        persistent_memory=False,
        dynamic_temperature_enabled=True,
        user_name="User",
        tools_list_dictionary=[RESPOND_TOOL],
    )

    agents = [agent1, agent2]

    chat = GroupChat(
        name="Crypto Tax Optimization Debate",
        description="Debate on optimizing tax savings for cryptocurrency transactions and investments",
        agents=agents,
        max_loops=10,
        threshold=0.5,
        idle_timeout=8.0,
    )

    history = chat.run(
        "How can one optimize tax savings for cryptocurrency transactions and investments? I bought some Bitcoin and Ethereum last year and want to minimize my tax liabilities this year."
    )
    print(history)
