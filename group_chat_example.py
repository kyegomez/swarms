from swarms.structs.agent import Agent
from swarms.structs.groupchat import GroupChat


if __name__ == "__main__":

    # Example agents
    agent1 = Agent(
        agent_name="Financial-Analysis-Agent",
        system_prompt="You are a financial analyst specializing in investment strategies.",
        model_name="gpt-4o",
        max_loops=1,
        dynamic_temperature_enabled=True,
    )

    agent2 = Agent(
        agent_name="Tax-Adviser-Agent",
        system_prompt="You are a tax adviser who provides clear and concise guidance on tax-related queries.",
        model_name="gpt-4o",
        max_loops=1,
        dynamic_temperature_enabled=True,
    )

    agents = [agent1, agent2]

    chat = GroupChat(
        name="Investment Advisory",
        description="Financial and tax analysis group",
        agents=agents,
        max_loops=1,
    )

    history = chat.run(
        "How to optimize tax strategy for investments?"
    )
    print(history)
