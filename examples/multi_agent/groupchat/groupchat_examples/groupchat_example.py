"""Investment advisory dynamic GroupChat using two OpenAI-backed agents."""

import os

from dotenv import load_dotenv

from swarms.structs.agent import Agent
from swarms.structs.groupchat import GroupChat, RESPOND_TOOL

if __name__ == "__main__":

    load_dotenv()

    os.environ.get("OPENAI_API_KEY")

    agent1 = Agent(
        agent_name="Financial-Analysis-Agent",
        description="You are a financial analyst specializing in investment strategies.",
        model_name="gpt-5.4",
        max_loops=1,
        persistent_memory=False,
        autosave=False,
        dashboard=False,
        verbose=True,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        retry_attempts=1,
        context_length=200000,
        max_tokens=15000,
        tools_list_dictionary=[RESPOND_TOOL],
    )

    agent2 = Agent(
        agent_name="Tax-Adviser-Agent",
        description="You are a tax adviser who provides clear and concise guidance on tax-related queries.",
        model_name="gpt-5.4",
        max_loops=1,
        persistent_memory=False,
        autosave=False,
        dashboard=False,
        verbose=True,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        retry_attempts=1,
        context_length=200000,
        max_tokens=15000,
        tools_list_dictionary=[RESPOND_TOOL],
    )

    agents = [agent1, agent2]

    chat = GroupChat(
        name="Investment Advisory",
        description="Financial and tax analysis group",
        agents=agents,
        max_loops=12,
        threshold=0.5,
        idle_timeout=10.0,
        output_type="all",
    )

    history = chat.run(
        "What are the best Japanese business methodologies to take over a market say like minerals and mining?. I need a 4,000 word report. Work together to write the report."
    )
    print(history)
